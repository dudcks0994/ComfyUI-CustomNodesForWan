"""
Unified WanAnimateToVideo node that handles long video generation internally
by chunking and avoiding redundant VAE encode/decode cycles.
"""

import nodes
import node_helpers
import torch
import comfy.model_management
import comfy.utils
import comfy.sample
import latent_preview
from typing import Tuple, Optional

class WanAnimateToVideoNative:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "positive": ("CONDITIONING", {"default": None}),
                "negative": ("CONDITIONING", {"default": None}),
                "vae": ("VAE", {"default": None}),
                "width": ("INT", {"default": 832, "min": 16, "max": nodes.MAX_RESOLUTION, "step": 16}),
                "height": ("INT", {"default": 480, "min": 16, "max": nodes.MAX_RESOLUTION, "step": 16}),
                "total_length": ("INT", {"default": 161, "min": 1, "max": nodes.MAX_RESOLUTION, "step": 1}),
                "chunk_length": ("INT", {"default": 81, "min": 1, "max": nodes.MAX_RESOLUTION, "step": 4}),
                "continue_motion_max_frames": ("INT", {"default": 5, "min": 1, "max": nodes.MAX_RESOLUTION, "step": 4}),
                "noise": ("NOISE", {"default": None}),
                "guider": ("GUIDER", {"default": None}),
                "sampler": ("SAMPLER", {"default": None}),
                "sigmas": ("SIGMAS", {"default": None}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096}),
                "clip_vision_output": ("CLIP_VISION_OUTPUT", {"default": None}),
                "reference_image": ("IMAGE", {"default": None}),
                "face_video": ("IMAGE", {"default": None}),
                "pose_video": ("IMAGE", {"default": None}),
                "background_video": ("IMAGE", {"default": None}),
                "character_mask": ("MASK", {"default": None}),
                "max_total_pixels": ("INT", {"default": 720 * 1200 * 81, "min": 1, "max": 99999999, "step": 1, "tooltip": "Maximum total pixels(include frames) for encoding. If the total pixels of the input videos are larger than this value, the encoding will be tiled."}),
            },
            "optional": {
                # "continue_generation_images": ("IMAGE", {"default": None}),
            }
        }
        
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    CATEGORY = "Mine"
    FUNCTION = "execute"

    def _prepare_single_chunk(
        self,
        positive,
        negative,
        vae,
        width: int,
        height: int,
        length: int,
        batch_size: int,
        continue_motion_max_frames: int,
        video_frame_offset: int,
        reference_image_latent,
        face_video: Optional[torch.Tensor] = None,
        pose_video: Optional[torch.Tensor] = None,
        background_video: Optional[torch.Tensor] = None,
        character_mask: Optional[torch.Tensor] = None,
        continue_motion_images: Optional[torch.Tensor] = None,
        max_total_pixels: int = 720 * 1200,
    ) -> Tuple:
        """
        Prepare conditioning and latent for a single chunk.
        """

        latent_length = ((length - 1) // 4) + 1
        latent_width = width // 8
        latent_height = height // 8
        trim_latent = 0
        ref_images_num = 0

        print(f"[WanAnimateToVideoNative] Latent length: {latent_length}, Latent width: {latent_width}, Latent height: {latent_height}")

        if latent_width * 8 != width or latent_height * 8 != height:
            raise ValueError(f"[WanAnimateToVideoNative] width, height must be divisible by 16. width: {width}, height: {height}")

        concat_latent_image = reference_image_latent

        trim_latent += concat_latent_image.shape[2]



        mask = torch.zeros((1, 4, concat_latent_image.shape[2], latent_height, latent_width), device=concat_latent_image.device, dtype=concat_latent_image.dtype)
        print(f"[WanAnimateToVideoNative] Concat latent image shape: {concat_latent_image.shape}")

        mask_refmotion = torch.ones((1, 1, latent_length * 4, latent_height, latent_width), device=mask.device, dtype=mask.dtype)

        if continue_motion_images is not None:
            if continue_motion_images.shape[1] != height or continue_motion_images.shape[2] != width:
                raise ValueError(f"[WanAnimateToVideoNative] Continue motion images are not the same size as generation size.")

            image = torch.ones((length, height, width, continue_motion_images.shape[-1]), device=continue_motion_images.device, dtype=continue_motion_images.dtype) * 0.5
            image[:continue_motion_max_frames] = continue_motion_images[:continue_motion_max_frames]

            ref_motion_latent_length = ((continue_motion_images.shape[0] - 1) // 4) + 1
            # ref_images_num = max(0, ref_motion_latent_length * 4 - 3)
            ref_images_num = continue_motion_images.shape[0]
            mask_refmotion[:, :, :ref_motion_latent_length * 4] = 0.0
            video_frame_offset -= continue_motion_max_frames
        else:
            image = torch.ones((length, height, width, 3), device=concat_latent_image.device, dtype=concat_latent_image.dtype) * 0.5
    
        if background_video is not None:
            if background_video.shape[0] <= video_frame_offset:
                raise ValueError(f"[WanAnimateToVideoNative] Background video is too short. Length: {background_video.shape[0]}, Video frame offset: {video_frame_offset}")
            # Position i in chunk ↔ source frame [video_frame_offset + i] (offset already decremented by ref_images_num).
            # image[:ref_images_num] is filled with continue_motion_images; image[ref_images_num:] must be the BG
            # for source frames [video_frame_offset + ref_images_num .. video_frame_offset + length - 1].
            background_video = background_video[video_frame_offset:]
            background_video = background_video[:length]
            if background_video.shape[0] < length:
                need_frames = length - background_video.shape[0]
                background_video = torch.cat((background_video,) + (background_video[-1:],) * (need_frames), dim=0)
            image[ref_images_num:] = background_video[ref_images_num:length]
            
        if character_mask is not None:
            if character_mask.shape[0] <= video_frame_offset:
                raise ValueError(f"[WanAnimateToVideoNative] Character mask is too short. Length: {character_mask.shape[0]}, Video frame offset: {video_frame_offset}")
            m_chunk = character_mask if character_mask.shape[0] == 1 else character_mask[video_frame_offset:]
            m_chunk = m_chunk[:length]
            if m_chunk.ndim != 3:
                raise ValueError(f"[WanAnimateToVideoNative] Character mask is not 3D. Shape: {m_chunk.shape}")
            if m_chunk.ndim == 3:
                m_chunk = m_chunk.unsqueeze(1)
                m_chunk = m_chunk.movedim(0, 1)
            if m_chunk.ndim == 4:
                m_chunk = m_chunk.unsqueeze(1)

            # Same alignment as background_video: m_chunk[i] is the mask for source frame [video_frame_offset + i].
            # We need mask_refmotion[ref_images_num : mask_refmotion.shape[2]] = m_chunk[ref_images_num : mask_refmotion.shape[2]],
            # so m_chunk must be padded up to mask_refmotion.shape[2] frames.
            if m_chunk.shape[2] < mask_refmotion.shape[2]:
                need_mask_frames = mask_refmotion.shape[2] - m_chunk.shape[2]
                m_chunk = torch.cat((m_chunk,) + (m_chunk[:, :, -1:],) * (need_mask_frames), dim=2)
            m_chunk = comfy.utils.common_upscale(m_chunk[:, :, :mask_refmotion.shape[2]], latent_width, latent_height, "nearest-exact", "center")
            mask_refmotion[:, :, ref_images_num:] = m_chunk[:, :, ref_images_num:mask_refmotion.shape[2]]
        
        full_concat_latent = torch.cat((concat_latent_image, self._vae_encode(vae, image, max_total_pixels)), dim=2)
        mask_refmotion = mask_refmotion.view(1, mask_refmotion.shape[2] // 4, 4, mask_refmotion.shape[3], mask_refmotion.shape[4]).transpose(1, 2)
        full_mask = torch.cat((mask, mask_refmotion), dim=2)
        
        positive = node_helpers.conditioning_set_values(positive, {"concat_latent_image": full_concat_latent, "concat_mask": full_mask})
        negative = node_helpers.conditioning_set_values(negative, {"concat_latent_image": full_concat_latent, "concat_mask": full_mask})

        if pose_video is not None:
            pose_chunk = pose_video[video_frame_offset:] if pose_video.shape[0] > video_frame_offset else None
            pose_chunk = pose_chunk[:length]
            # if continue_motion_images is not None:
            #     dummy_pose_chunk = torch.ones((continue_motion_max_frames, pose_chunk.shape[1], pose_chunk.shape[2], 3), device=pose_chunk.device, dtype=pose_chunk.dtype)
            #     pose_chunk = torch.cat((dummy_pose_chunk, pose_chunk), dim=0)
            if pose_chunk is not None and pose_chunk.shape[0] < length:
                pose_chunk = torch.cat((pose_chunk,) + (pose_chunk[-1:],) * (length - pose_chunk.shape[0]), dim=0)
            if pose_chunk is not None:
                pose_chunk = comfy.utils.common_upscale(pose_chunk[:length].movedim(-1, 1), width, height, "area", "center").movedim(1, -1)
                p_lat = self._vae_encode(vae, pose_chunk, max_total_pixels)
                positive = node_helpers.conditioning_set_values(positive, {"pose_video_latent": p_lat})
                negative = node_helpers.conditioning_set_values(negative, {"pose_video_latent": p_lat})

        if face_video is not None:
            face_chunk = face_video[video_frame_offset:] if face_video.shape[0] > video_frame_offset else None
            face_chunk = face_chunk[:length]
            print(f"[WanAnimateToVideoNative] Face chunk shape: {face_chunk.shape}")
            if face_chunk is not None:
                # if continue_motion_images is not None:
                #     face_chunk = torch.cat((continue_motion_images[:continue_motion_max_frames], face_chunk), dim=0)
                #     print(f"[WanAnimateToVideoNative] Face chunk shape after padding with continue motion images: {face_chunk.shape}")
                if face_chunk.shape[0] < length:
                    need_frames = length - face_chunk.shape[0]
                    face_chunk = torch.cat((face_chunk,) + (face_chunk[-1:],) * (need_frames), dim=0)
                print(f"[WanAnimateToVideoNative] Face chunk shape after padding: {face_chunk.shape}")
                face_chunk = (comfy.utils.common_upscale(face_chunk[:length].movedim(-1, 1), 512, 512, "area", "center")) * 2.0 - 1.0
                face_chunk = face_chunk.movedim(0, 1).unsqueeze(0)
                positive = node_helpers.conditioning_set_values(positive, {"face_video_pixels": face_chunk})
                negative = node_helpers.conditioning_set_values(negative, {"face_video_pixels": face_chunk * 0.0 - 1.0})

        latent = torch.zeros([batch_size, 16, latent_length + trim_latent, latent_height, latent_width], device=comfy.model_management.intermediate_device())
        return (positive, negative, {"samples": latent}, trim_latent)

    def _sample_chunk(
        self,
        noise,
        guider,
        sampler,
        sigmas,
        latent: dict,
    ) -> dict:
        """
        Internal sampling function using SamplerCustomAdvanced pattern.
        
        Args:
            noise: NOISE object with generate_noise() method and seed attribute
            guider: GUIDER object with sample() method and model_patcher attribute
            sampler: SAMPLER object
            sigmas: SIGMAS tensor
            latent: Latent dict with "samples" key
            
        Returns:
            Sampled latent dict
        """
        latent_image = latent["samples"]
        latent = latent.copy()
        latent_image = comfy.sample.fix_empty_latent_channels(guider.model_patcher, latent_image)
        latent["samples"] = latent_image

        noise_mask = None
        if "noise_mask" in latent:
            noise_mask = latent["noise_mask"]

        x0_output = {}
        callback = latent_preview.prepare_callback(guider.model_patcher, sigmas.shape[-1] - 1, x0_output)

        disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED
        samples = guider.sample(
            noise.generate_noise(latent), 
            latent_image, 
            sampler, 
            sigmas, 
            denoise_mask=noise_mask, 
            callback=callback, 
            disable_pbar=disable_pbar, 
            seed=noise.seed
        )
        samples = samples.to(comfy.model_management.intermediate_device())

        out = latent.copy()
        out["samples"] = samples
        return out

    def _calculate_chunk_info(self, total_length: int, chunk_length: int, overlap: int) -> list:
        chunks = []
        cur = 0
        while cur < total_length:
            is_first = (cur == 0)
            chunks.append((chunk_length, is_first))
            cur += (chunk_length if is_first else (chunk_length - overlap))
            if cur >= total_length + (overlap if not is_first else 0): break
        return chunks

    def _vae_decode(self, vae, samples, max_total_pixels):
        print(f"[WanAnimateToVideoNative] decoding Samples shape: {samples.shape}")
        height = samples.shape[3] * 8
        width = samples.shape[4] * 8
        length = samples.shape[2] * 4
        if (height * width) > max_total_pixels:
            tile_size = height if height > width else width
            if tile_size > 1024:
                tile_size = 1024
            print(f"[WanAnimateToVideoNative] Tiled VAE Decoding with tile size: {tile_size}")
            decoded = self._decode_tiled(vae, samples, tile_size=tile_size, temporal_size=length)
        else:
            print(f"[WanAnimateToVideoNative] Non-tiled VAE Decoding")
            decoded = vae.decode(samples)
        if len(decoded.shape) == 5: #Combine batches 할때 안하게 여기서 미리.
            print(f"[WanAnimateToVideoNative] Decoded shape: {decoded.shape}. try to reshape to 3D.")
            decoded = decoded.reshape(-1, decoded.shape[-3], decoded.shape[-2], decoded.shape[-1])
            print(f"[WanAnimateToVideoNative] Reshaped decoded shape: {decoded.shape}")
        return decoded

    def _decode_tiled(self, vae, samples, tile_size, temporal_size, overlap=64, temporal_overlap=8):
        if tile_size < overlap * 4:
            overlap = tile_size // 4
        if temporal_size < temporal_overlap * 2:
            temporal_overlap = temporal_overlap // 2
        temporal_compression = vae.temporal_compression_decode()
        if temporal_compression is not None:
            temporal_size = max(2, temporal_size // temporal_compression)
            temporal_overlap = max(1, min(temporal_size // 2, temporal_overlap // temporal_compression))
        else:
            temporal_size = None
            temporal_overlap = None

        compression = vae.spacial_compression_decode()
        images = vae.decode_tiled(samples, tile_x=tile_size // compression, tile_y=tile_size // compression, overlap=overlap // compression, tile_t=temporal_size, overlap_t=temporal_overlap)
        if len(images.shape) == 5: #Combine batches 할때 안하게 여기서 미리.
            images = images.reshape(-1, images.shape[-3], images.shape[-2], images.shape[-1])
        return images

    def _vae_encode(self, vae, pixels, max_total_pixels):
        height = pixels.shape[1]
        width = pixels.shape[2]
        length = pixels.shape[0]
        if (height * width) > max_total_pixels:
            tile_size = height if height < width else width
            if tile_size > 1072:
                tile_size = 1072
            print(f"[WanAnimateToVideoNative] Tiled VAE Encoding with tile size: {tile_size}")
            return self._encode_tiled(vae, pixels, tile_size=1024, temporal_size=length)
        else:
            print(f"[WanAnimateToVideoNative] Non-tiled VAE Encoding")
            return vae.encode(pixels[:,:,:,:3])

    def _encode_tiled(self, vae, pixels, tile_size, temporal_size, overlap=64, temporal_overlap=8):
        return vae.encode_tiled(pixels[:,:,:,:3], tile_x=tile_size, tile_y=tile_size, overlap=overlap, 
                               tile_t=temporal_size, overlap_t=temporal_overlap)

    def execute(self, positive, negative, vae, width, height, total_length, chunk_length, continue_motion_max_frames, 
                noise, guider, sampler, sigmas, batch_size, clip_vision_output=None, reference_image=None, 
                face_video=None, pose_video=None, background_video=None, character_mask=None, max_total_pixels=720*1200):
        
        if clip_vision_output is not None:
            positive = node_helpers.conditioning_set_values(positive, {"clip_vision_output": clip_vision_output})
            negative = node_helpers.conditioning_set_values(negative, {"clip_vision_output": clip_vision_output})
        
        reference_image = comfy.utils.common_upscale(reference_image.movedim(-1, 1), width, height, "area", "center").movedim(1, -1)

        reference_image_latent = self._vae_encode(vae, reference_image.repeat(4, 1, 1, 1), max_total_pixels)
        
        chunk_info = self._calculate_chunk_info(total_length, chunk_length, continue_motion_max_frames)
        all_output_images, video_frame_offset, prev_cm_images = [], 0, None

        # if continue_generation_images is not None:
        #     prev_cm_images = continue_generation_images[:continue_motion_max_frames]
        
        for chunk_idx, (gen_len, is_first) in enumerate(chunk_info):
            print(f"[WanAnimateToVideo] Chunk {chunk_idx+1}/{len(chunk_info)}, Offset: {video_frame_offset}")
            c_pos, c_neg, c_lat, trim_latent = self._prepare_single_chunk(positive, negative, vae, width, height, gen_len, batch_size, 
                                                          continue_motion_max_frames, video_frame_offset, 
                                                          reference_image_latent, face_video, pose_video, background_video, character_mask, 
                                                          prev_cm_images if not is_first else None, max_total_pixels)
            guider.set_conds(c_pos, c_neg)
            sampled = self._sample_chunk(noise, guider, sampler, sigmas, c_lat)
            
            samples_to_decode = sampled["samples"]
            # # Wan Model logic: first frame is reference, ignore it for video output 정확히는 레이턴트 렝스 1만큼빼주고 디코딩.
            samples_to_decode = samples_to_decode[:, :, trim_latent:]
            
            chunk_images = self._vae_decode(vae, samples_to_decode, max_total_pixels)
            if chunk_idx < len(chunk_info) - 1:
                prev_cm_images = chunk_images[-continue_motion_max_frames:].clone()
            out = chunk_images[continue_motion_max_frames:] if not is_first else chunk_images
            all_output_images.append(out.cpu())
            video_frame_offset += out.shape[0]

        return (torch.cat(all_output_images, dim=0)[:total_length],)
