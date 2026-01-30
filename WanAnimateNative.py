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
    """
    Unified node for generating long videos with WAN models.
    
    Internally handles chunked generation with optimizations:
    - Reference image is encoded only once (shared across all chunks)
    - Latents are passed directly between chunks for continue_motion (no redundant encode)
    - Each chunk is decoded individually and trimmed in IMAGE space to avoid temporal artifacts
    
    Frame generation math:
    - First chunk: generates `chunk_length` frames
    - Subsequent chunks: generates `chunk_length` frames but `continue_motion_max_frames` overlap
      so effectively adds `chunk_length - continue_motion_max_frames` new frames per chunk
    
    For total_length frames:
    - If total_length <= chunk_length: 1 chunk
    - Otherwise: 1 + ceil((total_length - chunk_length) / (chunk_length - continue_motion_max_frames)) chunks
    
    Example: total_length=157, chunk_length=81, continue_motion_max_frames=5
    - Chunk 1: 81 frames (output: 81 frames)
    - Chunk 2: 81 frames with 5 frame overlap (output: 76 frames after trimming)
    - Total: 81 + 76 = 157 frames
    """

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
                "max_total_pixels": ("INT", {"default": 720 * 1200, "min": 1, "max": 99999999, "step": 1, "tooltip": "Maximum total pixels for encoding. If the total pixels of the input videos are larger than this value, the encoding will be tiled."}),
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
        reference_latent: Optional[torch.Tensor] = None,
        face_video: Optional[torch.Tensor] = None,
        pose_video: Optional[torch.Tensor] = None,
        background_video: Optional[torch.Tensor] = None,
        character_mask: Optional[torch.Tensor] = None,
        continue_motion_images: Optional[torch.Tensor] = None,
        max_total_pixels: int = 720 * 1200,
    ) -> Tuple:
        """
        Prepare conditioning and latent for a single chunk.
        
        Returns: (positive, negative, latent_dict, trim_latent, trim_image, new_video_frame_offset)
        """
        trim_to_pose_video = False
        latent_length = ((length - 1) // 4) + 1
        latent_width = width // 8
        latent_height = height // 8
        trim_latent = 0

        # Use pre-encoded reference latent (optimization: encode only once)
        concat_latent_image = reference_latent
        mask = torch.zeros(
            (1, 4, concat_latent_image.shape[-3], concat_latent_image.shape[-2], concat_latent_image.shape[-1]),
            device=concat_latent_image.device, dtype=concat_latent_image.dtype
        )
        trim_latent += concat_latent_image.shape[2]
        ref_motion_latent_length = 0

        # Handle continue_motion (using pre-decoded images, same as original node)
        if continue_motion_images is not None:
            # Use the last continue_motion_max_frames from the decoded images
            continue_motion = continue_motion_images[-continue_motion_max_frames:]
            
            # CRITICAL: Adjust video_frame_offset like original code does!
            video_frame_offset -= continue_motion.shape[0]
            video_frame_offset = max(0, video_frame_offset)
            
            # Upscale and place in image tensor (same as original)
            continue_motion = comfy.utils.common_upscale(
                continue_motion[:length].movedim(-1, 1), width, height, "area", "center"
            ).movedim(1, -1)
            image = torch.ones(
                (length, height, width, continue_motion.shape[-1]),
                device=continue_motion.device, dtype=continue_motion.dtype
            ) * 0.5
            image[:continue_motion.shape[0]] = continue_motion
            ref_motion_latent_length = ((continue_motion.shape[0] - 1) // 4) + 1
        else:
            image = torch.ones((length, height, width, 3)) * 0.5
            ref_motion_latent_length = 0

        # Handle pose_video
        if pose_video is not None:
            if pose_video.shape[0] <= video_frame_offset:
                pose_video = None
            else:
                pose_video = pose_video[video_frame_offset:]

        if pose_video is not None:
            pose_video = comfy.utils.common_upscale(
                pose_video[:length].movedim(-1, 1), width, height, "area", "center"
            ).movedim(1, -1)
            if not trim_to_pose_video:
                if pose_video.shape[0] < length:
                    pose_video = torch.cat(
                        (pose_video,) + (pose_video[-1:],) * (length - pose_video.shape[0]), dim=0
                    )
            
            pose_video = pose_video[:, :, :, :3]

            pose_height = pose_video.shape[1]
            pose_width = pose_video.shape[2]
            total_pixels = pose_height * pose_width

            print(f"pose_video before encode")
            if total_pixels > max_total_pixels:
                print(f"[WanAnimateToVideoNative] Encoding tiled for pose_video")
                pose_video_latent = self._encode_tiled(vae, pose_video[:, :, :, :3], tile_size=1024, overlap=128, temporal_size=128, temporal_overlap=16)
            else:
                print(f"[WanAnimateToVideoNative] Encoding normal for pose_video")
                pose_video_latent = vae.encode(pose_video[:, :, :, :3])
            print(f"pose_video_latent encode done: {pose_video_latent.shape}")

            positive = node_helpers.conditioning_set_values(positive, {"pose_video_latent": pose_video_latent})
            negative = node_helpers.conditioning_set_values(negative, {"pose_video_latent": pose_video_latent})

            if trim_to_pose_video:
                latent_length = pose_video_latent.shape[2]
                length = latent_length * 4 - 3
                image = image[:length]

        # Handle face_video
        if face_video is not None:
            if face_video.shape[0] <= video_frame_offset:
                face_video = None
            else:
                face_video = face_video[video_frame_offset:]

        if face_video is not None:
            face_video = comfy.utils.common_upscale(
                face_video[:length].movedim(-1, 1), 512, 512, "area", "center"
            ) * 2.0 - 1.0
            face_video = face_video.movedim(0, 1).unsqueeze(0)
            positive = node_helpers.conditioning_set_values(positive, {"face_video_pixels": face_video})
            negative = node_helpers.conditioning_set_values(negative, {"face_video_pixels": face_video * 0.0 - 1.0})

        ref_images_num = max(0, ref_motion_latent_length * 4 - 3)
        
        # Handle background_video
        if background_video is not None:
            if background_video.shape[0] > video_frame_offset:
                background_video = background_video[video_frame_offset:]
                background_video = comfy.utils.common_upscale(
                    background_video[:length].movedim(-1, 1), width, height, "area", "center"
                ).movedim(1, -1)
                print(f"[DEBUG] background_video after upscale: {background_video.shape}, ref_images_num: {ref_images_num}, length: {length}")
                if background_video.shape[0] > ref_images_num:
                    image[ref_images_num:background_video.shape[0]] = background_video[ref_images_num:]
                    print(f"[DEBUG] image filled from {ref_images_num} to {background_video.shape[0]}")

        mask_refmotion = torch.ones(
            (1, 1, latent_length * 4, concat_latent_image.shape[-2], concat_latent_image.shape[-1]),
            device=mask.device, dtype=mask.dtype
        )
        print(f"[DEBUG] mask_refmotion initial shape: {mask_refmotion.shape}, latent_length: {latent_length}, latent_length*4: {latent_length * 4}")
        if ref_motion_latent_length > 0:
            mask_refmotion[:, :, :ref_motion_latent_length * 4] = 0.0

        # Handle character_mask
        if character_mask is not None:
            if character_mask.shape[0] > video_frame_offset or character_mask.shape[0] == 1:
                if character_mask.shape[0] == 1:
                    character_mask = character_mask.repeat((length,) + (1,) * (character_mask.ndim - 1))
                else:
                    character_mask = character_mask[video_frame_offset:]
                if character_mask.ndim == 3:
                    character_mask = character_mask.unsqueeze(1)
                    character_mask = character_mask.movedim(0, 1)
                if character_mask.ndim == 4:
                    character_mask = character_mask.unsqueeze(1)
                character_mask = comfy.utils.common_upscale(
                    character_mask[:, :, :length],
                    concat_latent_image.shape[-1], concat_latent_image.shape[-2],
                    "nearest-exact", "center"
                )
                print(f"[DEBUG] character_mask after upscale: {character_mask.shape}")
                # Extend character_mask to match latent_length * 4 by repeating last frame
                mask_temporal_size = latent_length * 4
                if character_mask.shape[2] < mask_temporal_size:
                    padding_size = mask_temporal_size - character_mask.shape[2]
                    last_frame = character_mask[:, :, -1:, :, :]
                    padding = last_frame.repeat(1, 1, padding_size, 1, 1)
                    character_mask = torch.cat([character_mask, padding], dim=2)
                    print(f"[DEBUG] character_mask extended to: {character_mask.shape} (padded {padding_size} frames)")
                
                if character_mask.shape[2] > ref_images_num:
                    mask_refmotion[:, :, ref_images_num:character_mask.shape[2]] = character_mask[:, :, ref_images_num:]
                    print(f"[DEBUG] mask_refmotion filled from {ref_images_num} to {character_mask.shape[2]}")

        # Build concat_latent_image by encoding the full image
        # This ensures temporal coherence across continue_motion boundary
        image_height = image.shape[1]
        image_width = image.shape[2]
        total_pixels = image_height * image_width
        
        print(f"[DEBUG] image shape before encode: {image.shape}")
        if total_pixels > max_total_pixels:
            print(f"[WanAnimateToVideoNative] Encoding tiled for image")
            image_latent = self._encode_tiled(vae, image[:, :, :, :3], tile_size=1024, overlap=128, temporal_size=128, temporal_overlap=16)
        else:
            print(f"[WanAnimateToVideoNative] Encoding normal for image")
            image_latent = vae.encode(image[:, :, :, :3])
        
        print(f"[DEBUG] image_latent shape after encode: {image_latent.shape}")
        concat_latent_image = torch.cat((concat_latent_image, image_latent), dim=2)

        mask_refmotion = mask_refmotion.view(
            1, mask_refmotion.shape[2] // 4, 4, mask_refmotion.shape[3], mask_refmotion.shape[4]
        ).transpose(1, 2)
        mask = torch.cat((mask, mask_refmotion), dim=2)
        
        positive = node_helpers.conditioning_set_values(
            positive, {"concat_latent_image": concat_latent_image, "concat_mask": mask}
        )
        negative = node_helpers.conditioning_set_values(
            negative, {"concat_latent_image": concat_latent_image, "concat_mask": mask}
        )

        latent = torch.zeros(
            [batch_size, 16, latent_length + trim_latent, latent_height, latent_width],
            device=comfy.model_management.intermediate_device()
        )
        out_latent = {"samples": latent}
        
        return (
            positive,
            negative,
            out_latent,
            trim_latent,
            max(0, ref_motion_latent_length * 4 - 3),
            video_frame_offset + length
        )

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

    def _calculate_chunk_info(self, total_length: int, chunk_length: int, continue_motion_max_frames: int) -> list:
        """
        Calculate chunk information for the given total length.
        """
        if total_length <= chunk_length:
            return [(total_length, total_length, True)]
        
        new_frames_per_chunk = chunk_length - continue_motion_max_frames
        
        chunks = [(chunk_length, chunk_length, True)]  # First chunk: output all frames
        remaining = total_length - chunk_length
        
        while remaining > 0:
            if remaining <= new_frames_per_chunk:
                chunks.append((chunk_length, remaining, False))
                remaining = 0
            else:
                chunks.append((chunk_length, new_frames_per_chunk, False))
                remaining -= new_frames_per_chunk
        
        return chunks

    def _decode_tiled(self, vae, samples, tile_size, overlap=64, temporal_size=64, temporal_overlap=8):
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
        return images

    def _encode_tiled(self, vae, pixels, tile_size, overlap, temporal_size=64, temporal_overlap=8):
        t = vae.encode_tiled(pixels[:,:,:,:3], tile_x=tile_size, tile_y=tile_size, overlap=overlap, tile_t=temporal_size, overlap_t=temporal_overlap)
        return t

    def execute(
        self,
        positive,
        negative,
        vae,
        width: int,
        height: int,
        total_length: int,
        chunk_length: int,
        continue_motion_max_frames: int,
        noise,
        guider,
        sampler,
        sigmas,
        batch_size: int,
        clip_vision_output=None,
        reference_image=None,
        face_video=None,
        pose_video=None,
        background_video=None,
        character_mask=None,
        max_total_pixels: int = 720 * 1200,
    ):
        """
        Execute unified video generation.
        """
        # Calculate chunks
        chunk_info = self._calculate_chunk_info(total_length, chunk_length, continue_motion_max_frames)
        
        # Pre-compute clip_vision_output into conditioning once
        if clip_vision_output is not None:
            positive = node_helpers.conditioning_set_values(positive, {"clip_vision_output": clip_vision_output})
            negative = node_helpers.conditioning_set_values(negative, {"clip_vision_output": clip_vision_output})
        
        # Pre-encode reference_image once (optimization: avoid redundant VAE encode)
        if reference_image is None:
            reference_image = torch.zeros((1, height, width, 3))
        ref_image_processed = comfy.utils.common_upscale(
            reference_image[:chunk_length].movedim(-1, 1), width, height, "area", "center"
        ).movedim(1, -1)
        image_input = ref_image_processed[:, :, :, :3]

        image_height = image_input.shape[1]
        image_width = image_input.shape[2]
        total_pixels = image_height * image_width
        
        if total_pixels > max_total_pixels:
            print(f"[WanAnimateToVideoNative] Encoding tiled for reference_image")
            reference_latent = self._encode_tiled(vae, image_input, tile_size=1024, overlap=128, temporal_size=128, temporal_overlap=16)
        else:
            print(f"[WanAnimateToVideoNative] Encoding normal for reference_image")
            reference_latent = vae.encode(image_input)
        
        # Memory optimization: pre-allocate final tensor instead of using list + cat
        # Calculate total output frames
        total_output_frames = sum(output_frames for _, output_frames, _ in chunk_info)
        final_images = None  # Will be allocated on first chunk (to get correct dtype/device)
        current_frame_idx = 0
        
        video_frame_offset = 0
        # Store the last N decoded IMAGE frames for continue_motion (same as original node behavior)
        prev_continue_motion_images = None
        
        for chunk_idx, (generation_length, output_frames, is_first) in enumerate(chunk_info):
            print(f"[WanAnimateToVideoUnified] Processing chunk {chunk_idx + 1}/{len(chunk_info)}, "
                  f"gen_length={generation_length}, output_frames={output_frames}, is_first={is_first}, "
                  f"video_frame_offset={video_frame_offset}")
            
            # Prepare chunk - always use chunk_length for consistent model behavior
            chunk_positive, chunk_negative, chunk_latent, trim_latent, trim_image, video_frame_offset = \
                self._prepare_single_chunk(
                    positive=positive,
                    negative=negative,
                    vae=vae,
                    width=width,
                    height=height,
                    length=generation_length,  # Always chunk_length for consistent quality
                    batch_size=batch_size,
                    continue_motion_max_frames=continue_motion_max_frames,
                    video_frame_offset=video_frame_offset,
                    reference_latent=reference_latent,  # Pre-encoded latent
                    face_video=face_video,
                    pose_video=pose_video,
                    background_video=background_video,
                    character_mask=character_mask,
                    continue_motion_images=prev_continue_motion_images if not is_first else None,
                    max_total_pixels=max_total_pixels,
                )
            
            # Update guider conditioning in-place (no recreation)
            guider.set_conds(chunk_positive, chunk_negative)
            
            # Sample using SamplerCustomAdvanced pattern (same seed for all chunks)
            sampled_latent = self._sample_chunk(
                noise=noise,
                guider=guider,
                sampler=sampler,
                sigmas=sigmas,
                latent=chunk_latent,
            )
            
            # Trim latent: remove reference image part (in latent space)
            trimmed_samples = sampled_latent["samples"][:, :, trim_latent:]

            h_latent = trimmed_samples.shape[-2] * 8
            w_latent = trimmed_samples.shape[-1] * 8
            expected_total_pixels = h_latent * w_latent

            if expected_total_pixels > max_total_pixels:
                print(f"[WanAnimateToVideoUnified] Decoding tiled for chunk {chunk_idx + 1}")
                chunk_images = self._decode_tiled(vae, trimmed_samples, tile_size=1024, overlap=128, temporal_size=128, temporal_overlap=16)
            else:
                print(f"[WanAnimateToVideoUnified] Decoding normal for chunk {chunk_idx + 1}")
                chunk_images = vae.decode(trimmed_samples)
            
            print(f"[DEBUG] trimmed_samples (latent) shape: {trimmed_samples.shape}")
            
            # Free trimmed_samples memory immediately after decode (no longer needed)
            del trimmed_samples
            
            if len(chunk_images.shape) == 5:  # Combine batches if needed
                chunk_images = chunk_images.reshape(-1, chunk_images.shape[-3], chunk_images.shape[-2], chunk_images.shape[-1])
            
            print(f"[DEBUG] chunk_images (decoded) shape: {chunk_images.shape}, output_frames: {output_frames}, trim_image: {trim_image}")
            
            # Store last N frames for continue_motion (same as original node!)
            if chunk_idx < len(chunk_info) - 1:  # Not the last chunk
                prev_continue_motion_images = chunk_images[-continue_motion_max_frames:].clone()
            
            # Trim overlap in IMAGE space (not latent space!) - this matches original behavior
            if is_first:
                output_images = chunk_images[:output_frames]
            else:
                output_images = chunk_images[trim_image:trim_image + output_frames]
            
            # Free chunk_images if different from output_images
            if output_images.data_ptr() != chunk_images.data_ptr():
                del chunk_images
            
            # Allocate final tensor on first chunk (now we know dtype and device)
            if final_images is None:
                final_images = torch.empty(
                    (total_output_frames, output_images.shape[1], output_images.shape[2], output_images.shape[3]),
                    dtype=output_images.dtype,
                    device=output_images.device
                )
            
            # Copy directly to pre-allocated tensor (avoid list + cat overhead)
            final_images[current_frame_idx:current_frame_idx + output_frames] = output_images
            current_frame_idx += output_frames
            
            # Free output_images
            del output_images
            
            print(f"[WanAnimateToVideoUnified] Chunk {chunk_idx + 1} frames written: {output_frames} "
                  f"(total so far: {current_frame_idx}/{total_output_frames})")
        
        print(f"[WanAnimateToVideoUnified] Total generated frames: {final_images.shape[0]}")
        
        return (final_images,)
