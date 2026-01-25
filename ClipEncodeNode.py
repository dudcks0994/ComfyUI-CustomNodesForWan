import hashlib
import os
import torch
import gc
import comfy.model_management as mm
import folder_paths
import comfy.sd

class CustomCLIPTextEncodeNode:
    """
    Receive Clip model and have input of text and output the encoded latent(conditioning).
    always encode text and unload the clip model after encoding.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip_name": (folder_paths.get_filename_list("text_encoders"), ),
                "type": (["stable_diffusion", "stable_cascade", "sd3", "stable_audio", "mochi", "ltxv", "pixart", "cosmos", "lumina2", "wan", "hidream", "chroma", "ace", "omnigen2", "qwen_image", "hunyuan_image"], ),         
                "positive_prompt": ("STRING", {"tooltip": "The text to be encoded."}),
                "negative_prompt": ("STRING", {"tooltip": "The text to be encoded."}),
                "use_disk_cache": ("BOOLEAN", {"default": True, "tooltip": "Cache the text embeddings to disk for faster re-use, under the custom_nodes/MyCustomNodes/text_embed_cache directory"}),
            },
            "optional": {
                "device": (["default", "cpu"], {"advanced": True}),
            }
        }
        
    RETURN_TYPES = ("CONDITIONING", "CONDITIONING")
    RETURN_NAMES = ("conditioning", "conditioning")
    OUTPUT_TOOLTIPS = ("A conditioning containing the embedded text used to guide the diffusion model.", "A conditioning containing the embedded text used to guide the diffusion model.")
    CATEGORY = "Mine"
    FUNCTION = "encode"

    def _generate_cache_filename(self, text):
        return hashlib.sha256(text.encode('utf-8')).hexdigest()

    def _clip_encode(self, clip, text):
        tokens = clip.tokenize(text)
        return clip.encode_from_tokens_scheduled(tokens)

    def encode(self, clip_name, type, positive_prompt, negative_prompt, use_disk_cache=True, device="default"):
        positive_cache_filename = self._generate_cache_filename(positive_prompt)
        negative_cache_filename = self._generate_cache_filename(negative_prompt)

        cache_dir = os.path.join(os.path.dirname(__file__), 'text_embed_cache')
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

        positive_cache_path = os.path.join(cache_dir, positive_cache_filename)
        negative_cache_path = os.path.join(cache_dir, negative_cache_filename)

        if use_disk_cache:
            if os.path.exists(positive_cache_path) and os.path.exists(negative_cache_path):
                positive_conditioning = torch.load(positive_cache_path)
                negative_conditioning = torch.load(negative_cache_path)
                return (positive_conditioning, negative_conditioning)

        clip_type = getattr(comfy.sd.CLIPType, type.upper(), comfy.sd.CLIPType.STABLE_DIFFUSION)
        clip_path = folder_paths.get_full_path_or_raise("text_encoders", clip_name)
        model_options = {}
        if device == "cpu":
            model_options["load_device"] = model_options["offload_device"] = torch.device("cpu")
        clip = comfy.sd.load_clip(ckpt_paths=[clip_path], embedding_directory=folder_paths.get_folder_paths("embeddings"), clip_type=clip_type, model_options=model_options)
        positive_conditioning = self._clip_encode(clip, positive_prompt)
        negative_conditioning = self._clip_encode(clip, negative_prompt)
        if use_disk_cache:
            torch.save(positive_conditioning, positive_cache_path)
            print(f"Saved positive conditioning to cache: {positive_cache_path}")
            torch.save(negative_conditioning, negative_cache_path)
            print(f"Saved negative conditioning to cache: {negative_cache_path}")
        del clip
        mm.soft_empty_cache()
        gc.collect()

        return (positive_conditioning, negative_conditioning)