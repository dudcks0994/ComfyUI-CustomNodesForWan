import node_helpers
import comfy.utils
import math
from typing_extensions import override
from comfy_api.latest import ComfyExtension, io
import comfy.model_management
import torch
import nodes

class CustomTextEncodeQwenImageEditNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="CustomTextEncodeQwenImageEditNode",
            category="Mine",
            inputs=[
                io.Clip.Input("clip"),
                io.String.Input("positive_prompt", multiline=True, dynamic_prompts=True, optional=True),
                io.String.Input("negative_prompt", multiline=True, dynamic_prompts=True, optional=True),
                io.Latent.Input("latents", optional=True),
                io.Image.Input("image1", optional=True),
                io.Image.Input("image2", optional=True),
                io.Image.Input("image3", optional=True),
                io.Image.Input("image4", optional=True),
            ],
            outputs=[
                io.Conditioning.Output("Positive"),
                io.Conditioning.Output("Negative"),
            ],
        )

    @classmethod
    def execute(cls, clip, positive_prompt, negative_prompt, latents=None, image1=None, image2=None, image3=None, image4=None) -> io.NodeOutput:
        ref_latents = []
        images = [image1, image2, image3, image4]
        images_vl = []
        llama_template = "<|im_start|>system\nDescribe the key features of the input image (color, shape, size, texture, objects, background), then explain how the user's text instruction should alter or modify the image. Generate a new image that meets the user's requirements while maintaining consistency with the original input where appropriate.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
        image_prompt = ""

        for i, image in enumerate(images):
            if image is not None:
                samples = image.movedim(-1, 1)
                total = int(384 * 384)

                scale_by = math.sqrt(total / (samples.shape[3] * samples.shape[2]))
                width = round(samples.shape[3] * scale_by)
                height = round(samples.shape[2] * scale_by)

                s = comfy.utils.common_upscale(samples, width, height, "area", "disabled")
                images_vl.append(s.movedim(1, -1))

                image_prompt += "Picture {}: <|vision_start|><|image_pad|><|vision_end|>".format(i + 1)

        positive_tokens = clip.tokenize(image_prompt + positive_prompt, images=images_vl, llama_template=llama_template)
        negative_tokens = clip.tokenize(image_prompt + negative_prompt, images=images_vl, llama_template=llama_template)
        positive_conditioning = clip.encode_from_tokens_scheduled(positive_tokens)
        negative_conditioning = clip.encode_from_tokens_scheduled(negative_tokens)
        if len(latents) > 0:
            positive_conditioning = node_helpers.conditioning_set_values(positive_conditioning, {"reference_latents": latents}, append=True)
            negative_conditioning = node_helpers.conditioning_set_values(negative_conditioning, {"reference_latents": latents}, append=True)
        return io.NodeOutput(positive_conditioning, negative_conditioning)
        
class CustomImagesEncodeNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="CustomImagesEncodeNode",
            category="Mine",
            inputs=[
                io.Vae.Input("vae"),
                io.Image.Input("image1", optional=True),
                io.Image.Input("image2", optional=True),
                io.Image.Input("image3", optional=True),
                io.Image.Input("image4", optional=True),
            ],
            outputs=[
                io.Latent.Output("latents"),
                io.Latent.Output("first_latent"),
            ],
        )
    @classmethod
    def execute(cls, vae, image1 = None, image2 = None, image3 = None, image4 = None) -> io.NodeOutput:
        images = [image1, image2, image3, image4]
        ref_latents = []
        for image in images:
            if image is not None:
                samples = image.movedim(-1, 1)
                total = int(1024 * 1024)
                scale_by = math.sqrt(total / (samples.shape[3] * samples.shape[2]))
                width = round(samples.shape[3] * scale_by / 8.0) * 8
                height = round(samples.shape[2] * scale_by / 8.0) * 8

                s = comfy.utils.common_upscale(samples, width, height, "area", "disabled")
                ref_latents.append(vae.encode(s.movedim(1, -1)[:, :, :, :3]))
        return io.NodeOutput(ref_latents, {"samples": ref_latents[0]})