import numpy as np
import torch
from PIL import Image

class WanI2VResizeImage:
    VERTICAL_CROP_OPTIONS = ["crop bottom", "crop top"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {}),
                "max_pixels": ("INT", {"default": 850000, "min": 64, "max": 10000000, "step": 1}),
                "vertical_crop": (
                    cls.VERTICAL_CROP_OPTIONS,
                    {
                        "default": "crop bottom",
                        "tooltip": "max_pixels가 원본보다 클 경우, 8의 배수로 맞추기 위해 세로 방향을 자를 때만 작동합니다.",
                    },
                ),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    FUNCTION = "resize_image"
    RETURN_TYPES = ("IMAGE", "INT", "INT",)
    RETURN_NAMES = ("IMAGE", "width", "height")
    OUTPUT_NODE = True
    CATEGORY = "Mine"

    @staticmethod
    def _floor_to_multiple(value, multiple=8):
        return max(multiple, (int(value) // multiple) * multiple)

    @staticmethod
    def _to_tensor(image_np):
        return torch.from_numpy(image_np.astype(np.float32))

    def _resize_batch(self, image_np, width, height):
        resized_images = []
        for img in image_np:
            pil_img = Image.fromarray((img * 255).clip(0, 255).astype(np.uint8))
            resized_pil = pil_img.resize((width, height), Image.LANCZOS)
            resized_np = np.array(resized_pil).astype(np.float32) / 255.0
            resized_images.append(resized_np)
        return self._to_tensor(np.stack(resized_images))

    def _crop_batch_to_multiple(self, image_np, width, height, vertical_crop):
        orig_height, orig_width = image_np[0].shape[:2]
        left = max(0, (orig_width - width) // 2)
        if vertical_crop == "crop top":
            top = max(0, orig_height - height)
        else:
            top = 0
        cropped = image_np[:, top:top + height, left:left + width, :]
        return self._to_tensor(cropped)

    def resize_image(self, image, max_pixels, vertical_crop="crop bottom", prompt=None, extra_pnginfo=None):
        # Ensure the input image is on CPU and convert to numpy array
        image_np = image.cpu().numpy()
        has_batch = image_np.ndim == 4
        if not has_batch:
            image_np = np.expand_dims(image_np, axis=0)
        
        # Get original dimensions from first image in batch
        orig_height, orig_width = image_np[0].shape[:2]
        
        # Calculate original pixel count
        original_pixels = orig_width * orig_height

        if original_pixels > max_pixels:
            scale_factor = (max_pixels / original_pixels) ** 0.5
            new_width = self._floor_to_multiple(orig_width * scale_factor)
            new_height = self._floor_to_multiple(orig_height * scale_factor)

            # Flooring each axis to a multiple of 8 usually keeps us under max_pixels.
            # Keep a defensive loop for very small or unusual sizes.
            while new_width * new_height > max_pixels and (new_width > 8 or new_height > 8):
                if new_width >= new_height and new_width > 8:
                    new_width -= 8
                elif new_height > 8:
                    new_height -= 8
                else:
                    break
            resized_tensor = self._resize_batch(image_np, new_width, new_height)
            mode = "resize"
        else:
            new_width = self._floor_to_multiple(orig_width)
            new_height = self._floor_to_multiple(orig_height)
            if new_width == orig_width and new_height == orig_height:
                resized_tensor = self._to_tensor(image_np)
            else:
                resized_tensor = self._crop_batch_to_multiple(image_np, new_width, new_height, vertical_crop)
            scale_factor = 1.0
            mode = "crop_to_multiple"

        if not has_batch:
            resized_tensor = resized_tensor.squeeze(0)

        # Update metadata if needed
        if extra_pnginfo is not None:
            extra_pnginfo["resize_mode"] = mode
            extra_pnginfo["resize_scale_factor"] = scale_factor
            extra_pnginfo["resized_width"] = new_width
            extra_pnginfo["resized_height"] = new_height

        return (resized_tensor, new_width, new_height)


# Backward compatible alias for existing workflows.
ResizeImageForWan = WanI2VResizeImage
