import numpy as np
import torch
from PIL import Image

class ResizeImageForWan:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {}),
                "max_pixels": ("INT", {"default": 850000, "min": 1, "max": 10000000, "step": 1}),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    FUNCTION = "resize_image"
    RETURN_TYPES = ("IMAGE", "INT", "INT",)
    RETURN_NAMES = ("IMAGE", "width", "height")
    OUTPUT_NODE = True
    CATEGORY = "Mine"

    def resize_image(self, image, max_pixels, prompt=None, extra_pnginfo=None):
        # Ensure the input image is on CPU and convert to numpy array
        image_np = image.cpu().numpy()
        
        # Get original dimensions from first image in batch
        if image_np.ndim == 4:
            orig_height, orig_width = image_np[0].shape[:2]
        else:
            orig_height, orig_width = image_np.shape[:2]
        
        # Calculate original pixel count
        original_pixels = orig_width * orig_height
        
        # Find the appropriate scale factor
        scale_factor = 1.0
        if original_pixels > max_pixels:
            # Start from 1.0 and decrease by 0.01 until we find a suitable scale
            for i in range(100):
                test_scale = 1.0 - (i * 0.01)
                test_width = int(orig_width * test_scale)
                test_height = int(orig_height * test_scale)
                test_pixels = test_width * test_height
                
                if test_pixels <= max_pixels:
                    scale_factor = test_scale
                    break
            
            # If still too large after 99% reduction, calculate exact scale needed
            if scale_factor == 1.0:
                scale_factor = (max_pixels / original_pixels) ** 0.5
        else:
            return (image, orig_width, orig_height)
        
        # Initialize new_width and new_height
        new_width = int(orig_width * scale_factor)
        new_height = int(orig_height * scale_factor)
        
        # Check if the image is in the format [batch, height, width, channel]
        if image_np.ndim == 4:
            # Process each image in the batch
            resized_images = []
            for img in image_np:
                # Convert to PIL Image
                pil_img = Image.fromarray((img * 255).astype(np.uint8))
                # Resize
                resized_pil = pil_img.resize((new_width, new_height), Image.LANCZOS)
                # Convert back to numpy and normalize
                resized_np = np.array(resized_pil).astype(np.float32) / 255.0
                resized_images.append(resized_np)
            
            # Stack the resized images back into a batch
            resized_batch = np.stack(resized_images)
            # Convert to torch tensor
            resized_tensor = torch.from_numpy(resized_batch)
        else:
            # If it's a single image, process it directly
            # Convert to PIL Image
            pil_img = Image.fromarray((image_np * 255).astype(np.uint8))
            # Resize
            resized_pil = pil_img.resize((new_width, new_height), Image.LANCZOS)
            # Convert back to numpy and normalize
            resized_np = np.array(resized_pil).astype(np.float32) / 255.0
            # Add batch dimension if it was originally present
            if image.dim() == 4:
                resized_np = np.expand_dims(resized_np, axis=0)
            # Convert to torch tensor
            resized_tensor = torch.from_numpy(resized_np)

        # Update metadata if needed
        if extra_pnginfo is not None:
            extra_pnginfo["resize_scale_factor"] = scale_factor
            extra_pnginfo["resized_width"] = new_width
            extra_pnginfo["resized_height"] = new_height

        return (resized_tensor, new_width, new_height)