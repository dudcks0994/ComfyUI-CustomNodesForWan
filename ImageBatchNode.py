import os
import numpy as np
import torch
from PIL import Image
import folder_paths
from datetime import datetime


class BatchImageSave:
    """
    배치 이미지들을 지정한 폴더에 저장하는 노드
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "folder_path": ("STRING", {"default": "", "multiline": False}),
                "filename_prefix": ("STRING", {"default": "image"}),
                "format": (["png", "jpg", "webp"], {"default": "png"}),
                "quality": ("INT", {"default": 95, "min": 1, "max": 100}),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "save_batch_images"
    CATEGORY = "Mine"
    OUTPUT_NODE = True

    def save_batch_images(self, images, folder_path, filename_prefix, format, quality):
        # 폴더 경로가 비어있으면 output 폴더 사용
        if not folder_path:
            save_dir = folder_paths.get_output_directory()
        else:
            save_dir = folder_path
        
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        
        # images shape: (batch, H, W, C)
        batch_size = images.shape[0]
        
        for i in range(batch_size):
            # 개별 이미지 추출
            img_tensor = images[i]
            
            # torch tensor를 numpy로 변환 (0-255 범위로)
            img_np = img_tensor.cpu().numpy()
            img_np = (img_np * 255).clip(0, 255).astype(np.uint8)
            
            # PIL Image로 변환
            pil_img = Image.fromarray(img_np)
            
            # 파일명 생성
            filename = f"{filename_prefix}_{i:05d}.{format}"
            filepath = os.path.join(save_dir, filename)
            
            # 이미지 저장
            if format == "jpg":
                pil_img.save(filepath, "JPEG", quality=quality)
            elif format == "webp":
                pil_img.save(filepath, "WEBP", quality=quality)
            else:  # png
                pil_img.save(filepath, "PNG")
            
            # print(f"BatchImageSave: Saved {filepath}")
        
        print(f"BatchImageSave: Total {batch_size} images saved to {save_dir}")
        
        return {}


class LoadBatchImage:
    """
    지정한 폴더에서 이미지들을 가져와서 배치로 반환하는 노드
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "folder_path": ("STRING", {"default": "", "multiline": False}),
                "start_index": ("INT", {"default": 0, "min": 0, "max": 9999}),
                "load_cap": ("INT", {"default": -1, "min": -1, "max": 9999}),
                "sort_by": (["name", "date_modified", "date_created"], {"default": "name"}),
                "reverse_order": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT",)
    RETURN_NAMES = ("images", "count",)
    FUNCTION = "load_batch_images"
    CATEGORY = "Mine"

    # 지원하는 이미지 확장자
    SUPPORTED_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.webp', '.bmp', '.gif', '.tiff', '.tif'}

    @classmethod
    def IS_CHANGED(cls, folder_path, start_index, load_cap, sort_by, reverse_order):
        # 폴더 내 이미지 파일들의 수정 시간을 확인하여 변경 감지
        if not folder_path or not os.path.exists(folder_path):
            return float("NaN")
        
        try:
            mtimes = []
            if os.path.isdir(folder_path):
                for filename in os.listdir(folder_path):
                    ext = os.path.splitext(filename)[1].lower()
                    if ext in cls.SUPPORTED_EXTENSIONS:
                        filepath = os.path.join(folder_path, filename)
                        mtimes.append(os.path.getmtime(filepath))
            
            if mtimes:
                return str(sorted(mtimes))
        except:
            pass
        
        return float("NaN")

    def load_batch_images(self, folder_path, start_index, load_cap, sort_by, reverse_order):
        if not folder_path or not os.path.exists(folder_path):
            # 폴더가 없으면 빈 이미지 반환
            empty_image = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
            return (empty_image, 0)
        
        # 이미지 파일 목록 수집
        image_files = []
        for filename in os.listdir(folder_path):
            ext = os.path.splitext(filename)[1].lower()
            if ext in self.SUPPORTED_EXTENSIONS:
                filepath = os.path.join(folder_path, filename)
                image_files.append(filepath)
        
        if not image_files:
            # 이미지가 없으면 빈 이미지 반환
            empty_image = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
            return (empty_image, 0)
        
        # 정렬
        if sort_by == "name":
            image_files.sort(key=lambda x: os.path.basename(x).lower())
        elif sort_by == "date_modified":
            image_files.sort(key=lambda x: os.path.getmtime(x))
        elif sort_by == "date_created":
            image_files.sort(key=lambda x: os.path.getctime(x))
        
        # 역순 정렬
        if reverse_order:
            image_files.reverse()
        
        # start_index 적용
        if start_index > 0:
            image_files = image_files[start_index:]
        
        # load_cap 적용
        if load_cap > 0:
            image_files = image_files[:load_cap]
        
        if not image_files:
            empty_image = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
            return (empty_image, 0)
        
        # 이미지 로드
        loaded_images = []
        for filepath in image_files:
            try:
                pil_img = Image.open(filepath)
                
                # RGBA인 경우 RGB로 변환
                if pil_img.mode == 'RGBA':
                    # 알파 채널을 흰색 배경으로 합성
                    background = Image.new('RGB', pil_img.size, (255, 255, 255))
                    background.paste(pil_img, mask=pil_img.split()[3])
                    pil_img = background
                elif pil_img.mode != 'RGB':
                    pil_img = pil_img.convert('RGB')
                
                # numpy 배열로 변환하고 0-1 범위로 정규화
                img_np = np.array(pil_img).astype(np.float32) / 255.0
                loaded_images.append(img_np)
                
            except Exception as e:
                print(f"LoadBatchImage: Error loading {filepath}: {e}")
                continue
        
        if not loaded_images:
            empty_image = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
            return (empty_image, 0)
        
        # 모든 이미지의 크기가 같은지 확인
        first_shape = loaded_images[0].shape
        valid_images = [img for img in loaded_images if img.shape == first_shape]
        
        if len(valid_images) != len(loaded_images):
            print(f"LoadBatchImage: Warning - {len(loaded_images) - len(valid_images)} images had different sizes and were skipped")
        
        if not valid_images:
            empty_image = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
            return (empty_image, 0)
        
        # 배치로 스택
        batch_np = np.stack(valid_images, axis=0)
        batch_tensor = torch.from_numpy(batch_np)
        
        print(f"LoadBatchImage: Loaded {len(valid_images)} images from {folder_path}")
        
        return (batch_tensor, len(valid_images))

