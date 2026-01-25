import os
from pydoc import classify_class_attrs
import numpy as np
import torch
from PIL import Image

class LoadMaskImageFromPath:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "folder_path": ("STRING", {
                    "default": "", 
                    "multiline": False,
                    "placeholder": "C:/path/to/folder (더블클릭으로 탐색)",
                    "mycustom_path_extensions": ""  # 빈 문자열 = 모든 파일/폴더 표시, 특정 확장자만 보려면 "npy,png,jpg" 형식으로 지정
                }),
                "obj_id_list": ("STRING", {"default": "-1", "multiline": False}),
                "start_frame": ("INT", {"default": -1, "min": -1, "max": 9999}),
                "load_cap": ("INT", {"default": -1, "min": -1, "max": 9999}),
            }
        }

    RETURN_TYPES = ("MASK", "IMAGE", "IMAGE", "MASK",)
    FUNCTION = "load_mask_image_from_path"
    RETURN_NAMES = ("masks", "pose_images", "face_images", "face_masks",)
    CATEGORY = "Mine"

    @classmethod
    def IS_CHANGED(cls, folder_path, obj_id_list, start_frame, load_cap):
        # 폴더 내 파일들의 이름과 수정 시간을 확인하여 변경 감지

        return_string: str = ""

        if not folder_path or not os.path.exists(folder_path):
            return "folder_not_exists"
        
        try:
            if os.path.isdir(folder_path):
                # .npy 파일들 체크
                for filename in os.listdir(folder_path):
                    if filename.endswith(".npy"):
                        filepath = os.path.join(folder_path, filename)
                        return_string += f"{filename}_{os.path.getmtime(filepath)}\n"
                
                # pose 폴더의 이미지 파일들도 체크
                pose_folder = os.path.join(folder_path, "pose")
                if os.path.isdir(pose_folder):
                    return_string += f"pose_folder_path_{os.path.getmtime(pose_folder)}\n"
                face_folder = os.path.join(folder_path, "face")
                if os.path.isdir(face_folder):
                    return_string += f"face_folder_path_{os.path.getmtime(face_folder)}\n"
            
            # 파일 이름과 수정 시간 조합 반환 (파일 추가/삭제/수정 모두 감지)
            return return_string
        except Exception as e:
            # 오류 발생 시에도 일관된 문자열 반환 (NaN 대신)
            return f"error_{folder_path}_{str(e)}"

    @classmethod
    def fingerprint_inputs(cls, folder_path, obj_id_list, start_frame, load_cap):
        # 폴더 내 파일들의 이름과 수정 시간을 확인하여 변경 감지
        if not folder_path or not os.path.exists(folder_path):
            return "folder_not_exists"
        
        try:
            file_info = []  # (파일이름, mtime) 튜플 리스트
            if os.path.isdir(folder_path):
                # .npy 파일들 체크
                for filename in os.listdir(folder_path):
                    if filename.endswith(".npy"):
                        filepath = os.path.join(folder_path, filename)
                        file_info.append((filename, os.path.getmtime(filepath)))
                
                # pose 폴더의 이미지 파일들도 체크
                pose_folder = os.path.join(folder_path, "pose")
                if os.path.isdir(pose_folder):
                    for filename in os.listdir(pose_folder):
                        filepath = os.path.join(pose_folder, filename)
                        if os.path.isfile(filepath):
                            file_info.append((f"pose/{filename}", os.path.getmtime(filepath)))
            
            # 파일 이름과 수정 시간 조합 반환 (파일 추가/삭제/수정 모두 감지)
            return f"{folder_path}_{obj_id_list}_{start_frame}_{load_cap}_{str(sorted(file_info))}"
        except Exception as e:
            # 오류 발생 시에도 일관된 문자열 반환 (NaN 대신)
            return f"error_{folder_path}_{str(e)}"

    def load_number_masks(self, folder_path, obj_id_list, start_frame, load_cap):
        if not folder_path or not os.path.exists(folder_path):
            # Return empty mask if folder doesn't exist
            empty_mask = torch.zeros((1, 64, 64), dtype=torch.float32)
            return empty_mask
        
        # Parse obj_id_list
        if obj_id_list.strip() == "-1":
            # Load all npy files
            obj_ids = None
        else:
            # Parse comma-separated numbers
            try:
                obj_ids = [int(x.strip()) for x in obj_id_list.split(",") if x.strip()]
            except ValueError:
                # If parsing fails, return empty mask
                empty_mask = torch.zeros((1, 64, 64), dtype=torch.float32)
                return empty_mask
        
        # Find all npy files matching the pattern
        npy_files = []
        if os.path.isdir(folder_path):
            for filename in os.listdir(folder_path):
                if filename.endswith(".npy") and filename != "face.npy":
                    # Extract the number prefix (e.g., "0" from "0_all_masks.npy")
                    try:
                        prefix = filename.split(".")[0]
                        obj_id = int(prefix)
                        
                        if obj_ids is None or obj_id in obj_ids:
                            filepath = os.path.join(folder_path, filename)
                            npy_files.append((obj_id, filepath))
                    except (ValueError, IndexError):
                        continue
        
        if not npy_files:
            # Return empty mask if no files found
            empty_mask = torch.zeros((1, 64, 64), dtype=torch.float32)
            return empty_mask
        
        # Sort by obj_id to ensure consistent ordering
        npy_files.sort(key=lambda x: x[0])
        
        # Load and combine all masks
        combined_masks = []
        for obj_id, filepath in npy_files:
            print(f"Loading number mask from {filepath}")
            try:
                mask_data = np.load(filepath)
                # Convert to torch tensor if needed
                if isinstance(mask_data, np.ndarray):
                    mask_tensor = torch.from_numpy(mask_data).float()
                    # Ensure proper shape: if 2D, add batch dimension; if already 3D, keep as is
                    if mask_tensor.dim() == 2:
                        mask_tensor = mask_tensor.unsqueeze(0)
                    combined_masks.append(mask_tensor)
            except Exception as e:
                print(f"Error loading mask from {filepath}: {e}")
                continue
        
        if not combined_masks:
            # Return empty mask if no masks were successfully loaded
            empty_mask = torch.zeros((1, 64, 64), dtype=torch.float32)
            return empty_mask
        
        # Combine all masks using max operation (like OR operation)
        # Get the shape from the first mask
        first_mask = combined_masks[0]
        result_mask = torch.zeros_like(first_mask)
        
        if len(combined_masks) == 1:
            # Single mask - return directly
            result_mask = combined_masks[0]
        else:
            # Multiple masks - merge with max operation
            for mask in combined_masks:
                result_mask = torch.max(result_mask, mask)
        
        # start_frame이 -1이 아니면 해당 인덱스부터의 프레임들만 반환
        if start_frame != -1 and result_mask.dim() >= 1:
            num_frames = result_mask.shape[0]
            if start_frame < num_frames:
                result_mask = result_mask[start_frame:]
            else:
                # start_frame이 범위를 벗어나면 마지막 프레임만 반환
                result_mask = result_mask[-1:]
        
        # load_cap이 -1이 아니면 해당 갯수만큼만 반환
        if load_cap > 0 and result_mask.dim() >= 1:
            num_frames = result_mask.shape[0]
            if load_cap < num_frames:
                result_mask = result_mask[:load_cap]
        
        return result_mask

    def load_batch_images(self, folder_path, start_index, load_cap, sort_by="name", reverse_order=False):
        if not folder_path or not os.path.exists(folder_path):
            # 폴더가 없으면 빈 이미지 반환
            empty_image = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
            return (empty_image, 0)
        
        # 이미지 파일 목록 수집
        image_files = []

        SUPPORTED_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.webp', '.bmp', '.gif', '.tiff', '.tif'}

        for filename in os.listdir(folder_path):
            ext = os.path.splitext(filename)[1].lower()
            if ext in SUPPORTED_EXTENSIONS:
                filepath = os.path.join(folder_path, filename)
                image_files.append(filepath)
        
        if not image_files:
            # 이미지가 없으면 빈 이미지 반환
            empty_image = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
            return empty_image
        
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
            return empty_image
        
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
            return empty_image
        
        # 모든 이미지의 크기가 같은지 확인
        first_shape = loaded_images[0].shape
        valid_images = [img for img in loaded_images if img.shape == first_shape]
        
        if len(valid_images) != len(loaded_images):
            print(f"LoadBatchImage: Warning - {len(loaded_images) - len(valid_images)} images had different sizes and were skipped")
        
        if not valid_images:
            empty_image = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
            return empty_image
        
        # 배치로 스택
        batch_np = np.stack(valid_images, axis=0)
        batch_tensor = torch.from_numpy(batch_np)
        
        print(f"LoadBatchImage: Loaded {len(valid_images)} images from {folder_path}")
        
        return batch_tensor
    
    def load_face_mask(self, folder_path, start_frame, load_cap):
        if not folder_path or not os.path.exists(folder_path):
            # raise error for notification
            raise ValueError(f"Face mask folder not found: {folder_path}")
        try:
            if not os.path.exists(os.path.join(folder_path, "face.npy")):
                return None
            mask_data = np.load(os.path.join(folder_path, "face.npy"))
            # Convert to torch tensor if needed
            if isinstance(mask_data, np.ndarray):
                mask_tensor = torch.from_numpy(mask_data).float()
                # Ensure proper shape: if 2D, add batch dimension; if already 3D, keep as is
                if mask_tensor.dim() == 2:
                    mask_tensor = mask_tensor.unsqueeze(0)
                # start_frame이 -1이 아니면 해당 인덱스부터의 프레임들만 반환
                if start_frame != -1 and mask_tensor.dim() >= 1:
                    num_frames = mask_tensor.shape[0]
                    if start_frame < num_frames:
                        mask_tensor = mask_tensor[start_frame:]
                    else:
                        # start_frame이 범위를 벗어나면 마지막 프레임만 반환
                        mask_tensor = mask_tensor[-1:]
                # load_cap이 -1이 아니면 해당 갯수만큼만 반환
                if load_cap > 0 and mask_tensor.dim() >= 1:
                    num_frames = mask_tensor.shape[0]
                    if load_cap < num_frames:
                        mask_tensor = mask_tensor[:load_cap]
                return mask_tensor
        except Exception as e:
            print(f"Error loading face mask from {folder_path}: {e}")
            raise ValueError(f"Error loading face mask from {folder_path}: {e}")

    def load_mask_image_from_path(self, folder_path, obj_id_list, start_frame, load_cap):
        pose_folder_path = os.path.join(folder_path, "pose")
        face_folder_path = os.path.join(folder_path, "face")
        return self.load_number_masks(folder_path, obj_id_list, start_frame, load_cap), self.load_batch_images(pose_folder_path, start_frame, load_cap), self.load_batch_images(face_folder_path, start_frame, load_cap), self.load_face_mask(folder_path, start_frame, load_cap)