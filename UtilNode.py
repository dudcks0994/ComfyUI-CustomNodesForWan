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
            return torch.zeros((1, 64, 64), dtype=torch.float32)
        
        # 1. obj_id_list 파싱
        if obj_id_list.strip() == "-1":
            obj_ids = None
        else:
            try:
                obj_ids = [int(x.strip()) for x in obj_id_list.split(",") if x.strip()]
            except ValueError:
                return torch.zeros((1, 64, 64), dtype=torch.float32)
        
        # 2. 파일 목록 필터링 (경로만 수집)
        npy_files = []
        for filename in os.listdir(folder_path):
            if filename.endswith(".npy") and filename != "face.npy":
                try:
                    prefix = filename.split(".")[0]
                    obj_id = int(prefix)
                    if obj_ids is None or obj_id in obj_ids:
                        npy_files.append((obj_id, os.path.join(folder_path, filename)))
                except (ValueError, IndexError):
                    continue
        
        if not npy_files:
            return torch.zeros((1, 64, 64), dtype=torch.float32)
        
        npy_files.sort(key=lambda x: x[0])
        
        # 3. 필요한 프레임만 로드 및 병합
        combined_masks = []
        
        for _, filepath in npy_files:
            try:
                # mmap_mode를 사용하여 파일 전체 로드를 방지
                mask_data_mmap = np.load(filepath, mmap_mode='r')
                num_frames = mask_data_mmap.shape[0]
                
                # 슬라이싱 범위 계산 (핵심!)
                actual_start = start_frame
                if load_cap > 0:
                    actual_end = min(actual_start + load_cap, num_frames)
                    print(f"actual_start: {actual_start}, actual_end: {actual_end}")
                else:
                    actual_end = num_frames
                
                # 필요한 구간만 메모리에 복사
                mask_part = mask_data_mmap[actual_start:actual_end].copy()
                mask_tensor = torch.from_numpy(mask_part).float()
                
                if mask_tensor.dim() == 2:
                    mask_tensor = mask_tensor.unsqueeze(0)
                
                combined_masks.append(mask_tensor)
                
            except Exception as e:
                print(f"Error loading number mask {filepath}: {e}")
                continue

        if not combined_masks:
            return torch.zeros((1, 64, 64), dtype=torch.float32)

        # 4. 여러 마스크가 있을 경우 Max 연산으로 병합
        # 이미 슬라이싱이 끝난 상태이므로 합치기만 하면 됨
        result_mask = combined_masks[0]
        for i in range(1, len(combined_masks)):
            # 만약 파일마다 프레임 수가 다를 수 있다면 짧은 쪽에 맞추거나 예외처리가 필요함
            # 여기서는 단순히 max 연산을 수행
            current_mask = combined_masks[i]
            
            # 프레임 수가 다를 경우를 대비한 최소 크기 맞춤 (선택 사항)
            min_frames = min(result_mask.shape[0], current_mask.shape[0])
            result_mask = torch.max(result_mask[:min_frames], current_mask[:min_frames])
            
        return result_mask

    def load_batch_images(self, folder_path, start_index, load_cap, sort_by="name", reverse_order=False):
        if not folder_path or not os.path.exists(folder_path):
            empty_image = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
            return (empty_image, 0)
        
        SUPPORTED_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.webp', '.bmp', '.gif', '.tiff', '.tif'}
        
        # 1. 파일 경로만 수집 (메모리 사용 적음)
        image_files = [
            os.path.join(folder_path, f) 
            for f in os.listdir(folder_path) 
            if os.path.splitext(f)[1].lower() in SUPPORTED_EXTENSIONS
        ]
        
        if not image_files:
            return torch.zeros((1, 64, 64, 3), dtype=torch.float32)
        
        # 2. 정렬 수행
        if sort_by == "name":
            image_files.sort(key=lambda x: os.path.basename(x).lower())
        elif sort_by == "date_modified":
            image_files.sort(key=lambda x: os.path.getmtime(x))
        elif sort_by == "date_created":
            image_files.sort(key=lambda x: os.path.getctime(x))
        
        if reverse_order:
            image_files.reverse()
        
        # 3. [핵심] 로드하기 전에 리스트 슬라이싱
        # start_index부터 시작해서 load_cap 개수만큼만 남김
        end_index = None if load_cap <= 0 else (start_index + load_cap)
        image_files = image_files[start_index:end_index]
        
        if not image_files:
            return torch.zeros((1, 64, 64, 3), dtype=torch.float32)

        # 4. 이제 선택된 파일만 실제로 로드 (RAM 절약)
        loaded_images = []
        for filepath in image_files:
            try:
                with Image.open(filepath) as pil_img: # with문 사용으로 메모리 해제 보장
                    if pil_img.mode == 'RGBA':
                        background = Image.new('RGB', pil_img.size, (255, 255, 255))
                        background.paste(pil_img, mask=pil_img.split()[3])
                        pil_img = background
                    elif pil_img.mode != 'RGB':
                        pil_img = pil_img.convert('RGB')
                    
                    img_np = np.array(pil_img).astype(np.float32) / 255.0
                    loaded_images.append(img_np)
            except Exception as e:
                print(f"Error loading {filepath}: {e}")
                continue
        
        # (이후 동일하게 shape 체크 및 stack 진행)
        if not loaded_images:
            return torch.zeros((1, 64, 64, 3), dtype=torch.float32)
        
        first_shape = loaded_images[0].shape
        valid_images = [img for img in loaded_images if img.shape == first_shape]
        
        batch_np = np.stack(valid_images, axis=0)
        return torch.from_numpy(batch_np)
    
    def load_face_mask(self, folder_path, start_frame, load_cap):
        if not folder_path or not os.path.exists(folder_path):
            raise ValueError(f"Face mask folder not found: {folder_path}")
        
        try:
            mask_path = os.path.join(folder_path, "face.npy")
            if not os.path.exists(mask_path):
                return None

            # [핵심] mmap_mode='r'을 사용하여 파일을 메모리에 직접 올리지 않고 연결만 함
            mask_data_mmap = np.load(mask_path, mmap_mode='r')
            
            num_frames = mask_data_mmap.shape[0]
            
            # 슬라이싱 범위 계산
            actual_start = start_frame if (0 <= start_frame < num_frames) else (num_frames - 1 if start_frame >= num_frames else 0)
            
            if load_cap > 0:
                actual_end = min(actual_start + load_cap, num_frames)
            else:
                actual_end = num_frames
                
            # 필요한 부분만 메모리로 복사 (copy()를 호출할 때 실제 RAM 점유)
            mask_data = mask_data_mmap[actual_start:actual_end].copy()
            
            mask_tensor = torch.from_numpy(mask_data).float()
            if mask_tensor.dim() == 2:
                mask_tensor = mask_tensor.unsqueeze(0)
                
            return mask_tensor

        except Exception as e:
            print(f"Error loading face mask: {e}")
            raise ValueError(f"Error loading face mask: {e}")

    def load_mask_image_from_path(self, folder_path, obj_id_list, start_frame, load_cap):
        pose_folder_path = os.path.join(folder_path, "pose")
        face_folder_path = os.path.join(folder_path, "face")
        return self.load_number_masks(folder_path, obj_id_list, start_frame, load_cap), self.load_batch_images(pose_folder_path, start_frame, load_cap), self.load_batch_images(face_folder_path, start_frame, load_cap), self.load_face_mask(folder_path, start_frame, load_cap)