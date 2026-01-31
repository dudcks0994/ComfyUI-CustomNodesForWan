import os
import numpy as np
import torch
import scipy.ndimage

class LoadMaskNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "folder_path": ("STRING", {"default": "", "multiline": False}),
                "obj_id_list": ("STRING", {"default": "-1", "multiline": False}),
                "start_frame": ("INT", {"default": -1, "min": -1, "max": 9999}),
                "load_cap": ("INT", {"default": -1, "min": -1, "max": 9999}),
            }
        }

    RETURN_TYPES = ("MASK",)
    FUNCTION = "load_masks"
    CATEGORY = "Mine"

    @classmethod
    def IS_CHANGED(cls, folder_path, obj_id_list, start_frame, load_cap):
        # 폴더 내 .npy 파일들의 수정 시간을 확인하여 변경 감지
        if not folder_path or not os.path.exists(folder_path):
            return float("NaN")
        
        try:
            mtimes = []
            if os.path.isdir(folder_path):
                for filename in os.listdir(folder_path):
                    if filename.endswith(".npy"):
                        filepath = os.path.join(folder_path, filename)
                        mtimes.append(os.path.getmtime(filepath))
            
            if mtimes:
                # 파일 목록과 수정 시간들의 조합 반환
                return str(sorted(mtimes))
        except:
            pass
        
        # 문제가 있으면 항상 재실행
        return float("NaN")

    def load_masks(self, folder_path, obj_id_list, start_frame, load_cap):
        if not folder_path or not os.path.exists(folder_path):
            # Return empty mask if folder doesn't exist
            empty_mask = torch.zeros((1, 64, 64), dtype=torch.float32)
            return (empty_mask,)
        
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
                return (empty_mask,)
        
        # Find all npy files matching the pattern
        npy_files = []
        if os.path.isdir(folder_path):
            for filename in os.listdir(folder_path):
                if filename.endswith(".npy"):
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
            return (empty_mask,)
        
        # Sort by obj_id to ensure consistent ordering
        npy_files.sort(key=lambda x: x[0])
        
        # Load and combine all masks
        combined_masks = []
        for obj_id, filepath in npy_files:
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
            return (empty_mask,)
        
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
        
        return (result_mask,)


class SaveMaskNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
                "save_path": ("STRING", {"default": "", "multiline": False}),
                "original_frames_count": ("INT", {"default": 0, "min": 0, "max": 9999}),
                "start_index": ("INT", {"default": 0, "min": 0, "max": 9999}),
                "end_index": ("INT", {"default": -1, "min": -1, "max": 9999}),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "save_mask"
    CATEGORY = "Mine"
    OUTPUT_NODE = True

    def save_mask(self, mask, save_path, original_frames_count, start_index, end_index):
        if not save_path:
            print("SaveMaskNode: save_path is empty")
            return {}
        
        # 저장 경로의 디렉토리가 없으면 생성
        save_dir = os.path.dirname(save_path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        
        # mask shape: (input_frames, H, W)
        input_frames = mask.shape[0]
        H, W = mask.shape[1], mask.shape[2]
        
        # original_frames_count가 0이면 입력 마스크 프레임 수 사용
        if original_frames_count == 0:
            original_frames_count = input_frames
        
        # end_index가 -1이면 입력 마스크 기준 마지막 프레임까지
        if end_index == -1:
            end_index = start_index + input_frames - 1
        
        # 인덱스 범위 검증
        start_index = max(0, min(start_index, original_frames_count - 1))
        end_index = max(0, min(end_index, original_frames_count - 1))
        
        # start_index가 end_index보다 크면 swap
        if start_index > end_index:
            start_index, end_index = end_index, start_index
        
        # 결과 마스크 생성 (original_frames_count 크기로 전부 0으로 초기화)
        result_mask = torch.zeros((original_frames_count, H, W), dtype=mask.dtype, device=mask.device)
        
        # start_index ~ end_index 범위에 입력 마스크 복사
        # 입력 마스크의 프레임 수와 인덱스 범위가 다를 수 있으므로 조정
        copy_length = min(end_index - start_index + 1, input_frames)
        result_mask[start_index:start_index + copy_length] = mask[:copy_length]
        
        # numpy로 변환하여 저장
        mask_np = result_mask.cpu().numpy()
        
        # save_path가 .npy로 끝나지 않으면 mask.npy 추가
        if not save_path.endswith(".npy"):
            save_path = os.path.join(save_path, "mask.npy")
        
        np.save(save_path, mask_np)
        print(f"SaveMaskNode: Saved mask to {save_path} (frames {start_index} to {start_index + copy_length - 1} filled, total {original_frames_count} frames)")
        
        return {}


class MaskPaddingNode:
    """
    마스크에 앞뒤로 빈 더미 마스크를 추가하여 저장하는 노드
    - 입력 마스크에서 지정된 수만큼 추출
    - 앞뒤로 빈 마스크 패딩 추가
    - mask_modified.npy로 저장
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
                "save_path": ("STRING", {"default": "", "multiline": False}),
                "front_dummy_count": ("INT", {"default": 0, "min": 0, "max": 9999}),
                "extract_count": ("INT", {"default": -1, "min": -1, "max": 9999}),
                "back_dummy_count": ("INT", {"default": 0, "min": 0, "max": 9999}),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "pad_and_save_mask"
    CATEGORY = "Mine"
    OUTPUT_NODE = True

    def pad_and_save_mask(self, mask, save_path, front_dummy_count, extract_count, back_dummy_count):
        if not save_path:
            print("MaskPaddingNode: save_path is empty")
            return {}
        
        # 저장 경로의 디렉토리가 없으면 생성
        save_dir = os.path.dirname(save_path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        
        # mask shape: (frames, H, W)
        input_frames = mask.shape[0]
        H, W = mask.shape[1], mask.shape[2]
        
        # extract_count가 -1이면 전체 마스크 사용
        if extract_count == -1:
            extract_count = input_frames
        
        # 추출할 프레임 수가 입력 마스크보다 크면 입력 마스크 수로 제한
        extract_count = min(extract_count, input_frames)
        
        # 마스크에서 extract_count만큼 추출
        extracted_mask = mask[:extract_count]
        
        # 앞쪽 빈 마스크 생성
        front_dummy = torch.zeros((front_dummy_count, H, W), dtype=mask.dtype, device=mask.device)
        
        # 뒤쪽 빈 마스크 생성
        back_dummy = torch.zeros((back_dummy_count, H, W), dtype=mask.dtype, device=mask.device)
        
        # 결합: front_dummy + extracted_mask + back_dummy
        result_mask = torch.cat([front_dummy, extracted_mask, back_dummy], dim=0)
        
        # numpy로 변환하고 bool 타입으로 저장 (파일 크기 최소화)
        mask_np = result_mask.cpu().numpy().astype(bool)
        
        # save_path가 .npy로 끝나지 않으면 mask_modified.npy 추가
        if os.path.isdir(save_path):
            save_path = os.path.join(save_path, "mask_modified.npy")
        else:
            save_path = save_path
        
        np.save(save_path, mask_np)
        
        total_frames = front_dummy_count + extract_count + back_dummy_count
        print(f"MaskPaddingNode: Saved to {save_path}")
        print(f"  - Front dummy: {front_dummy_count} frames")
        print(f"  - Extracted: {extract_count} frames (from {input_frames} input frames)")
        print(f"  - Back dummy: {back_dummy_count} frames")
        print(f"  - Total: {total_frames} frames, Shape: ({total_frames}, {H}, {W}), dtype: bool")
        
        return {}


class MaskResizingNode:
    """
    마스크를 지정된 width, height로 리사이즈하는 노드
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
                "width": ("INT", {"default": 512, "min": 1, "max": 8192}),
                "height": ("INT", {"default": 512, "min": 1, "max": 8192}),
            }
        }

    RETURN_TYPES = ("MASK",)
    FUNCTION = "resize_mask"
    CATEGORY = "Mine"

    def resize_mask(self, mask, width, height):
        import torch.nn.functional as F
        
        # mask shape: (frames, H, W) 또는 (H, W)
        if mask.dim() == 2:
            # 2D 마스크인 경우 배치 차원 추가
            mask = mask.unsqueeze(0)
        
        # (frames, H, W) -> (frames, 1, H, W) for interpolate
        mask_4d = mask.unsqueeze(1)
        
        # 리사이즈 수행 (bilinear interpolation)
        resized = F.interpolate(mask_4d, size=(height, width), mode='bilinear', align_corners=False)
        
        # (frames, 1, H, W) -> (frames, H, W)
        resized = resized.squeeze(1)
        
        return (resized,)

class MyGrowMask:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
                "expand": ("INT", {"default": 11, "min": -1024, "max": 1024}),
                "tapered_corners": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "face_mask": ("MASK",),
            }
        }
    
    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "grow_mask"
    CATEGORY = "Mine"

    def grow_mask(self, mask, expand, tapered_corners, face_mask=None):
        c = 0 if tapered_corners else 1
        kernel = np.array([[c, 1, c],
                           [1, 1, 1],
                           [c, 1, c]])
        mask = mask.reshape((-1, mask.shape[-2], mask.shape[-1]))
        out = []
        
        # face_mask 전처리
        if face_mask is not None:
            face_mask = face_mask.reshape((-1, face_mask.shape[-2], face_mask.shape[-1]))
        
        for i, m in enumerate(mask):
            original = m.numpy()
            output = original.copy()
            
            for _ in range(abs(expand)):
                if expand < 0:
                    output = scipy.ndimage.grey_erosion(output, footprint=kernel)
                else:
                    output = scipy.ndimage.grey_dilation(output, footprint=kernel)
            
            # face_mask가 있으면, 확장된 부분에서 face_mask 영역 제거
            if face_mask is not None and expand > 0:
                fm = face_mask[i % len(face_mask)].numpy().astype(np.float32)
                fm = np.clip(fm, 0, 1)  # 0-1 범위로 정규화
                # 확장된 부분 = output - original (새로 확장된 영역)
                expanded_part = np.maximum(output - original, 0)
                # 확장된 부분에서 face_mask와 겹치는 영역 제거
                expanded_part_without_face = expanded_part * (1 - fm)
                # 원본 + 수정된 확장 부분
                output = np.clip(original + expanded_part_without_face, 0, 1)
            
            output = torch.from_numpy(output)
            out.append(output)
        return (torch.stack(out, dim=0),)

from tqdm import tqdm
from comfy import model_management
main_device = model_management.get_torch_device()
offload_device = model_management.unet_offload_device()

class MyBlockifyMask:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "masks": ("MASK",),
                    "block_size": ("INT", {"default": 32, "min": 5, "max": 512, "step": 1, "tooltip": "Size of blocks in pixels (smaller = smaller blocks)"}),
                },
                "optional": {
                    "device": (["cpu", "gpu"], {"default": "cpu", "tooltip": "Device to use for processing"}),
                    "face_mask": ("MASK",),
                }
        }

    RETURN_TYPES = ("MASK", )
    RETURN_NAMES = ("mask",)
    FUNCTION = "process"
    CATEGORY = "Mine"
    DESCRIPTION = "Creates a block mask by dividing the bounding box of each mask into blocks of the specified size and filling in blocks that contain any part of the original mask."

    def process(self, masks, block_size, device="cpu", face_mask=None):

        if masks == None:
            return (None,)
        processing_device = main_device if device == "gpu" else torch.device("cpu")
        
        masks = masks.to(processing_device)
        original_masks = masks.clone()  # 원본 마스크 저장
        batch_size, height, width = masks.shape
        
        # face_mask 전처리
        if face_mask is not None:
            face_mask = face_mask.to(processing_device)
            face_mask = face_mask.reshape((-1, face_mask.shape[-2], face_mask.shape[-1]))
        
        result_masks = torch.zeros_like(masks)
        
        for i in tqdm(range(batch_size), desc="BlockifyMask batch"):
            mask = masks[i]
            
            # Find bounding box efficiently
            mask_bool = mask > 0
            if not mask_bool.any():
                continue
                
            y_indices = torch.nonzero(mask_bool.any(dim=1), as_tuple=True)[0]
            x_indices = torch.nonzero(mask_bool.any(dim=0), as_tuple=True)[0]
            
            if len(y_indices) == 0 or len(x_indices) == 0:
                continue
                
            y_min, y_max = y_indices[0], y_indices[-1]
            x_min, x_max = x_indices[0], x_indices[-1]
            
            bbox_width = x_max - x_min + 1
            bbox_height = y_max - y_min + 1
            
            # Calculate block grid
            w_divisions = max(1, bbox_width // block_size)
            h_divisions = max(1, bbox_height // block_size)
            
            w_slice = bbox_width // w_divisions
            h_slice = bbox_height // h_divisions
            
            # Create coordinate grids only for bbox region
            y_coords = torch.arange(y_min, y_max + 1, device=processing_device).view(-1, 1)
            x_coords = torch.arange(x_min, x_max + 1, device=processing_device).view(1, -1)
            
            # Calculate block indices for bbox region
            w_block_indices = (x_coords - x_min) // w_slice
            h_block_indices = (y_coords - y_min) // h_slice
            
            # Clamp to valid range
            w_block_indices = w_block_indices.clamp(0, w_divisions - 1)
            h_block_indices = h_block_indices.clamp(0, h_divisions - 1)
            
            # Create unique block IDs by combining h and w indices
            block_ids = h_block_indices * w_divisions + w_block_indices
            
            # Get mask region within bbox
            mask_region = mask[y_min:y_max+1, x_min:x_max+1]
            
            # Find which blocks have content using scatter_add
            max_blocks = h_divisions * w_divisions
            block_content = torch.zeros(max_blocks, device=processing_device)
            block_content.scatter_add_(0, block_ids.flatten(), mask_region.flatten())
            
            # Create result for blocks that have content
            has_content = block_content > 0
            block_mask = has_content[block_ids]
            
            # Fill the result
            result_masks[i, y_min:y_max+1, x_min:x_max+1] = block_mask.float()
        
        # face_mask가 있으면, 블록화로 새로 생긴 영역에서 face_mask 부분 제거
        if face_mask is not None:
            for i in range(batch_size):
                fm = face_mask[i % len(face_mask)].float()
                fm = torch.clamp(fm, 0, 1)  # 0-1 범위로 정규화
                original = original_masks[i]
                blockified = result_masks[i]
                
                # 새로 생긴 영역 = 블록화 결과 - 원본 (블록화로 확장된 부분)
                new_area = torch.clamp(blockified - original, 0, 1)
                # 새로 생긴 영역에서 face_mask와 겹치는 부분 제거
                new_area_without_face = new_area * (1 - fm)
                # 최종 = 원본 + 수정된 새로 생긴 영역
                result_masks[i] = torch.clamp(original + new_area_without_face, 0, 1)
        
        return (result_masks.clamp(0, 1),)

class MyMaskSubtractNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
                    "required": {
                        "masks_a": ("MASK",),
                        "masks_b": ("MASK",),
                    }
                }

    CATEGORY = "Mine"

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("MASKS",)

    FUNCTION = "subtract_masks"

    def subtract_masks(self, masks_a, masks_b):
        if masks_b is None:
            return (masks_a,)
        subtracted_masks = torch.clamp(masks_a - masks_b, 0, 255)
        return (subtracted_masks,)