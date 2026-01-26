import torch

def estimate_vae_vram(
    height: int,
    width: int,
    frames: int,
    dtype=torch.float16,
    operation: str = "encode"  # "encode" or "decode"
) -> dict:
    """
    VAE 연산에 필요한 VRAM을 예측합니다.
    
    WAN VAE 특성:
    - Spatial compression: 8x (height/8, width/8)
    - Temporal compression: 4x ((frames-1)/4 + 1)
    - Latent channels: 16
    """
    bytes_per_element = 2 if dtype == torch.float16 else 4
    
    # 이미지 텐서 크기 (B, H, W, C)
    image_size = frames * height * width * 3 * bytes_per_element
    
    # 라텐트 텐서 크기 (B, C, T, H, W)
    latent_frames = ((frames - 1) // 4) + 1
    latent_h = height // 8
    latent_w = width // 8
    latent_size = 16 * latent_frames * latent_h * latent_w * bytes_per_element
    
    # VAE 중간 활성화 (경험적 추정치: 입출력의 3-6배)
    # 실제로는 모델 구조에 따라 다름
    activation_multiplier = 4.0
    
    if operation == "encode":
        input_size = image_size
        output_size = latent_size
    else:  # decode
        input_size = latent_size
        output_size = image_size
    
    intermediate_size = max(input_size, output_size) * activation_multiplier
    total_estimated = input_size + output_size + intermediate_size
    
    return {
        "input_gb": input_size / (1024**3),
        "output_gb": output_size / (1024**3),
        "intermediate_gb": intermediate_size / (1024**3),
        "total_estimated_gb": total_estimated / (1024**3),
        "dimensions": {
            "image": f"{frames}x{height}x{width}x3",
            "latent": f"16x{latent_frames}x{latent_h}x{latent_w}"
        }
    }


class VRAMTracker:
    """VRAM 사용량 추적 유틸리티"""
    
    def __init__(self):
        self.logs = []
    
    def start_tracking(self, operation_name: str):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            return {
                "name": operation_name,
                "start_allocated": torch.cuda.memory_allocated(),
                "start_reserved": torch.cuda.memory_reserved()
            }
        return None
    
    def end_tracking(self, context: dict) -> dict:
        if context is None or not torch.cuda.is_available():
            return {}
        
        torch.cuda.synchronize()
        
        result = {
            "name": context["name"],
            "start_gb": context["start_allocated"] / (1024**3),
            "end_gb": torch.cuda.memory_allocated() / (1024**3),
            "peak_gb": torch.cuda.max_memory_allocated() / (1024**3),
            "delta_gb": (torch.cuda.memory_allocated() - context["start_allocated"]) / (1024**3)
        }
        self.logs.append(result)
        print(f"[VRAM-{result['name']}] Peak: {result['peak_gb']:.3f}GB, Delta: {result['delta_gb']:.3f}GB")
        return result
    
    def get_summary(self) -> str:
        if not self.logs:
            return "No VRAM tracking data"
        
        total_peak = max(log["peak_gb"] for log in self.logs)
        return f"Total operations: {len(self.logs)}, Max peak: {total_peak:.3f}GB"

if __name__ == "__main__":
    print(estimate_vae_vram(1920, 800, 57, operation="decode"))