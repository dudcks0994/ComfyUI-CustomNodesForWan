import math

class CalculateResolution:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 1}),
                "height": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 1}),
                "max_pixels": ("INT", {"default": 840000, "min": 0, "max": 10000000, "step": 1}),
            },
        }

    FUNCTION = "calculate_resolution"
    RETURN_TYPES = ("INT", "INT",)
    RETURN_NAMES = ("width", "height")
    OUTPUT_NODE = True
    CATEGORY = "Mine"

    def calculate_resolution(self, width, height, max_pixels):
        """
        1. 현재 가로세로가 이미 조건(16배수 & max_pixels 이하)을 충족하면 그대로 반환
        2. 충족하지 않으면: 비율 유지, 16배수, 원본보다 작으면서 max_pixels 이하 중 가장 큰 값
        """
        
        current_area = width * height
        
        # 조건1: 이미 16의 배수인지 확인
        is_width_16 = (width % 16 == 0)
        is_height_16 = (height % 16 == 0)
        # 조건2: 면적이 max_pixels 이하인지 확인
        is_under_max = (current_area <= max_pixels)
        
        # 모든 조건 충족시 그대로 반환
        if is_width_16 and is_height_16 and is_under_max:
            return width, height

        # === 조건 미충족: 원본보다 작으면서 가장 큰 값 찾기 ===
        
        # 1. 원본 비율의 최소 단위(기약분수) 구하기
        # 예: 1920x1080 -> 16:9
        common_divisor = math.gcd(width, height)
        w_base = width // common_divisor
        h_base = height // common_divisor
        
        # 2. 16으로 나누어 떨어지기 위한 '최소 보정 배수' 찾기
        factor_w = 16 // math.gcd(w_base, 16)
        factor_h = 16 // math.gcd(h_base, 16)
        
        # 두 조건을 모두 만족하는 최소공배수(LCM) 계산
        min_multiplier = (factor_w * factor_h) // math.gcd(factor_w, factor_h)
        
        # 3. 조건을 만족하는 '최소 단위 블록' 크기 확정
        unit_width = w_base * min_multiplier
        unit_height = h_base * min_multiplier
        unit_area = unit_width * unit_height
        
        # [예외처리] 최소 단위 블록 자체가 이미 제한보다 큰 경우
        if unit_area > max_pixels:
            print(f"불가능: 이 비율을 정확히 유지하며 16배수를 맞추려면 최소 {unit_width}x{unit_height} ({unit_area}px)이 필요한데, 제한({max_pixels}px)보다 큽니다.")
            return 0, 0

        # 4. 원본보다 작으면서 max_pixels 이하인 가장 큰 값 찾기
        # 조건: (N * unit_width) <= width AND (N * unit_height) <= height AND N^2 * unit_area <= max_pixels
        
        # 각 조건별 최대 N 계산
        max_n_by_width = width // unit_width
        max_n_by_height = height // unit_height
        max_n_by_pixels = int(math.sqrt(max_pixels / unit_area))
        
        # 세 조건 중 가장 작은 값이 최종 N
        max_n = min(max_n_by_width, max_n_by_height, max_n_by_pixels)
        
        if max_n < 1:
            print(f"불가능: 원본({width}x{height})보다 작으면서 조건을 만족하는 해상도를 찾을 수 없습니다.")
            return 0, 0
        
        final_width = unit_width * max_n
        final_height = unit_height * max_n
        
        return final_width, final_height