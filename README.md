# MyCustomNodes

ComfyUI에서 개인 워크플로를 보조하기 위한 유틸리티 커스텀 노드 모음입니다. 대부분의 노드는 `Mine` 카테고리에 등록됩니다.

## 설치

이 폴더를 ComfyUI의 `custom_nodes` 아래에 둔 뒤 ComfyUI를 재시작합니다.

```text
ComfyUI/custom_nodes/MyCustomNodes
```

프론트엔드 위젯을 사용하는 노드는 브라우저 캐시가 남아 있으면 변경 사항이 바로 보이지 않을 수 있습니다. 노드가 표시되지 않거나 `Show*` 계열 출력이 갱신되지 않으면 ComfyUI 재시작 후 브라우저에서 강력 새로고침을 해 주세요.

## 노드 목록

### WanI2VResizeImage

Wan I2V 계열 워크플로에서 입력 이미지의 픽셀 수와 해상도 배수를 맞추기 위한 이미지 리사이즈 노드입니다. 기존 워크플로 호환을 위해 `ResizeImageForWan` 이름도 legacy alias로 남아 있습니다.

- 입력: `image`, `max_pixels`, `vertical_crop`
- 출력: 리사이즈된 `IMAGE`, 최종 `width`, 최종 `height`
- 기능: 원본 이미지의 `width * height`가 `max_pixels`보다 크면 비율을 유지하면서 축소하고, 최종 width/height를 8의 배수로 맞춥니다. 원본이 `max_pixels` 이하이면 업스케일하지 않고, 필요한 만큼만 crop해서 8의 배수로 맞춥니다.
- `vertical_crop`: 원본이 `max_pixels` 이하라서 resize 없이 8의 배수로 crop할 때만 작동합니다. `crop bottom`은 아래쪽 픽셀을 자르고, `crop top`은 위쪽 픽셀을 자릅니다.
- 용도: Wan I2V 생성 전에 이미지가 너무 커서 VRAM을 많이 쓰는 상황을 줄이고, 모델에 넣기 좋은 8 배수 해상도로 정리할 때 사용합니다.

### CalculateResolution

원본 비율을 유지하면서 16의 배수이고 `max_pixels` 이하인 해상도를 계산합니다.

- 입력: `width`, `height`, `max_pixels`
- 출력: 계산된 `width`, `height`
- 기능: 이미 조건을 만족하면 원본 값을 그대로 반환하고, 아니면 원본보다 작거나 같은 후보 중 조건을 만족하는 최대 해상도를 찾습니다.
- 용도: 비디오/latent 기반 모델처럼 16 배수 해상도를 요구하는 워크플로에서 안전한 크기를 계산할 때 사용합니다.

### MathNode

여러 숫자 입력을 하나의 연산자로 누적 계산하는 수학 유틸 노드입니다.

- 입력: `operation`, `num_inputs`, 동적 hidden 입력 `value_1`부터 `value_100`
- 출력: `FLOAT`, `INT`, `STRING`
- 기능: `+`, `-`, `*`, `/`, `%` 연산을 왼쪽에서 오른쪽 순서로 적용합니다. 숫자로 변환할 수 없는 값은 `0.0`으로 처리합니다.
- 용도: 슬라이더, 프레임 수, 해상도 값 등을 간단히 조합해 다른 노드에 전달할 때 사용합니다.

### ShowInt

입력된 정수 값을 노드 UI에 표시하는 출력 노드입니다.

- 입력: `INT`
- 출력: 없음
- 기능: 리스트 입력을 받아 각 값을 문자열로 변환한 뒤 프론트엔드 위젯에 표시합니다.
- 용도: 계산된 프레임 수, 폭/높이, 인덱스 값을 워크플로 안에서 확인할 때 사용합니다.

### ShowFloat

입력된 실수 값을 노드 UI에 표시하는 출력 노드입니다.

- 입력: `FLOAT`
- 출력: 없음
- 기능: 리스트 입력을 받아 각 값을 문자열로 변환한 뒤 프론트엔드 위젯에 표시합니다.
- 용도: CFG, 스케일, 비율, 계산 결과 등을 중간 확인할 때 사용합니다.

### ShowStringText

입력된 문자열을 노드 UI에 표시하는 출력 노드입니다.

- 입력: `STRING`, `text_color`
- 출력: 없음
- 기능: 문자열 리스트를 표시용 텍스트 위젯으로 보여줍니다. `text_color`에서 표시 글자색을 선택할 수 있고, 이미 표시된 텍스트도 색상 변경 시 바로 갱신됩니다.
- 용도: 프롬프트, 치환 결과, 동적으로 생성된 텍스트를 워크플로 안에서 확인할 때 사용합니다.

### CustomCLIPTextEncodeNode

텍스트 인코더 파일을 직접 선택해 positive/negative 프롬프트를 conditioning으로 변환하는 CLIP 인코드 노드입니다.

- 입력: `clip_name`, `type`, `positive_prompt`, `negative_prompt`, `use_disk_cache`, 선택 `device`
- 출력: `Positive Conditioning`, `Negative Conditioning`
- 기능: 선택한 text encoder를 로드해 프롬프트를 인코딩하고, 사용 후 CLIP 모델을 해제합니다. `use_disk_cache`가 켜져 있으면 `text_embed_cache` 폴더에 인코딩 결과를 저장해 재사용합니다.
- 용도: Wan, SD3, Qwen Image 등 다양한 CLIP 타입에서 텍스트 conditioning을 만들고, 반복 실행 시 텍스트 인코딩 비용을 줄일 때 사용합니다.

### CustomTextEncodeQwenImageEditNode

Qwen Image Edit 계열처럼 텍스트와 참조 이미지를 함께 사용하는 워크플로용 conditioning 생성 노드입니다.

- 입력: `clip`, `positive_prompt`, `negative_prompt`, 선택 `latents`, 선택 `image1`부터 `image4`
- 출력: `Positive`, `Negative`
- 기능: 최대 4장의 이미지를 vision token 입력으로 변환하고, 이미지 수정 지시용 llama template과 함께 positive/negative conditioning을 생성합니다. latent가 연결되면 conditioning에 `reference_latents`를 추가합니다.
- 용도: 참조 이미지의 특징을 유지하면서 텍스트 지시에 따라 이미지를 편집하는 Qwen Image Edit 워크플로에 사용합니다.
- 주의: 현재 구현은 `latents`가 없을 때 안전 처리가 부족할 수 있으므로, 문제가 생기면 `CustomImagesEncodeNode`의 latent 출력을 함께 연결하는 구성이 좋습니다.

### CustomImagesEncodeNode

최대 4장의 참조 이미지를 VAE latent로 인코딩하는 노드입니다.

- 입력: `vae`, 선택 `image1`부터 `image4`
- 출력: `latents`, `first_latent`
- 기능: 각 이미지를 약 1024x1024 기준으로 리사이즈하고 8의 배수 크기로 맞춘 뒤 VAE로 latent를 만듭니다. 전체 latent 리스트와 첫 번째 latent를 함께 반환합니다.
- 용도: Qwen Image Edit 등에서 여러 참조 이미지 latent를 conditioning에 전달할 때 사용합니다.
- 주의: 적어도 하나의 이미지를 연결해야 합니다.

### WanAnimateToVideoNative

Wan Animate 계열 장문 비디오 생성을 내부 chunk 방식으로 처리하는 비디오 생성 노드입니다.

- 입력: `positive`, `negative`, `vae`, `width`, `height`, `total_length`, `chunk_length`, `continue_motion_max_frames`, `noise`, `guider`, `sampler`, `sigmas`, `batch_size`, `clip_vision_output`, `reference_image`, `face_video`, `pose_video`, `background_video`, `character_mask`, `max_total_pixels`
- 출력: 생성된 비디오 프레임 `IMAGE`
- 기능: 전체 길이를 여러 chunk로 나누어 샘플링하고, 이전 chunk의 마지막 프레임 일부를 다음 chunk의 motion reference로 사용합니다. reference image, background video, pose video, face video, character mask를 conditioning에 반영합니다. VAE encode/decode는 크기에 따라 tiled 방식으로 처리합니다.
- 용도: 긴 Wan Animate 비디오를 한 번에 만들기 어렵거나, 배경/포즈/얼굴/캐릭터 마스크를 함께 제어하고 싶을 때 사용합니다.
- 주의: `reference_image`는 사실상 필수입니다. `width`와 `height`는 16의 배수로 설정하는 것이 안전합니다.

### BatchImageSave

배치 이미지 텐서를 지정 폴더에 이미지 파일들로 저장하는 출력 노드입니다.

- 입력: `images`, `folder_path`, `filename_prefix`, `format`, `quality`
- 출력: 없음
- 기능: 배치의 각 이미지를 `filename_prefix_00000.ext` 형식으로 저장합니다. `format`은 `png`, `jpg`, `webp`를 지원합니다.
- 용도: 중간 프레임, pose/face 이미지, 전처리 결과를 외부 폴더로 저장할 때 사용합니다.

### LoadBatchImage

폴더 안의 이미지 파일들을 읽어 배치 `IMAGE`로 반환합니다.

- 입력: `folder_path`, `start_index`, `load_cap`, `sort_by`, `reverse_order`
- 출력: `images`, `count`
- 기능: `png`, `jpg`, `jpeg`, `webp`, `bmp`, `gif`, `tiff`, `tif` 파일을 로드합니다. 이름, 수정일, 생성일 기준 정렬과 역순 정렬을 지원합니다. 크기가 다른 이미지는 제외됩니다.
- 용도: 프레임 이미지 시퀀스를 ComfyUI 배치 이미지로 불러올 때 사용합니다.

### LoadMaskNode

폴더 안의 번호 기반 `.npy` 마스크 파일을 불러와 하나의 마스크로 병합합니다.

- 입력: `folder_path`, `obj_id_list`, `start_frame`, `load_cap`
- 출력: `MASK`
- 기능: `0.npy`, `1.npy`처럼 숫자로 시작하는 `.npy` 파일을 읽습니다. `obj_id_list`가 `-1`이면 전체를 사용하고, `1,3,5`처럼 지정하면 해당 ID만 사용합니다. 여러 마스크는 max 연산으로 합칩니다.
- 용도: 객체별 segmentation mask를 하나의 캐릭터/영역 마스크로 합쳐 사용할 때 적합합니다.

### SaveMaskNode

입력 마스크를 `.npy` 파일로 저장하는 출력 노드입니다.

- 입력: `mask`, `save_path`, `original_frames_count`, `start_index`, `end_index`
- 출력: 없음
- 기능: 입력 마스크를 전체 프레임 길이 안의 지정 구간에 삽입하고 나머지는 빈 마스크로 채워 저장합니다. `save_path`가 `.npy`가 아니면 `mask.npy`로 저장합니다.
- 용도: 일부 구간에서 만든 마스크를 원본 영상 길이에 맞춰 다시 저장할 때 사용합니다.

### MaskPaddingNode

마스크 앞뒤에 빈 프레임을 추가하고 `.npy`로 저장합니다.

- 입력: `mask`, `save_path`, `front_dummy_count`, `extract_count`, `back_dummy_count`
- 출력: 없음
- 기능: 입력 마스크 앞에 빈 마스크 `front_dummy_count`개, 뒤에 `back_dummy_count`개를 붙입니다. `extract_count`가 `-1`이면 전체 입력 마스크를 사용합니다.
- 용도: 마스크 시퀀스와 영상 프레임 타이밍을 맞추거나, 특정 구간만 추출해 앞뒤 패딩을 추가할 때 사용합니다.

### MaskResizingNode

마스크를 지정한 크기로 리사이즈합니다.

- 입력: `mask`, `width`, `height`
- 출력: `MASK`
- 기능: 2D 또는 프레임 배치 마스크를 bilinear interpolation으로 지정 크기에 맞춥니다.
- 용도: 이미지/비디오 크기 변경 후 기존 마스크 해상도를 맞출 때 사용합니다.

### MyGrowMask

마스크 영역을 확장하거나 축소합니다.

- 입력: `mask`, `expand`, `tapered_corners`, 선택 `face_mask`
- 출력: `mask`
- 기능: `expand`가 양수면 dilation으로 마스크를 키우고, 음수면 erosion으로 줄입니다. `tapered_corners`가 켜져 있으면 코너가 더 부드럽게 처리됩니다. `face_mask`가 있으면 확장된 새 영역 중 얼굴 마스크와 겹치는 부분을 제거합니다.
- 용도: segmentation mask를 약간 넓히거나 줄여 합성/인페인트 경계를 조정할 때 사용합니다.

### MyBlockifyMask

마스크 영역을 블록 단위로 단순화합니다.

- 입력: `masks`, `block_size`, 선택 `device`, 선택 `face_mask`
- 출력: `mask`
- 기능: 각 마스크의 bounding box를 블록 격자로 나눈 뒤 원본 마스크가 포함된 블록을 채웁니다. `device`는 `cpu` 또는 `gpu`를 선택할 수 있습니다. `face_mask`가 있으면 새로 생긴 블록 영역 중 얼굴과 겹치는 부분을 제거합니다.
- 용도: 세밀한 마스크를 더 큰 블록형 영역으로 만들어 배경/캐릭터 분리나 거친 영역 제어에 사용할 때 적합합니다.

### MyMaskSubtractNode

두 마스크를 빼서 겹치는 영역을 제거합니다.

- 입력: `masks_a`, `masks_b`
- 출력: `MASKS`
- 기능: `masks_a - masks_b`를 계산하고 0 이상으로 clamp합니다.
- 용도: 전체 캐릭터 마스크에서 얼굴 마스크를 빼거나, 특정 제외 영역을 제거할 때 사용합니다.

### LoadMaskImageFromPath

마스크, pose 이미지, face 이미지, face mask를 한 폴더 구조에서 함께 불러오는 복합 로더입니다.

- 입력: `folder_path`, `obj_id_list`, `start_frame`, `load_cap`
- 출력: `masks`, `pose_images`, `face_images`, `face_masks`
- 기능: 루트 폴더의 숫자 `.npy` 마스크를 불러오고, `pose` 하위 폴더의 이미지를 pose frame으로, `face` 하위 폴더의 이미지를 face frame으로 불러옵니다. 루트 폴더의 `face.npy`가 있으면 face mask로 불러옵니다. 큰 `.npy` 파일은 `mmap_mode='r'`로 필요한 구간만 읽습니다.
- 용도: Wan Animate처럼 character mask, pose video, face video, face mask를 함께 요구하는 워크플로에서 한 번에 데이터를 로드할 때 사용합니다.

권장 폴더 구조:

```text
dataset_folder/
  0.npy
  1.npy
  face.npy
  pose/
    00000.png
    00001.png
  face/
    00000.png
    00001.png
```

## 프론트엔드 기능

`web/js` 폴더에는 일부 노드의 UI 보조 스크립트가 포함되어 있습니다.

- `show_stuff.js`: `ShowInt`, `ShowFloat`, `ShowStringText` 실행 결과를 노드 안의 읽기 전용 텍스트 위젯으로 표시합니다.
- `folder_path_widget.js`: 경로 입력 위젯에서 폴더/파일 탐색을 돕는 커스텀 path widget 기능을 제공합니다.
- 기타 JS 파일: 이미지 배치, 텍스트 로드, 마스크 로드 등 일부 유틸 노드의 프론트 동작을 보조합니다.

## 캐시와 생성 파일

- `CustomCLIPTextEncodeNode`는 `text_embed_cache` 폴더에 텍스트 conditioning 캐시를 저장할 수 있습니다.
- `BatchImageSave`, `SaveMaskNode`, `MaskPaddingNode`는 지정한 경로에 파일을 생성합니다.
- 경로 입력 노드는 로컬 파일시스템 경로를 직접 사용하므로, ComfyUI가 접근 가능한 경로를 지정해야 합니다.
