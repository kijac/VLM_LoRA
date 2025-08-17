# VLM LoRA for Hello Carbot

헬로카봇 장난감 이미지 인식을 위한 Vision Language Model (VLM) LoRA 프로젝트입니다.

## 모델 정보

- **Base Model**: `naver-hyperclovax/HyperCLOVAX-SEED-Vision-Instruct-3B`
- **특징**: 한국어에 특화된 Vision Language Model
- **용도**: 헬로카봇 장난감 이미지 분석 및 매칭

## 설치

### 1. 의존성 설치

```bash
pip install -r requirements.txt
```

### 2. 모델 다운로드

HyperCLOVAX Vision 모델을 다운로드해야 합니다:

```bash
# 방법 1: 스크립트 사용
python scripts/download_model.py

# 방법 2: 직접 다운로드
python -c "from huggingface_hub import snapshot_download; snapshot_download('naver-hyperclovax/HyperCLOVAX-SEED-Vision-Instruct-3B', local_dir='models/hyperclovax-vision-3b')"
```

**주의**: 모델 파일이 매우 크므로 (약 6GB) 다운로드에 시간이 걸릴 수 있습니다.

## 프로젝트 구조

```
vlm_LoRA/
├── core/           # 핵심 모듈
│   ├── config.py   # 설정 (컬럼 매핑, 모델명 등)
│   ├── datasets.py # 데이터셋 클래스
│   ├── inference.py # 추론 로직
│   ├── retrieval.py # 검색/인덱싱
│   ├── utils.py    # 유틸리티 함수
│   └── vlm_lora.py # LoRA 훈련
├── data/           # 데이터
│   ├── images/     # 이미지 파일들
│   ├── raw/        # 원본 데이터 (Excel 등)
│   └── sft/        # SFT 데이터
├── scripts/        # 실행 스크립트
└── tests/          # 테스트
```

## TODO & 우선순위

### P0 (필수)
- [ ] COLUMN_MAP 실제 엑셀 컬럼명으로 매핑
- [ ] prepare_sft.py 실행 후 JSONL 품질 점검
- [ ] train_lora.py 1 epoch 스모크 테스트
- [ ] build_index.py 신품 이미지만 인덱싱
- [ ] infer.py 단일 이미지 추론 성공

### P1 (정확도 향상)
- [ ] 하드 네거티브 쌍 구성
- [ ] 재랭크 프롬프트 개선
- [ ] 속성 추출 프롬프트 고정

### P2 (운영/편의)
- [ ] 예외처리 강화
- [ ] 가격 출처/업데이트일자 필드 추가
