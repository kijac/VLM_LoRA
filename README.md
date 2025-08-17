# VLM LoRA for Hello Carbot

헬로카봇 장난감 이미지 인식을 위한 Vision Language Model (VLM) LoRA 프로젝트입니다.

## 모델 정보

- **Base Model**: `kakaocorp/kanana-1.5-v-3b-instruct`
- **특징**: 한국어와 영어에 특화된 Vision Language Model (3.67B 파라미터)
- **용도**: 헬로카봇 장난감 이미지 분석 및 매칭
- **아키텍처**: Vision2Seq (Image-Text-to-Text) 모델

## 설치

### 1. 의존성 설치

```bash
pip install -r requirements.txt
```

### 2. 모델 다운로드

Kanana-1.5-v-3b-instruct 모델을 다운로드해야 합니다:

```bash
# 방법 1: 스크립트 사용
python scripts/download_model.py

# 방법 2: 직접 다운로드
python -c "from huggingface_hub import snapshot_download; snapshot_download('kakaocorp/kanana-1.5-v-3b-instruct', local_dir='models/kanana-1.5-v-3b-instruct')"
```

**주의**: 모델 파일이 매우 크므로 (약 3.67GB) 다운로드에 시간이 걸릴 수 있습니다.

## 스크립트 사용법

### 모델 다운로드

```bash
# 기본 설정으로 다운로드
python scripts/download_model.py

# 다른 저장 경로 지정
python scripts/download_model.py --output_dir "models/my_kanana"

# 특정 버전 다운로드
python scripts/download_model.py --revision "main"
```

### Git LFS를 사용한 다운로드 (더 빠름)

```bash
# 기본 설정으로 다운로드 (Git LFS 필요)
python scripts/download_model_git.py

# 다른 저장 경로 지정
python scripts/download_model_git.py --output_dir "models/my_kanana"

# 특정 버전 다운로드
python scripts/download_model_git.py --revision "main"
```

### 모델 테스트

```bash
# 모델 로드 및 기본 추론 테스트
python scripts/test_kanana_model.py
```

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
