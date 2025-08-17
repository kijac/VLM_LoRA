#!/usr/bin/env python3
"""
HyperCLOVAX-SEED-Vision-Instruct-3B 모델 다운로드 스크립트
"""

import os
import argparse
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer

def download_model(model_name, output_dir, revision="main"):
    """
    모델을 로컬에 다운로드
    
    Args:
        model_name: Hugging Face 모델명
        output_dir: 저장할 디렉토리
        revision: 모델 버전 (기본값: main)
    """
    print(f"모델 다운로드 시작: {model_name}")
    print(f"저장 위치: {output_dir}")
    print(f"버전: {revision}")
    
    # 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # 모델 다운로드
        print("모델 파일 다운로드 중...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            trust_remote_code=True, 
            revision=revision,
            cache_dir=output_dir
        )
        
        # 프로세서 다운로드
        print("프로세서 다운로드 중...")
        processor = AutoProcessor.from_pretrained(
            model_name, 
            trust_remote_code=True, 
            revision=revision,
            cache_dir=output_dir
        )
        
        # 토크나이저 다운로드
        print("토크나이저 다운로드 중...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            revision=revision,
            cache_dir=output_dir
        )
        
        print("✅ 모델 다운로드 완료!")
        print(f"모델 크기: {model.get_memory_footprint() / 1024**3:.2f} GB")
        
    except Exception as e:
        print(f"❌ 다운로드 실패: {e}")
        raise

def main():
    parser = argparse.ArgumentParser(description="HyperCLOVAX 모델 다운로드")
    parser.add_argument(
        "--model_name", 
        default="naver-hyperclovax/HyperCLOVAX-SEED-Vision-Instruct-3B",
        help="다운로드할 모델명"
    )
    parser.add_argument(
        "--output_dir", 
        default="models/hyperclovax-vision-3b",
        help="모델 저장 디렉토리"
    )
    parser.add_argument(
        "--revision", 
        default="main",
        help="모델 버전 (main 또는 v0.1.0)"
    )
    
    args = parser.parse_args()
    
    download_model(args.model_name, args.output_dir, args.revision)

if __name__ == "__main__":
    main()
