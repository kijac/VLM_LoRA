#!/usr/bin/env python3
"""
Git LFS를 사용한 Kanana 모델 다운로드 스크립트

사용법:
    # 기본 설정으로 다운로드 (Git LFS 필요)
    python scripts/download_model_git.py

    # 다른 저장 경로 지정
    python scripts/download_model_git.py --output_dir "models/my_kanana"

    # 특정 버전 다운로드
    python scripts/download_model_git.py --revision "main"
"""

import os
import subprocess
import argparse

def download_model_git(model_name, output_dir, revision="main"):
    """
    Git LFS를 사용하여 모델 다운로드
    
    Args:
        model_name: Hugging Face 모델명 (예: kakaocorp/kanana-1.5-v-3b-instruct)
        output_dir: 저장할 디렉토리
        revision: 모델 버전
    """
    print(f"Git LFS를 사용한 모델 다운로드 시작: {model_name}")
    print(f"저장 위치: {output_dir}")
    
    # 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    # Hugging Face URL 생성
    hf_url = f"https://huggingface.co/{model_name}"
    
    try:
        # Git LFS 클론
        print("Git LFS 클론 중...")
        cmd = [
            "git", "lfs", "clone", 
            "--depth", "1",
            "--branch", revision,
            hf_url, 
            output_dir
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Git LFS 다운로드 완료!")
        else:
            print(f"❌ Git LFS 다운로드 실패: {result.stderr}")
            raise Exception("Git LFS 다운로드 실패")
            
    except FileNotFoundError:
        print("❌ Git LFS가 설치되지 않았습니다.")
        print("Git LFS 설치 방법:")
        print("  Windows: https://git-lfs.com/")
        print("  macOS: brew install git-lfs")
        print("  Ubuntu: sudo apt install git-lfs")
        raise
    except Exception as e:
        print(f"❌ 다운로드 실패: {e}")
        raise

def main():
    parser = argparse.ArgumentParser(description="Git LFS를 사용한 Kanana 모델 다운로드")
    parser.add_argument(
        "--model_name", 
        default="kakaocorp/kanana-1.5-v-3b-instruct",
        help="다운로드할 모델명"
    )
    parser.add_argument(
        "--output_dir", 
        default="models/kanana-1.5-v-3b-instruct",
        help="모델 저장 디렉토리"
    )
    parser.add_argument(
        "--revision", 
        default="main",
        help="모델 버전 (main 또는 특정 태그)"
    )
    
    args = parser.parse_args()
    
    download_model_git(args.model_name, args.output_dir, args.revision)

if __name__ == "__main__":
    main()
