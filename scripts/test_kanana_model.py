#!/usr/bin/env python3
"""
Kanana-1.5-v-3b-instruct 모델 테스트 스크립트

사용법:
    # 기본 테스트 실행
    python scripts/test_kanana_model.py

    # 모델이 정상적으로 로드되고 추론이 가능한지 확인
    # 테스트 이미지로 간단한 추론을 수행합니다
"""

import os
import torch
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor
from core.config import get_model_path

def test_kanana_model():
    """Kanana 모델 로드 및 기본 추론 테스트"""
    
    model_path = get_model_path()
    print(f"모델 경로: {model_path}")
    
    try:
        # 모델 로드
        print("모델 로딩 중...")
        model = AutoModelForVision2Seq.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        model.eval()
        
        # 프로세서 로드
        print("프로세서 로딩 중...")
        processor = AutoProcessor.from_pretrained(
            model_path, 
            trust_remote_code=True
        )
        
        print("✅ 모델 로드 성공!")
        
        # 간단한 테스트 이미지 생성 (1x1 픽셀)
        test_image = Image.new('RGB', (224, 224), color='red')
        
        # 테스트 배치 생성
        batch = [{
            "image": [test_image],
            "conv": [
                {"role": "system", "content": "The following is a conversation between a curious human and AI assistant."},
                {"role": "user", "content": "<image>"},
                {"role": "user", "content": "이 이미지에 대해 간단히 설명해주세요."}
            ]
        }]
        
        # 입력 처리
        inputs = processor.batch_encode_collate(
            batch, padding_side="left", add_generation_prompt=True, max_length=8192
        )
        inputs = {k: v.to(model.device) if hasattr(v, 'to') else v for k, v in inputs.items()}
        
        # 생성 설정
        gen_kwargs = {
            "max_new_tokens": 50,
            "temperature": 0,
            "top_p": 1.0,
            "num_beams": 1,
            "do_sample": False,
        }
        
        # 추론 실행
        print("추론 실행 중...")
        with torch.no_grad():
            outputs = model.generate(**inputs, **gen_kwargs)
        
        # 결과 디코딩
        text_outputs = processor.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        print(f"✅ 추론 성공!")
        print(f"결과: {text_outputs[0]}")
        
        return True
        
    except Exception as e:
        print(f"❌ 테스트 실패: {e}")
        return False

if __name__ == "__main__":
    success = test_kanana_model()
    if success:
        print("\n🎉 Kanana 모델 테스트 완료! 모델이 정상적으로 작동합니다.")
    else:
        print("\n💥 Kanana 모델 테스트 실패! 문제를 확인해주세요.")
