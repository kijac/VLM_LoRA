from dataclasses import dataclass
import os

@dataclass
class ColumnMap:
    product_id: str = "product_id"
    brand: str = "brand"
    series: str = "series"
    model: str = "model"
    color: str = "color"
    attributes: str = "attributes"
    new_price: str = "new_price"
    used_price: str = "used_price"
    new_images: str = "new_images"
    used_images: str = "used_images"

COLUMN_MAP = ColumnMap()

# 기본 모델 설정 - Kanana-1.5-v-3b-instruct로 변경
DEFAULT_BASE_MODEL = "kakaocorp/kanana-1.5-v-3b-instruct"

# 로컬 모델 경로 설정
LOCAL_MODEL_PATH = "models/kanana-1.5-v-3b-instruct"

def get_model_path():
    """로컬 모델이 있으면 로컬 경로, 없으면 원격 모델명 반환"""
    if os.path.exists(LOCAL_MODEL_PATH):
        return LOCAL_MODEL_PATH
    return DEFAULT_BASE_MODEL
