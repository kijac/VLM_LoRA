import os, json, math
from PIL import Image

def load_image(path: str):
    with Image.open(path) as im:
        return im.convert("RGB")

def to_list(cell):
    if cell is None: return []
    s = str(cell)
    for sep in ["|",";",".",","]:
        if sep in s:
            return [p.strip() for p in s.split(sep) if p.strip()]
    return [s.strip()]

def safe_float(x):
    try: return float(x)
    except: return None
