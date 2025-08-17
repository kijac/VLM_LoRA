import os, json, faiss
from peft import PeftModel
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig
from .utils import load_image, safe_float

def load_vlm(base_model, lora_dir):
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4")
    proc = AutoProcessor.from_pretrained(lora_dir if os.path.exists(os.path.join(lora_dir,"processor_config.json")) else base_model,
                                         trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(base_model, device_map="auto",
                                                 quantization_config=quant, trust_remote_code=True)
    model = PeftModel.from_pretrained(model, lora_dir); model.eval()
    return proc, model

def extract_attrs(proc, model, img):
    prompt = ("이 헬로카봇 장난감 이미지를 분석해서 JSON 형태로 정보를 알려주세요. "
              "다음 필드들을 포함해주세요: brand, series, model, color, robot_name, vehicle_type, accessories. "
              "알 수 없는 정보는 null로 표시해주세요.")
    
    # Kanana 모델에 맞는 입력 형식
    batch = [{
        "image": [img],
        "conv": [
            {"role": "system", "content": "The following is a conversation between a curious human and AI assistant."},
            {"role": "user", "content": "<image>"},
            {"role": "user", "content": prompt}
        ]
    }]
    
    inputs = proc.batch_encode_collate(batch, padding_side="left", add_generation_prompt=True, max_length=8192)
    inputs = {k: v.to(model.device) if hasattr(v, 'to') else v for k, v in inputs.items()}
    
    gen_kwargs = {
        "max_new_tokens": 256,
        "temperature": 0,
        "top_p": 1.0,
        "num_beams": 1,
        "do_sample": False,
    }
    
    out = model.generate(**inputs, **gen_kwargs)
    s = proc.tokenizer.batch_decode(out, skip_special_tokens=True)[0]
    try:
        st, ed = s.rfind("{"), s.rfind("}")
        return json.loads(s[st:ed+1]) if st!=-1 and ed!=-1 else {}
    except: return {}

def pair_rerank(proc, model, img_a, img_b):
    prompt = ("두 헬로카봇 장난감 이미지를 비교해서 같은 제품(SKU)인지 판단해주세요. "
              "0점(완전히 다른 제품)부터 1점(동일한 제품)까지 점수로 답해주세요.")
    
    # Kanana 모델에 맞는 입력 형식
    batch = [{
        "image": [img_a, img_b],
        "conv": [
            {"role": "system", "content": "The following is a conversation between a curious human and AI assistant."},
            {"role": "user", "content": "<image>"},
            {"role": "user", "content": "<image>"},
            {"role": "user", "content": prompt}
        ]
    }]
    
    inputs = proc.batch_encode_collate(batch, padding_side="left", add_generation_prompt=True, max_length=8192)
    inputs = {k: v.to(model.device) if hasattr(v, 'to') else v for k, v in inputs.items()}
    
    gen_kwargs = {
        "max_new_tokens": 16,
        "temperature": 0,
        "top_p": 1.0,
        "num_beams": 1,
        "do_sample": False,
    }
    
    out = model.generate(**inputs, **gen_kwargs)
    s = proc.tokenizer.batch_decode(out, skip_special_tokens=True)[0]
    import re
    nums = [float(x) for x in re.findall(r"[0-9]*\.?[0-9]+", s)]
    return max(0.0, min(1.0, nums[-1])) if nums else 0.0

def run_infer(used_path, excel_df, image_root, base_model, lora_dir, index_dir, topk=8):
    used = load_image(used_path)
    proc, model = load_vlm(base_model, lora_dir)
    attrs = extract_attrs(proc, model, used)

    index = faiss.read_index(os.path.join(index_dir, "index.faiss"))
    meta = json.load(open(os.path.join(index_dir,"meta.json"),"r",encoding="utf-8"))

    # OpenCLIP 임베딩: enc는 build_index에서만 사용(여긴 스킵) → 실제 구현시 공유 or 재생성
    from .retrieval import OpenCLIPEncoder
    enc = OpenCLIPEncoder()
    D, I = index.search(enc.embed(used), topk)

    cands = []
    for idx, sim in zip(I[0], D[0]):
        m = meta[idx]
        cand_img = load_image(os.path.join(image_root, m["image"]))
        score = pair_rerank(proc, model, used, cand_img)
        cands.append({
            "product_id": m["product_id"], "brand": m["brand"], "series": m["series"], "model": m["model"],
            "index_image": m["image"], "similarity": float(sim), "rerank_score": float(score)
        })
    cands.sort(key=lambda x:(x["rerank_score"], x["similarity"]), reverse=True)
    best = cands[0] if cands else None

    new_price = used_price = None
    if best is not None:
        row = excel_df.loc[excel_df["product_id"].astype(str) == str(best["product_id"])]
        if not row.empty:
            r = row.iloc[0]
            new_price, used_price = r.get("new_price"), r.get("used_price")

    return {"attrs": attrs, "candidates": cands, "best": (best and {
        **best, "new_price": safe_float(new_price), "used_price": safe_float(used_price)
    })}
