import json, os
from torch.utils.data import Dataset
from .utils import load_image

class SFTDataset(Dataset):
    def __init__(self, jsonl_path, image_root, processor, max_samples=None):
        self.processor = processor
        self.image_root = image_root
        self.samples = [json.loads(l) for l in open(jsonl_path, "r", encoding="utf-8")]
        if max_samples: self.samples = self.samples[:max_samples]

    def __len__(self): return len(self.samples)

    def __getitem__(self, i):
        ex = self.samples[i]
        img = load_image(os.path.join(self.image_root, ex["image"]))
        
        # Kanana 모델에 맞는 입력 형식
        batch = [{
            "image": [img],
            "conv": [
                {"role": "system", "content": "The following is a conversation between a curious human and AI assistant."},
                {"role": "user", "content": "<image>"},
                {"role": "user", "content": ex["instruction"]},
                {"role": "assistant", "content": ex["output"]}
            ]
        }]
        
        model_inputs = self.processor.batch_encode_collate(
            batch, padding_side="left", add_generation_prompt=False, max_length=8192
        )
        model_inputs["labels"] = model_inputs["input_ids"].clone()
        return {k:v.squeeze(0) for k,v in model_inputs.items()}
