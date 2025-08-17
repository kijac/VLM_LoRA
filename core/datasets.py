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
        msgs = [
            {"role":"user","content":[{"type":"image","image":img},{"type":"text","text":ex["instruction"]}]},
            {"role":"assistant","content":[{"type":"text","text":ex["output"]}]}
        ]
        text = self.processor.apply_chat_template(msgs, add_generation_prompt=False)
        model_inputs = self.processor(text=[text], images=[img], return_tensors="pt", padding=True)
        model_inputs["labels"] = model_inputs["input_ids"].clone()
        return {k:v.squeeze(0) for k,v in model_inputs.items()}
