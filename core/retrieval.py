import os, json, numpy as np, faiss, open_clip
from tqdm import tqdm
from .utils import load_image

class OpenCLIPEncoder:
    def __init__(self, name="ViT-L-14", pretrained="openai", device=None):
        import torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, _, self.pp = open_clip.create_model_and_transforms(name, pretrained=pretrained, device=self.device)
        self.model.eval()

    @torch.no_grad()
    def embed(self, img):
        t = self.pp(img).unsqueeze(0).to(self.device)
        z = self.model.encode_image(t)
        z = z / z.norm(p=2, dim=-1, keepdim=True)
        return z.detach().cpu().numpy().astype("float32")

def build_index(rows, image_root, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    enc = OpenCLIPEncoder()
    feats, meta = [], []
    for r in tqdm(rows, desc="index"):
        for rel in r["new_images"]:
            path = os.path.join(image_root, rel)
            if not os.path.exists(path): continue
            vec = enc.embed(load_image(path))[0]
            feats.append(vec); meta.append({**r, "image": rel})
    feats = np.stack(feats, 0)
    faiss.normalize_L2(feats)
    idx = faiss.IndexFlatIP(feats.shape[1]); idx.add(feats)
    faiss.write_index(idx, os.path.join(out_dir, "index.faiss"))
    json.dump(meta, open(os.path.join(out_dir,"meta.json"),"w"), ensure_ascii=False, indent=2)
