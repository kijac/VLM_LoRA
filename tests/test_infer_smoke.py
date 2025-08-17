import pandas as pd
from core.inference import run_infer
from core.config import get_model_path

def test_infer_smoke():
    df = pd.read_excel("data/raw/datasets.xlsx")
    out = run_infer("samples/used.jpg", df, "data/images",
                    get_model_path(), "outputs/carbot_lora", "outputs/carbot_index", topk=2)
    assert isinstance(out, dict)
    assert "candidates" in out
