from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model
from .datasets import SFTDataset

def load_base_with_lora(base_model, use_qlora=True):
    quant = None; dtype=None; device_map="auto"
    if use_qlora:
        quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True,
                                   bnb_4bit_quant_type="nf4")
    proc = AutoProcessor.from_pretrained(base_model, trust_remote_code=True)
    base = AutoModelForVision2Seq.from_pretrained(base_model, device_map=device_map,
                                                quantization_config=quant, trust_remote_code=True)
    
    # Kanana 모델에 맞는 LoRA 설정 (Vision2Seq 모델용)
    lora = LoraConfig(
        r=16, 
        lora_alpha=32, 
        lora_dropout=0.05,
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
        bias="none", 
        task_type="SEQ_2_SEQ_LM"  # Vision2Seq 모델용으로 변경
    )
    return proc, get_peft_model(base, lora)

def train_lora(jsonl, image_root, base_model, out_dir, epochs=1, lr=2e-5, batch=1, max_samples=None, use_qlora=True):
    proc, model = load_base_with_lora(base_model, use_qlora)
    ds = SFTDataset(jsonl, image_root, proc, max_samples)
    args = TrainingArguments(output_dir=out_dir, num_train_epochs=epochs, per_device_train_batch_size=batch,
                             learning_rate=lr, gradient_accumulation_steps=8, logging_steps=10,
                             save_steps=200, save_total_limit=2, warmup_ratio=0.03, lr_scheduler_type="cosine",
                             optim="paged_adamw_8bit" if use_qlora else "adamw_torch", report_to=[])
    trainer = Trainer(model=model, args=args, train_dataset=ds,
                      tokenizer=proc.tokenizer if hasattr(proc,"tokenizer") else proc)
    trainer.train()
    model.save_pretrained(out_dir); proc.save_pretrained(out_dir)
