import os
import sys
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gentle SFT Training Script (No TRL)")
    parser.add_argument("--output_dir", type=str, default="./qwen_gentle_sft", help="Output directory")
    # TWEAK 1: Defaulting to 1 epoch for gentler training
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=5e-6, help="Learning rate for SFT")
    parser.add_argument("--batch_size", type=int, default=2, help="Per-device batch size")
    parser.add_argument("--grad_accum", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--neftune_alpha", type=float, default=5.0, help="NEFTune noise scale")
    
    args = parser.parse_args()
    TARGET_DIR = args.output_dir
    NUM_EPOCHS = args.epochs
    LEARNING_RATE = args.learning_rate
    BATCH_SIZE = args.batch_size
    GRAD_ACCUM = args.grad_accum
    NEFTUNE_ALPHA = args.neftune_alpha

else:
    TARGET_DIR = "./fallback_dir"
    NUM_EPOCHS = 1
    LEARNING_RATE = 5e-5
    BATCH_SIZE = 2
    GRAD_ACCUM = 4
    NEFTUNE_ALPHA = 5.0

print("Loading ML libraries... (This might take a minute)")

import re
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from torch.utils.tensorboard import SummaryWriter
from math_verify import parse, verify

# ── Configuration ────────────────────────────────────────────────────────────
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
MAX_SEQ_LEN = 768

SYSTEM_PROMPT = (
    "You are a math reasoning assistant. Think step by step inside "
    "<think>...</think> tags, then give your final answer inside \\boxed{}."
)

# ── Data Processing ─────────────────────────────────────────────────────────
BOXED_OPEN_RE = re.compile(r"\\boxed\{")

def extract_boxed(text: str) -> str:
    starts = [m.start() for m in BOXED_OPEN_RE.finditer(text)]
    if not starts:
        return None
    idx = starts[-1] + len("\\boxed{")
    depth, i = 1, idx
    while i < len(text) and depth > 0:
        if text[i] == "{": depth += 1
        elif text[i] == "}": depth -= 1
        i += 1
    return text[idx : i - 1].strip() if depth == 0 else None

def format_eval_example(ex):
    parts = str(ex["answer"]).split("####")
    final_answer = parts[1].strip() if len(parts) > 1 else ""
    return {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": ex["question"]},
        ],
        "answer": final_answer,
    }

# ── TWEAK 3: Custom SFT Tokenization & Masking ──────────────────────────────
def tokenize_and_mask(ex, tokenizer):
    """
    TWEAK 3: We strictly mask the user prompt so the model only calculates
    loss on its generated answer, preventing it from 'forgetting' how to read prompts.
    """
    parts = str(ex["answer"]).split("####")
    reasoning = parts[0].strip()
    final_answer = parts[1].strip() if len(parts) > 1 else ""
    assistant_content = f"<think>\n{reasoning}\n</think>\n\\boxed{{{final_answer}}}"

    messages_prompt = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": ex["question"]}
    ]
    prompt_text = tokenizer.apply_chat_template(messages_prompt, tokenize=False, add_generation_prompt=True)
    
    messages_full = messages_prompt + [{"role": "assistant", "content": assistant_content}]
    full_text = tokenizer.apply_chat_template(messages_full, tokenize=False)

    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(full_text, add_special_tokens=False, truncation=True, max_length=MAX_SEQ_LEN)["input_ids"]

    labels = full_ids.copy()
    prompt_len = min(len(prompt_ids), len(labels))
    
    # Masking out the prompt tokens
    for i in range(prompt_len):
        labels[i] = -100

    return {"input_ids": full_ids, "labels": labels}

def custom_collate_fn(features, tokenizer):
    batch_input_ids = [torch.tensor(f["input_ids"], dtype=torch.long) for f in features]
    batch_labels = [torch.tensor(f["labels"], dtype=torch.long) for f in features]
    
    input_ids = torch.nn.utils.rnn.pad_sequence(batch_input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    labels = torch.nn.utils.rnn.pad_sequence(batch_labels, batch_first=True, padding_value=-100)
    attention_mask = (input_ids != tokenizer.pad_token_id).long()
    
    return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}

# ── TWEAK 4: Custom NEFTune Hook ─────────────────────────────────────────────
def add_neftune_hook(model, noise_alpha=5.0):
    """
    Dynamically adds uniform noise to the embedding layer during training.
    Scale formula: alpha / sqrt(sequence_length * hidden_dimension)
    """
    def neftune_forward_hook(module, inputs, output):
        if not model.training:
            return output
        embeds = output
        seq_length, hidden_size = embeds.size(1), embeds.size(2)
        
        dims = torch.tensor(seq_length * hidden_size, dtype=embeds.dtype, device=embeds.device)
        mag_norm = noise_alpha / torch.sqrt(dims)
        
        noise = torch.zeros_like(embeds).uniform_(-mag_norm, mag_norm)
        return embeds + noise

    embeds_layer = model.get_input_embeddings()
    return embeds_layer.register_forward_hook(neftune_forward_hook)

# ── Evaluation ───────────────────────────────────────────────────────────────
def evaluate_model(model, tokenizer, dataset, tb_writer, phase_name, step=0):
    print(f"\n--- Starting {phase_name} Evaluation ---")
    model.eval()
    correct = 0
    total = len(dataset)
    
    for i, item in enumerate(tqdm(dataset, desc=f"Evaluating ({phase_name})")):
        prompt_text = tokenizer.apply_chat_template(item["prompt"], tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=512,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=False
            )
        
        generated_text = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        pred = extract_boxed(generated_text)
        gt = item["answer"]
        
        try:
            if pred and verify(parse(pred), parse(gt)):
                correct += 1
        except Exception:
            pass

    accuracy = correct / total
    print(f"{phase_name} Accuracy: {correct}/{total} ({accuracy:.2%})")
    
    if tb_writer:
        tb_writer.add_scalar(f"Accuracy/{phase_name}", accuracy, step)
    
    model.train()
    return accuracy

# ── Main Script ──────────────────────────────────────────────────────────────
def main(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    tb_writer = SummaryWriter(log_dir=os.path.join(output_dir, "logs"))

    print(f"Loading tokenizer and base model from {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    base_model.config.use_cache = False

    print("Preparing GSM8K dataset...")
    raw_train = load_dataset("openai/gsm8k", "main", split="train")
    raw_test = load_dataset("openai/gsm8k", "main", split="test")
    
    eval_dataset = raw_test.select(range(500)).map(format_eval_example, remove_columns=raw_test.column_names)
    
    # TWEAK 1: Subsample the training data down to 1000 examples
    print("TWEAK 1: Subsampling the training dataset to 1,000 examples...")
    raw_train = raw_train.shuffle(seed=42).select(range(1000))
    
    train_dataset = raw_train.map(
        lambda ex: tokenize_and_mask(ex, tokenizer), 
        remove_columns=raw_train.column_names
    )
    
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        collate_fn=lambda features: custom_collate_fn(features, tokenizer)
    )

    # 1. Baseline Evaluation
    evaluate_model(base_model, tokenizer, eval_dataset, tb_writer, phase_name="Baseline", step=0)

    # 2. Setup LoRA
    # TWEAK 2: Reduced LoRA Rank (r=8) and Alpha (16)
    print("\nInitializing fresh LoRA adapters (TWEAK 2: rank=8, alpha=16)...")
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    
    model = get_peft_model(base_model, lora_config)
    model.gradient_checkpointing_enable()

    # TWEAK 4: Enable NEFTune via Forward Hook
    if NEFTUNE_ALPHA > 0:
        print(f"\nTWEAK 4: Activating NEFTune embedding noise (alpha={NEFTUNE_ALPHA})...")
        neftune_hook_handle = add_neftune_hook(model, noise_alpha=NEFTUNE_ALPHA)

    # 3. Custom PyTorch Training Loop
    print(f"\nStarting Custom Training Loop! Tensorboard logs at: {os.path.join(output_dir, 'logs')}")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    global_step = 0
    
    model.train()
    for epoch in range(NUM_EPOCHS):
        print(f"\n--- Epoch {epoch + 1}/{NUM_EPOCHS} ---")
        
        progress_bar = tqdm(train_dataloader, desc="Training")
        optimizer.zero_grad()
        
        for step, batch in enumerate(progress_bar):
            input_ids = batch["input_ids"].to(model.device)
            labels = batch["labels"].to(model.device)
            attention_mask = batch["attention_mask"].to(model.device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss / GRAD_ACCUM
            
            loss.backward()
            
            if (step + 1) % GRAD_ACCUM == 0 or (step + 1) == len(train_dataloader):
                optimizer.step()
                optimizer.zero_grad()
                
                global_step += 1
                tb_writer.add_scalar("Loss/train", loss.item() * GRAD_ACCUM, global_step)
                progress_bar.set_postfix({"loss": f"{loss.item() * GRAD_ACCUM:.4f}"})

    # Cleanup TWEAK 4 Hook
    if NEFTUNE_ALPHA > 0:
        neftune_hook_handle.remove()

    # 4. Post-Training Evaluation
    evaluate_model(model, tokenizer, eval_dataset, tb_writer, phase_name="Post-Training", step=NUM_EPOCHS)

    # 5. Merge LoRA and Save
    print("\nMerging LoRA weights with base model...")
    merged_model = model.merge_and_unload()
    
    final_save_path = os.path.join(output_dir, "final_merged_model")
    print(f"Saving merged model to {final_save_path}...")
    merged_model.save_pretrained(final_save_path)
    tokenizer.save_pretrained(final_save_path)
    
    tb_writer.close()
    print("Pipeline complete!")


if __name__ == "__main__":
    main(TARGET_DIR)
