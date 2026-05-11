import os
import sys
import argparse
from peft import PeftModel

# 1. Ask for directory and parameters BEFORE doing heavy imports
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GRPO Training Script")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save checkpoints, logs, and config.")
    
    # --- Resume from checkpoint ---
    parser.add_argument("--resume_from", type=str, default=None, help="Path to a previous run's checkpoint (e.g., ./runs/exp1/final) to continue training.")
    
    # Reward Shaping Parameters
    parser.add_argument("--reward_think", type=float, default=0.1, help="Reward for including <think> tags")
    parser.add_argument("--reward_box", type=float, default=0.1, help="Reward for including \\boxed{} tags")
    parser.add_argument("--reward_correct", type=float, default=1.0, help="Reward for a mathematically correct answer")
    parser.add_argument("--reward_incorrect", type=float, default=0.0, help="Reward for an incorrect answer")
    
    args = parser.parse_args()
    
    TARGET_DIR = args.output_dir
    RESUME_FROM = args.resume_from
    
    if not TARGET_DIR:
        TARGET_DIR = input("Please specify an output directory for this run (e.g., './runs/exp2'): ").strip()
        if not TARGET_DIR:
            print("Error: No output directory provided. Exiting.")
            sys.exit(1)
            
    REWARD_THINK = args.reward_think
    REWARD_BOX = args.reward_box
    REWARD_CORRECT = args.reward_correct
    REWARD_INCORRECT = args.reward_incorrect

else:
    TARGET_DIR = "./fallback_dir"
    RESUME_FROM = None
    REWARD_THINK = 0.1
    REWARD_BOX = 0.1
    REWARD_CORRECT = 1.0
    REWARD_INCORRECT = 0.0

print("Loading ML libraries... (This might take a minute over the Docker mount)")

import re
import json
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from math_verify import parse, verify

# Import our custom GRPO trainer
from grpo import GRPOConfig, GRPOTrainer

# ── Configuration ────────────────────────────────────────────────────────────
SFT_MODEL_DIR = "./qwen2.5-1.5b-sft_v4_base-merged"             # <-- Point this to your local downloaded model

DATASET_ID    = "openai/gsm8k"
MAX_PROMPT_LEN     = 512
MAX_COMPLETION_LEN = 1024 

NUM_TRAIN     = 2000
NUM_TEST      = 500          # 500-prompt slice for the Eval Callback
NUM_EPOCHS    = 1
LOG_STEPS     = 1
SAVE_STEPS    = 50
EVAL_STEPS    = 50           # Evaluate against test set every 50 steps

# Hyperparameters
NUM_GENERATIONS  = 6
PER_DEVICE_BS    = 1        
GRAD_ACCUM       = 8         
LEARNING_RATE    = 4e-6      
BETA             = 0.04      
EPSILON          = 0.2
MAX_GRAD_NORM    = 1.0

SYSTEM_PROMPT = (
    "You are a math reasoning assistant. Think step by step inside "
    r"<think>...</think> tags, then give your final answer inside \boxed{}"
)

# ── Data Processing ─────────────────────────────────────────────────────────
GSM_GOLD_RE = re.compile(r"####\s*(-?[\d,\.]+)")

def extract_gt(answer_field: str) -> str:
    m = GSM_GOLD_RE.search(answer_field)
    return m.group(1).replace(",", "").strip() if m else answer_field.strip().split()[-1]

def to_prompt(ex):
    return {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": ex["question"]},
        ],
        "answer": extract_gt(ex["answer"]),
    }

# ── Rewards ──────────────────────────────────────────────────────────────────
THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
BOXED_OPEN_RE = re.compile(r"\\boxed\{")

def extract_boxed(text):
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

def _to_text(completion):
    if isinstance(completion, list) and completion and isinstance(completion[0], dict):
        return completion[0].get("content", "")
    return completion if isinstance(completion, str) else str(completion)

def format_reward(completions, **kwargs):
    rewards = []
    for c in completions:
        text = _to_text(c)
        score = 0.0
        if THINK_RE.search(text): score += REWARD_THINK
        if extract_boxed(text) is not None: score += REWARD_BOX
        rewards.append(score)
    return rewards

def correctness_reward(completions, answer, **kwargs):
    rewards = []
    for c, ans in zip(completions, answer):
        text = _to_text(c)
        pred = extract_boxed(text)
        if pred is None:
            rewards.append(REWARD_INCORRECT) # Penalize/score missing boxes as incorrect
            continue
        try:
            if verify(parse(pred), parse(ans)):
                rewards.append(REWARD_CORRECT)
            else:
                rewards.append(REWARD_INCORRECT)
        except Exception:
            rewards.append(REWARD_INCORRECT)
    return rewards

# ── Main Script ──────────────────────────────────────────────────────────────
def main(output_dir):
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Dump the run configuration for experiment tracking
    run_config = {
        "SFT_MODEL_DIR": SFT_MODEL_DIR,
        "OUTPUT_DIR": output_dir,
        "DATASET_ID": DATASET_ID,
        "NUM_TRAIN": NUM_TRAIN,
        "NUM_TEST": NUM_TEST,
        "NUM_EPOCHS": NUM_EPOCHS,
        "NUM_GENERATIONS": NUM_GENERATIONS,
        "PER_DEVICE_BS": PER_DEVICE_BS,
        "GRAD_ACCUM": GRAD_ACCUM,
        "LEARNING_RATE": LEARNING_RATE,
        "BETA": BETA,
        "EPSILON": EPSILON,
        "MAX_GRAD_NORM": MAX_GRAD_NORM,
        "MAX_PROMPT_LEN": MAX_PROMPT_LEN,
        "MAX_COMPLETION_LEN": MAX_COMPLETION_LEN,
        "SYSTEM_PROMPT": SYSTEM_PROMPT,
        "RESUME_FROM": RESUME_FROM,
        "REWARDS": {
            "reward_think": REWARD_THINK,
            "reward_box": REWARD_BOX,
            "reward_correct": REWARD_CORRECT,
            "reward_incorrect": REWARD_INCORRECT
        }
    }
    
    config_path = os.path.join(output_dir, "run_config.json")
    with open(config_path, "w") as f:
        json.dump(run_config, f, indent=4)
    print(f"Run configuration saved to {config_path}")
    
    print(f"Loading dataset {DATASET_ID}...")
    raw_train = load_dataset(DATASET_ID, "main", split="train").shuffle(seed=42)
    raw_test  = load_dataset(DATASET_ID, "main", split="test").shuffle(seed=42)
    
    raw_train = raw_train.select(range(min(NUM_TRAIN, len(raw_train))))
    raw_test  = raw_test.select(range(min(NUM_TEST, len(raw_test))))
    
    train_ds = raw_train.map(to_prompt, remove_columns=raw_train.column_names)
    test_ds  = raw_test.map(to_prompt, remove_columns=raw_test.column_names)
    
    print(f"Loading base model and tokenizer from {SFT_MODEL_DIR}...")
    tokenizer = AutoTokenizer.from_pretrained(SFT_MODEL_DIR)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # We ALWAYS load the original base model first
    base_model = AutoModelForCausalLM.from_pretrained(
        SFT_MODEL_DIR, 
        dtype=torch.bfloat16, 
        device_map="auto"
    )
    base_model.config.use_cache = False
    
    # --- Check if we are resuming or starting fresh ---
    if RESUME_FROM:
        print(f"Resuming training! Loading existing LoRA adapter from: {RESUME_FROM}")
        from peft import PeftModel
        # Load the previous adapter and ensure it is set up for further training
        policy_model = PeftModel.from_pretrained(base_model, RESUME_FROM, is_trainable=True)
    else:
        print("Initializing fresh LoRA adapters...")
        lora_config = LoraConfig(
            r=16, 
            lora_alpha=32, 
            lora_dropout=0.0, 
            bias="none", 
            task_type="CAUSAL_LM",
            target_modules=["q_proj","k_proj","v_proj","o_proj",
                            "gate_proj","up_proj","down_proj"]
        )
        policy_model = get_peft_model(base_model, lora_config)
        
    policy_model.gradient_checkpointing_enable()

    print("Configuring GRPO Trainer...")
    cfg = GRPOConfig(
        output_dir=output_dir,
        per_device_batch_size=PER_DEVICE_BS,
        num_generations=NUM_GENERATIONS,
        grad_accumulation_steps=GRAD_ACCUM,
        learning_rate=LEARNING_RATE,
        beta=BETA,
        epsilon=EPSILON,
        max_prompt_length=MAX_PROMPT_LEN,
        max_completion_length=MAX_COMPLETION_LEN,
        logging_steps=LOG_STEPS,
        save_steps=SAVE_STEPS,
        eval_steps=EVAL_STEPS,
        num_epochs=NUM_EPOCHS
    )

    trainer = GRPOTrainer(
        model=policy_model,
        tokenizer=tokenizer,
        reward_funcs=[format_reward, correctness_reward],
        config=cfg,
        train_dataset=train_ds,
        eval_dataset=test_ds
    )

    tb_logdir = os.path.join(output_dir, "logs")
    print(f"Starting Training! To view metrics, open a separate terminal and run:")
    print(f"    tensorboard --logdir {tb_logdir}")
    trainer.train()


if __name__ == "__main__":
    main(TARGET_DIR)
