"""
Optimized GRPO implementation with Eval Callbacks, Metrics Accumulation, 
Native TensorBoard Logging, and Token-Level Progress Bars.
"""

from __future__ import annotations

import collections
import gc
import os
import time
import json
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter
from transformers import PreTrainedModel, PreTrainedTokenizerBase, LogitsProcessor, LogitsProcessorList
from tqdm import tqdm

try:
    from peft import PeftModel
except ImportError:
    PeftModel = None  # type: ignore

RewardFn = Callable[..., List[float]]

# --- Custom Logits Processor for Token-Level Progress Tracking ---
class TqdmLogitsProcessor(LogitsProcessor):
    def __init__(self, pbar):
        self.pbar = pbar

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        self.pbar.update(1)
        return scores


@dataclass
class GRPOConfig:
    output_dir: str = "grpo_output"

    # optimization
    learning_rate: float = 5e-6
    num_epochs: int = 1
    per_device_batch_size: int = 1          
    grad_accumulation_steps: int = 4        
    max_grad_norm: float = 1.0
    warmup_ratio: float = 0.05
    weight_decay: float = 0.0

    # GRPO-specific
    num_generations: int = 4                
    beta: float = 0.04                      
    epsilon: float = 0.2                    

    # generation
    max_prompt_length: int = 512
    max_completion_length: int = 768
    temperature: float = 0.9
    top_p: float = 1.0

    # logging / checkpointing / eval
    logging_steps: int = 1
    save_steps: int = 50
    eval_steps: int = 50                    
    save_total_limit: int = 2
    seed: int = 42

    # misc
    fp16: bool = True


class GRPOTrainer:
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        reward_funcs: List[RewardFn],
        config: GRPOConfig,
        train_dataset,
        eval_dataset=None,                  
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.reward_funcs = reward_funcs
        self.cfg = config
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        self.device = next(model.parameters()).device
        self._is_peft = PeftModel is not None and isinstance(model, PeftModel)
        
        os.makedirs(self.cfg.output_dir, exist_ok=True)
        
        tb_log_dir = os.path.join(self.cfg.output_dir, "logs")
        self.tb_writer = SummaryWriter(log_dir=tb_log_dir)
        tqdm.write(f"[GRPO] TensorBoard logging initialized at: {tb_log_dir}")
        
        torch.manual_seed(self.cfg.seed)

    # ─────────────────────────── core math ────────────────────────────────

    @staticmethod
    def _group_advantages(rewards: torch.Tensor, group_size: int) -> torch.Tensor:
        grouped = rewards.view(-1, group_size)
        mean = grouped.mean(dim=1, keepdim=True)
        std = grouped.std(dim=1, keepdim=True)
        adv = (grouped - mean) / (std + 1e-8)
        return adv.view(-1)

    def _grpo_loss(self, new_logp, old_logp, ref_logp, advantages, completion_mask):
        log_ratio = new_logp - old_logp
        ratio = log_ratio.exp()

        adv = advantages.unsqueeze(1) 
        unclipped = ratio * adv
        clipped = torch.clamp(ratio, 1.0 - self.cfg.epsilon, 1.0 + self.cfg.epsilon) * adv
        policy_obj = torch.min(unclipped, clipped)

        log_ref_minus_pi = ref_logp - new_logp
        kl = log_ref_minus_pi.exp() - log_ref_minus_pi - 1.0

        per_token = policy_obj - self.cfg.beta * kl

        mask = completion_mask.float()
        seq_len = mask.sum(dim=1).clamp(min=1.0)
        seq_obj = (per_token * mask).sum(dim=1) / seq_len
        loss = -seq_obj.mean()

        denom = mask.sum().clamp(min=1.0)
        is_clipped = ((ratio < 1 - self.cfg.epsilon) | (ratio > 1 + self.cfg.epsilon)).float()
        metrics = {
            "loss": loss.item(),
            "kl": ((kl * mask).sum() / denom).item(),
            "ratio_mean": ((ratio * mask).sum() / denom).item(),
            "clip_frac": ((is_clipped * mask).sum() / denom).item(),
            "completion_len": mask.sum(dim=1).mean().item(),
        }
        return loss, metrics

    # ───────────────────────── log-prob computation ───────────────────────

    @staticmethod
    def _token_logprobs(model, full_ids, full_mask, completion_start) -> torch.Tensor:
        outputs = model(input_ids=full_ids, attention_mask=full_mask, use_cache=False)
        logits = outputs.logits  

        target_ids = full_ids[:, completion_start:]                       
        target_logits = logits[:, completion_start - 1 : -1, :]           

        gathered = target_logits.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
        token_logp = (gathered - target_logits.logsumexp(dim=-1)).float()
        return token_logp  

    def _ref_logprobs(self, full_ids, full_mask, completion_start) -> torch.Tensor:
        was_training = self.model.training
        self.model.eval()
        with torch.no_grad():
            if self._is_peft:
                with self.model.disable_adapter():
                    res = self._token_logprobs(self.model, full_ids, full_mask, completion_start)
            else:
                res = self._token_logprobs(self.model, full_ids, full_mask, completion_start)
        
        if was_training:
            self.model.train()
        return res

    # ───────────────────────────── sampling ───────────────────────────────

    def _generate_group(self, prompt_ids, prompt_mask, pbar=None):
        G = self.cfg.num_generations
        prompt_ids = prompt_ids.repeat_interleave(G, dim=0)
        prompt_mask = prompt_mask.repeat_interleave(G, dim=0)
        P = prompt_ids.size(1)

        was_training = self.model.training
        self.model.eval()
        
        orig_cache = getattr(self.model.config, "use_cache", False)
        self.model.config.use_cache = True
        
        processors = LogitsProcessorList()
        if pbar is not None:
            processors.append(TqdmLogitsProcessor(pbar))
        
        with torch.no_grad():
            out = self.model.generate(
                input_ids=prompt_ids,
                attention_mask=prompt_mask,
                max_new_tokens=self.cfg.max_completion_length,
                do_sample=True,
                temperature=self.cfg.temperature,
                top_p=self.cfg.top_p,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                logits_processor=processors # <-- Attach the progress tracker here
            )
            
        self.model.config.use_cache = orig_cache
        if was_training:
            self.model.train()

        completion_ids = out[:, P:]                                       
        eos_id = self.tokenizer.eos_token_id
        is_eos = completion_ids == eos_id
        N, C = completion_ids.shape
        eos_pos = torch.full((N,), C, dtype=torch.long, device=completion_ids.device)
        has_eos = is_eos.any(dim=1)
        if has_eos.any():
            first_eos = is_eos.float().argmax(dim=1)
            eos_pos[has_eos] = first_eos[has_eos]
        col = torch.arange(C, device=completion_ids.device).unsqueeze(0)
        completion_mask = (col <= eos_pos.unsqueeze(1)).long()

        full_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        return out, completion_ids, completion_mask, full_mask, P

    # ────────────────────────────── rewards ───────────────────────────────

    def _score(self, completion_texts, reward_kwargs):
        completions_conv = [[{"role": "assistant", "content": t}] for t in completion_texts]
        total = torch.zeros(len(completion_texts), device=self.device)
        per_fn = {}
        for fn in self.reward_funcs:
            scores = fn(completions=completions_conv, **reward_kwargs)
            scores_t = torch.tensor(scores, dtype=torch.float32, device=self.device)
            total = total + scores_t
            per_fn[f"rewards/{fn.__name__}"] = scores_t.mean().item()
        return total, per_fn

    # ──────────────────────────── train loop ──────────────────────────────

    def _tokenize_prompts(self, prompts) -> Dict[str, torch.Tensor]:
        texts = [self.tokenizer.apply_chat_template(p, tokenize=False, add_generation_prompt=True) for p in prompts]
        enc = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=self.cfg.max_prompt_length)
        return {k: v.to(self.device) for k, v in enc.items()}

    def _iter_batches(self):
        ds = self.train_dataset
        B = self.cfg.per_device_batch_size
        n = len(ds)
        for epoch in range(self.cfg.num_epochs):
            order = torch.randperm(n, generator=torch.Generator().manual_seed(self.cfg.seed + epoch)).tolist()
            for i in range(0, n, B):
                idx = order[i : i + B]
                rows = [ds[int(j)] for j in idx]
                yield epoch, rows

    def _make_scheduler(self, optimizer, total_steps: int):
        warmup = max(1, int(total_steps * self.cfg.warmup_ratio))
        def lr_lambda(step):
            if step < warmup: return step / warmup
            progress = (step - warmup) / max(1, total_steps - warmup)
            return 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.141592653589793)).item())
        return LambdaLR(optimizer, lr_lambda)

    def _log(self, payload: Dict[str, Any]):
        step = payload.get("step", 0)
        compact = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in payload.items()}
        # Must use tqdm.write to prevent visual artifacts with the bars
        tqdm.write(str(compact))
        
        for k, v in payload.items():
            if k == "step": 
                continue
            if isinstance(v, (int, float)):
                self.tb_writer.add_scalar(k, v, step)

    def evaluate(self, step: int):
        if not self.eval_dataset: return
        tqdm.write(f"\n--- Running Evaluation at step {step} ---")
        
        eval_slice = [self.eval_dataset[i] for i in range(min(200, len(self.eval_dataset)))]
        prompts = [r["prompt"] for r in eval_slice]
        answers = [r["answer"] for r in eval_slice]
        
        B = self.cfg.per_device_batch_size * 2
        all_texts = []
        
        self.model.eval()
        orig_cache = getattr(self.model.config, "use_cache", False)
        self.model.config.use_cache = True
        
        with torch.no_grad():
            for i in range(0, len(prompts), B):
                batch_prompts = prompts[i:i+B]
                enc = self._tokenize_prompts(batch_prompts)
                P = enc["input_ids"].size(1)
                
                out = self.model.generate(
                    input_ids=enc["input_ids"],
                    attention_mask=enc["attention_mask"],
                    max_new_tokens=self.cfg.max_completion_length,
                    do_sample=False, 
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
                completion_ids = out[:, P:]
                texts = self.tokenizer.batch_decode(completion_ids, skip_special_tokens=True)
                all_texts.extend(texts)
                
        self.model.config.use_cache = orig_cache
        self.model.train()
        
        _, per_fn = self._score(all_texts, {"answer": answers})
        
        eval_metrics = {"step": step}
        for k, v in per_fn.items(): 
            eval_metrics[f"eval_{k}"] = v
            
        self._log(eval_metrics)

        # --- NEW: Save Evaluation Generations ---
        import json # Local import or add to top of file
        eval_results = []
        for p, g, t in zip(prompts, answers, all_texts):
            eval_results.append({
                "prompt": p,
                "gold_answer": g,
                "generated_text": t
            })
            
        eval_file_path = os.path.join(self.cfg.output_dir, f"eval_generations_step_{step}.json")
        with open(eval_file_path, "w", encoding="utf-8") as f:
            json.dump(eval_results, f, indent=4)
            
        tqdm.write(f"[GRPO] Saved evaluation generations to {eval_file_path}")
        tqdm.write("----------------------------------------\n")

    def train(self):
        cfg = self.cfg
        B, G, accum = cfg.per_device_batch_size, cfg.num_generations, cfg.grad_accumulation_steps

        trainable = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = AdamW(trainable, lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

        prompts_per_optim_step = B * accum
        total_optim_steps = (len(self.train_dataset) // prompts_per_optim_step) * cfg.num_epochs
        scheduler = self._make_scheduler(optimizer, total_optim_steps)

        self.model.train()
        optimizer.zero_grad()

        global_step = 0
        micro = 0
        t0 = time.time()

        accum_metrics = collections.defaultdict(float)
        accum_per_fn = collections.defaultdict(float)
        accum_reward = 0.0

        # === INIT MULTI-LEVEL PROGRESS BARS ===
        pbar_main = tqdm(total=total_optim_steps, desc="Overall Training", position=0, unit="step", colour="green")
        pbar_inner = tqdm(total=accum, desc="Micro-batches", position=1, leave=False, unit="micro", colour="blue")

        for epoch, rows in self._iter_batches():
            prompts = [r["prompt"] for r in rows]
            answers = [r["answer"] for r in rows]
            enc = self._tokenize_prompts(prompts)

            pbar_inner.set_description("Phase: Generating")
            
            # Temporary 3rd bar just for generation progress
            pbar_gen = tqdm(total=cfg.max_completion_length, desc="Tokens Generated", position=2, leave=False, colour="red")
            
            full_ids, completion_ids, completion_mask, full_mask, P = self._generate_group(
                enc["input_ids"], enc["attention_mask"], pbar=pbar_gen
            )
            pbar_gen.close()

            completion_texts = self.tokenizer.batch_decode(
                completion_ids * completion_mask + (1 - completion_mask) * self.tokenizer.pad_token_id,
                skip_special_tokens=True,
            )
            answers_rep = [a for a in answers for _ in range(G)]

            pbar_inner.set_description("Phase: Scoring")
            rewards, per_fn = self._score(completion_texts, {"answer": answers_rep})
            advantages = self._group_advantages(rewards, G)

            pbar_inner.set_description("Phase: Forward & Loss")
            new_logp = self._token_logprobs(self.model, full_ids, full_mask, P)
            old_logp = new_logp.detach()
            ref_logp = self._ref_logprobs(full_ids, full_mask, P)

            loss, metrics = self._grpo_loss(new_logp, old_logp, ref_logp, advantages, completion_mask)
            
            pbar_inner.set_description("Phase: Backprop")
            (loss / accum).backward()
            
            micro += 1
            pbar_inner.update(1)

            for k, v in metrics.items(): accum_metrics[f"metrics/{k}"] += v / accum
            for k, v in per_fn.items(): accum_per_fn[k] += v / accum
            accum_reward += rewards.mean().item() / accum

            if micro % accum == 0:
                pbar_inner.set_description("Phase: Optimizing")
                grad_norm = torch.nn.utils.clip_grad_norm_(trainable, cfg.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                pbar_main.update(1)
                pbar_main.set_postfix({
                    "reward": f"{accum_reward:.3f}", 
                    "kl": f"{accum_metrics['metrics/kl']:.3f}"
                })

                # FOOLPROOF RESET: Close and completely recreate the inner bar
                pbar_inner.close()
                pbar_inner = tqdm(total=accum, desc="Micro-batches", position=1, leave=False, unit="micro", colour="blue")

                if global_step % cfg.logging_steps == 0:
                    self._log({
                        "step": global_step,
                        "training/epoch": epoch,
                        "training/lr": scheduler.get_last_lr()[0],
                        "metrics/reward_mean": accum_reward,
                        "metrics/grad_norm": grad_norm.item(),
                        **accum_per_fn,
                        **accum_metrics,
                    })

                accum_metrics.clear()
                accum_per_fn.clear()
                accum_reward = 0.0

                if global_step % cfg.eval_steps == 0:
                    self.evaluate(global_step)

                if global_step % cfg.save_steps == 0:
                    self.save(os.path.join(cfg.output_dir, f"checkpoint-{global_step}"))

        pbar_inner.close()
        pbar_main.close()
        self.save(os.path.join(cfg.output_dir, "final"))
        self.tb_writer.close() 

    # ───────────────────────────── saving ─────────────────────────────────

    def save(self, path: str):
        os.makedirs(path, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        tqdm.write(f"[GRPO] saved to {path}")
