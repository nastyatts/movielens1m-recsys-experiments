from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer, TrainingArguments

from common import EMB, LLM_DATA, MOVIE_ORDER, OUT, write_json


LLM_MOVIE_TOKEN_DATA = Path(__file__).resolve().parents[1] / "data" / "llm_movie_tokens"
MOVIE_TOKEN_MAP = LLM_DATA / "movie_token_map.json"
MOVIE_TOKEN_OUT = OUT / "movie_token_lm"
GLOBAL_PROJECTION_OUT = OUT / "movie_token_lm_projection"


class MovieTokenLMDataset(Dataset):
    def __init__(self, path: Path, tokenizer, max_seq_length: int, token_id_to_movie_pos: dict[int, int], max_samples: int = 0):
        self.rows = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.rows.append(json.loads(line))
                if max_samples > 0 and len(self.rows) >= max_samples:
                    break
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.token_id_to_movie_pos = token_id_to_movie_pos

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.rows[idx]
        prompt = row["input_text"]
        target_text = row["target_text"]
        prompt_ids = self.tokenizer(prompt, add_special_tokens=True)["input_ids"]
        answer_ids = self.tokenizer(target_text, add_special_tokens=False)["input_ids"]
        if len(answer_ids) != 1:
            raise ValueError(f"target_text must tokenize to exactly one movie token: {target_text!r} -> {answer_ids}")
        target_token_id = int(answer_ids[0])
        if target_token_id not in self.token_id_to_movie_pos:
            raise ValueError(f"target_text is not a known movie token: {target_text!r} -> {answer_ids}")
        input_ids = prompt_ids[-self.max_seq_length :]
        pad_len = self.max_seq_length - len(input_ids)
        if pad_len > 0:
            input_ids += [self.tokenizer.pad_token_id] * pad_len
        attention_mask = [1 if token_id != self.tokenizer.pad_token_id else 0 for token_id in input_ids]
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(self.token_id_to_movie_pos[target_token_id], dtype=torch.long),
        }


def load_token_map() -> dict:
    if not MOVIE_TOKEN_MAP.exists():
        raise FileNotFoundError(f"Missing {MOVIE_TOKEN_MAP}. Run scripts/40_build_movie_token_lm_data.py first.")
    return json.loads(MOVIE_TOKEN_MAP.read_text(encoding="utf-8"))


def ensure_movie_tokens(tokenizer, token_map: dict) -> None:
    tokens = list(token_map["movie_id_to_token"].values())
    tokenizer.add_special_tokens({"additional_special_tokens": tokens})
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token


def refresh_token_ids(tokenizer, token_map: dict) -> dict:
    movie_id_to_token = token_map["movie_id_to_token"]
    movie_id_to_token_id = {str(movie_id): int(tokenizer.convert_tokens_to_ids(token)) for movie_id, token in movie_id_to_token.items()}
    token_id_to_movie_id = {str(token_id): int(movie_id) for movie_id, token_id in movie_id_to_token_id.items()}
    out = dict(token_map)
    out["movie_id_to_token_id"] = movie_id_to_token_id
    out["token_id_to_movie_id"] = token_id_to_movie_id
    out["tokenizer_vocab_size"] = len(tokenizer)
    return out


def init_movie_token_embeddings(model, token_map: dict, projection: nn.Linear) -> None:
    if not EMB.exists() or not MOVIE_ORDER.exists():
        raise FileNotFoundError(f"Missing movie text embeddings: {EMB} / {MOVIE_ORDER}")
    emb = np.load(EMB).astype(np.float32)
    order = np.load(MOVIE_ORDER).astype(int).tolist()
    pos = {int(movie_id): i for i, movie_id in enumerate(order)}
    input_emb = model.get_input_embeddings()
    output_emb = model.get_output_embeddings()
    device = input_emb.weight.device
    dtype = input_emb.weight.dtype
    with torch.no_grad():
        for movie_id_str, token_id in token_map["movie_id_to_token_id"].items():
            movie_id = int(movie_id_str)
            if movie_id not in pos:
                continue
            vec = torch.tensor(emb[pos[movie_id]], dtype=torch.float32, device=device).unsqueeze(0)
            projected = projection(vec).to(dtype=dtype).squeeze(0)
            input_emb.weight[int(token_id)].copy_(projected)
            if output_emb is not None and output_emb.weight.shape == input_emb.weight.shape:
                output_emb.weight[int(token_id)].copy_(projected)


def movie_text_vectors_for_tokens(token_map: dict) -> tuple[list[int], torch.Tensor]:
    emb = np.load(EMB).astype(np.float32)
    order = np.load(MOVIE_ORDER).astype(int).tolist()
    pos = {int(movie_id): i for i, movie_id in enumerate(order)}
    token_ids = []
    vectors = []
    for movie_id_str, token_id in token_map["movie_id_to_token_id"].items():
        token_ids.append(int(token_id))
        movie_id = int(movie_id_str)
        if movie_id in pos:
            vectors.append(emb[pos[movie_id]])
        else:
            vectors.append(np.zeros(emb.shape[1], dtype=np.float32))
    return token_ids, torch.tensor(np.stack(vectors), dtype=torch.float32)


class MovieTokenProjectionLM(nn.Module):
    def __init__(self, peft_model, projection: nn.Linear, movie_token_ids: list[int], movie_text_vectors: torch.Tensor):
        super().__init__()
        self.peft_model = peft_model
        self.projection = projection
        input_emb = peft_model.get_input_embeddings()
        vocab_size = input_emb.weight.shape[0]
        hidden_size = input_emb.weight.shape[1]
        self.movie_output_head = nn.Linear(hidden_size, len(movie_token_ids), bias=False).to(input_emb.weight.device)
        token_to_movie_pos = torch.full((vocab_size,), -1, dtype=torch.long)
        token_to_movie_pos[torch.tensor(movie_token_ids, dtype=torch.long)] = torch.arange(len(movie_token_ids), dtype=torch.long)
        self.register_buffer("token_to_movie_pos", token_to_movie_pos, persistent=False)
        self.register_buffer("movie_text_vectors", movie_text_vectors.float(), persistent=True)
        output_emb = peft_model.get_output_embeddings()
        with torch.no_grad():
            if output_emb is not None and output_emb.weight.shape[1] == hidden_size:
                token_ids = torch.tensor(movie_token_ids, dtype=torch.long, device=output_emb.weight.device)
                init_weight = output_emb.weight[token_ids].detach().to(self.movie_output_head.weight.device, dtype=self.movie_output_head.weight.dtype)
            else:
                init_weight = self.projection(movie_text_vectors.to(self.projection.weight.device)).detach()
                init_weight = init_weight.to(self.movie_output_head.weight.device, dtype=self.movie_output_head.weight.dtype)
            self.movie_output_head.weight.copy_(init_weight)

    def _backbone_forward(self, token_embeds: torch.Tensor, attention_mask: torch.Tensor):
        base_model = self.peft_model.base_model.model
        if hasattr(base_model, "model"):
            return base_model.model(
                inputs_embeds=token_embeds,
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
            )
        if hasattr(base_model, "transformer"):
            return base_model.transformer(
                inputs_embeds=token_embeds,
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
            )
        outputs = self.peft_model(
            inputs_embeds=token_embeds,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
        return type("BackboneOutput", (), {"last_hidden_state": outputs.hidden_states[-1]})()

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        token_embeds = self.peft_model.get_input_embeddings()(input_ids)
        token_pos = self.token_to_movie_pos[input_ids]
        mask = token_pos.ge(0)
        if mask.any():
            unique_pos = torch.unique(token_pos[mask])
            projected = self.projection(self.movie_text_vectors[unique_pos].to(token_embeds.device)).to(dtype=token_embeds.dtype)
            projected_by_pos = torch.zeros(
                (self.movie_text_vectors.shape[0], token_embeds.shape[-1]),
                device=token_embeds.device,
                dtype=token_embeds.dtype,
            )
            projected_by_pos[unique_pos] = projected
            token_embeds = token_embeds.clone()
            token_embeds[mask] = projected_by_pos[token_pos[mask]]
        outputs = self._backbone_forward(token_embeds, attention_mask)
        last_pos = attention_mask.sum(dim=1).clamp(min=1) - 1
        batch_idx = torch.arange(token_embeds.shape[0], device=token_embeds.device)
        last_hidden = outputs.last_hidden_state[batch_idx, last_pos]
        movie_logits = self.movie_output_head(last_hidden.to(self.movie_output_head.weight.dtype))
        loss = F.cross_entropy(movie_logits, labels) if labels is not None else None
        return {"loss": loss, "logits": movie_logits}

    def get_input_embeddings(self):
        return self.peft_model.get_input_embeddings()


class MovieTokenProjectionTrainer(Trainer):
    def __init__(self, *args, projection_learning_rate: float, movie_head_learning_rate: float, **kwargs):
        super().__init__(*args, **kwargs)
        self.projection_learning_rate = projection_learning_rate
        self.movie_head_learning_rate = movie_head_learning_rate

    def create_optimizer(self):
        if self.optimizer is None:
            projection_params = []
            movie_head_params = []
            other_params = []
            for name, param in self.model.named_parameters():
                if not param.requires_grad:
                    continue
                if name.startswith("projection."):
                    projection_params.append(param)
                elif name.startswith("movie_output_head."):
                    movie_head_params.append(param)
                else:
                    other_params.append(param)
            self.optimizer = torch.optim.AdamW(
                [
                    {"params": other_params, "lr": self.args.learning_rate},
                    {"params": projection_params, "lr": self.projection_learning_rate},
                    {"params": movie_head_params, "lr": self.movie_head_learning_rate},
                ],
                betas=(0.9, 0.999),
                eps=1e-8,
            )
        return self.optimizer


def save_movie_token_embedding_rows(model, token_ids: list[int], output_dir: Path) -> None:
    input_emb = model.get_input_embeddings().weight.detach().cpu()[token_ids].clone()
    output_emb_layer = model.get_output_embeddings()
    payload = {"token_ids": token_ids, "input_embeddings": input_emb}
    if output_emb_layer is not None:
        payload["output_embeddings"] = output_emb_layer.weight.detach().cpu()[token_ids].clone()
    torch.save(payload, output_dir / "movie_token_embeddings.pt")


def sync_input_embeddings_from_projection(model, projection: nn.Linear, movie_token_ids: list[int], movie_text_vectors: torch.Tensor) -> None:
    input_emb = model.get_input_embeddings()
    device = input_emb.weight.device
    token_ids_tensor = torch.tensor(movie_token_ids, dtype=torch.long, device=device)
    with torch.no_grad():
        projected = projection(movie_text_vectors.to(projection.weight.device))
        input_emb.weight[token_ids_tensor] = projected.to(device=device, dtype=input_emb.weight.dtype)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="meta-llama/Llama-3.2-3B-Instruct")
    parser.add_argument("--run_name", required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--max_train_samples", type=int, default=1000)
    parser.add_argument("--max_val_samples", type=int, default=300)
    parser.add_argument("--max_steps", type=int, default=100)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--projection_learning_rate", type=float, default=1e-3)
    parser.add_argument("--movie_head_learning_rate", type=float, default=1e-3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = MOVIE_TOKEN_OUT / "runs" / args.run_name
    if run_dir.exists() and not args.overwrite:
        raise FileExistsError(f"Run directory exists: {run_dir}. Use --overwrite to replace.")
    run_dir.mkdir(parents=True, exist_ok=True)
    GLOBAL_PROJECTION_OUT.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "train_config.json", vars(args))

    try:
        import bitsandbytes  # noqa: F401
    except Exception as exc:
        raise RuntimeError("4-bit QLoRA requires bitsandbytes.") from exc

    token_map = load_token_map()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    ensure_movie_tokens(tokenizer, token_map)
    token_map = refresh_token_ids(tokenizer, token_map)
    tokenizer.save_pretrained(str(run_dir))
    write_json(run_dir / "movie_token_map.json", token_map)

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(args.model_name, quantization_config=quant_config, device_map="auto")
    model.resize_token_embeddings(len(tokenizer))
    model.config.use_cache = False

    hidden_size = int(model.get_input_embeddings().weight.shape[1])
    text_dim = int(np.load(EMB, mmap_mode="r").shape[1])
    projection = nn.Linear(text_dim, hidden_size).to(model.get_input_embeddings().weight.device)
    nn.init.normal_(projection.weight, std=0.02)
    nn.init.zeros_(projection.bias)
    init_movie_token_embeddings(model, token_map, projection)
    projection_config = {
        "source_embedding": str(EMB),
        "movie_id_order": str(MOVIE_ORDER),
        "text_dim": text_dim,
        "hidden_size": hidden_size,
        "projection_trained": True,
        "input_movie_embeddings_synced_from_projection": True,
        "scoring_head": "movie_output_head",
    }
    write_json(
        run_dir / "projection_config.json",
        projection_config,
    )
    write_json(
        GLOBAL_PROJECTION_OUT / "projection_config.json",
        {"run_dir": str(run_dir), **projection_config},
    )

    model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)
    movie_token_ids, movie_text_vectors = movie_text_vectors_for_tokens(token_map)
    token_id_to_movie_pos = {int(token_id): idx for idx, token_id in enumerate(movie_token_ids)}
    model.print_trainable_parameters()
    wrapped_model = MovieTokenProjectionLM(model, projection, movie_token_ids, movie_text_vectors)

    train_ds = MovieTokenLMDataset(
        LLM_MOVIE_TOKEN_DATA / "train_movie_token_lm.jsonl",
        tokenizer,
        args.max_seq_length,
        token_id_to_movie_pos,
        args.max_train_samples,
    )
    val_ds = MovieTokenLMDataset(
        LLM_MOVIE_TOKEN_DATA / "val_movie_token_lm.jsonl",
        tokenizer,
        args.max_seq_length,
        token_id_to_movie_pos,
        args.max_val_samples,
    )

    training_args = TrainingArguments(
        output_dir=str(run_dir),
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        fp16=True,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=max(25, args.max_steps // 2),
        save_strategy="no",
        report_to=[],
        remove_unused_columns=False,
    )
    trainer = MovieTokenProjectionTrainer(
        model=wrapped_model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds if len(val_ds) else None,
        projection_learning_rate=args.projection_learning_rate,
        movie_head_learning_rate=args.movie_head_learning_rate,
    )
    trainer.train()
    sync_input_embeddings_from_projection(model, projection, movie_token_ids, movie_text_vectors)
    model.save_pretrained(str(run_dir))
    tokenizer.save_pretrained(str(run_dir))
    torch.save(projection.state_dict(), run_dir / "projection.pt")
    torch.save(projection.state_dict(), GLOBAL_PROJECTION_OUT / "projection.pt")
    torch.save(wrapped_model.movie_output_head.state_dict(), run_dir / "movie_output_head.pt")
    save_movie_token_embedding_rows(model, movie_token_ids, run_dir)
    (run_dir / "run_commands.txt").write_text(
        f"python scripts/41_train_movie_token_lm_qlora.py --model_name {args.model_name} --run_name {args.run_name}\n"
        f"python scripts/42_eval_movie_token_lm.py --model_name {args.model_name} --run_name {args.run_name} --split test\n",
        encoding="utf-8",
    )
    print(f"saved movie-token LM run: {run_dir}")


if __name__ == "__main__":
    main()
