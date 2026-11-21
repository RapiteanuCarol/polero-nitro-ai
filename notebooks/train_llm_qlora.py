import argparse
import gc
import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    BitsAndBytesConfig,
    get_cosine_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType


MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
MAX_LENGTH = 384
BATCH_SIZE = 2
GRAD_ACCUM = 8
EPOCHS = 2
LR = 2e-4
SEED = 42


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_prompt(row):
    q_curr = str(row["interview_question"]).strip() if pd.notna(row["interview_question"]) else ""
    a_curr = str(row["interview_answer"]).strip() if pd.notna(row["interview_answer"]) else ""
    pres = str(row["president"]).strip() if pd.notna(row["president"]) else "Necunoscut"
    ctx = str(row["context"]).strip() if pd.notna(row["context"]) else "Necunoscut"

    parts = [
        "Clasifica strategia de raspuns a presedintelui la intrebarea jurnalistului.",
        f"Context: {ctx} | Presedinte: {pres}",
        f"Intrebare: {q_curr}",
        f"Raspuns: {a_curr}",
    ]

    prev_q = str(row["prev_question"]).strip() if pd.notna(row["prev_question"]) else ""
    prev_a = str(row["prev_answer"]).strip() if pd.notna(row["prev_answer"]) else ""
    if prev_q or prev_a:
        gap = row["turn_gap"] if pd.notna(row["turn_gap"]) else -1
        parts.append(f"Schimb anterior (turn_gap={gap}):")
        if prev_q:
            parts.append(f"Intrebare anterioara: {prev_q[:250]}")
        if prev_a:
            parts.append(f"Raspuns anterior: {prev_a[:250]}")

    return "\n".join(parts)


class EvasionDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            max_length=max_length,
            padding=False,
        )
        self.labels = labels

    def __len__(self):
        return len(self.encodings["input_ids"])

    def __getitem__(self, idx):
        item = {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
        }
        if self.labels is not None:
            item["labels"] = int(self.labels[idx])
        return item


def make_collate_fn(pad_token_id):
    def collate_fn(batch):
        max_len = max(len(x["input_ids"]) for x in batch)
        input_ids = []
        attention_mask = []
        labels = []
        for x in batch:
            pad_len = max_len - len(x["input_ids"])
            input_ids.append(x["input_ids"] + [pad_token_id] * pad_len)
            attention_mask.append(x["attention_mask"] + [0] * pad_len)
            if "labels" in x:
                labels.append(x["labels"])

        out = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }
        if labels:
            out["labels"] = torch.tensor(labels, dtype=torch.long)
        return out

    return collate_fn


def create_model(num_labels, pad_token_id):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=num_labels,
        quantization_config=bnb_config,
        device_map="auto",
        max_memory={0: "11GiB"},
        dtype=torch.float16,
    )
    model.config.pad_token_id = pad_token_id
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        modules_to_save=["score"],
    )
    model = get_peft_model(model, lora_config)

    for name, param in model.named_parameters():
        if "score" in name and "weight" in name:
            nn.init.normal_(param, mean=0.0, std=0.005)

    return model


@torch.no_grad()
def predict_probs(model, loader, device):
    model.eval()
    all_probs = []
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        with torch.amp.autocast("cuda", dtype=torch.float16):
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        probs = F.softmax(logits.float(), dim=-1).cpu().numpy()
        all_probs.append(probs)
    return np.vstack(all_probs).astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--merge-only", action="store_true")
    args = parser.parse_args()

    set_seed(SEED)
    os.makedirs("predictions/llm_folds", exist_ok=True)

    with open("data/processed/classes_order.txt", "r", encoding="utf-8") as f:
        classes = [line.strip().split(": ", 1)[1] for line in f if line.strip()]
    num_labels = len(classes)
    label_to_id = {c: i for i, c in enumerate(classes)}

    train_df = pd.read_csv("data/processed/train_context.csv")
    test_df = pd.read_csv("data/processed/test_context.csv")

    y_all = train_df["evasion_label"].map(label_to_id).values
    folds = train_df["fold"].values

    if not args.merge_only:
        device = torch.device("cuda:0")
        train_texts = [build_prompt(row) for _, row in train_df.iterrows()]
        test_texts = [build_prompt(row) for _, row in test_df.iterrows()]

        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        collate_fn = make_collate_fn(tokenizer.pad_token_id)

        test_ds = EvasionDataset(test_texts, None, tokenizer, MAX_LENGTH)
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE * 4, shuffle=False, collate_fn=collate_fn)

        for fold in args.folds:
            oof_path = f"predictions/llm_folds/oof_fold_{fold}.npy"
            test_path = f"predictions/llm_folds/test_fold_{fold}.npy"

            if os.path.exists(oof_path) and os.path.exists(test_path):
                print(f"[Fold {fold}] Deja salvat, sarim peste.", flush=True)
                continue

            print(f"\n=== Incepere antrenare Fold {fold} ===", flush=True)
            t0 = time.time()

            trn_idx = np.where(folds != fold)[0]
            val_idx = np.where(folds == fold)[0]

            trn_ds = EvasionDataset([train_texts[i] for i in trn_idx], y_all[trn_idx], tokenizer, MAX_LENGTH)
            val_ds = EvasionDataset([train_texts[i] for i in val_idx], y_all[val_idx], tokenizer, MAX_LENGTH)

            trn_loader = DataLoader(trn_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
            val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE * 4, shuffle=False, collate_fn=collate_fn)

            counts = np.bincount(y_all[trn_idx], minlength=num_labels).astype(np.float32)
            weights = (len(trn_idx) / np.maximum(counts, 1.0)) ** 0.5
            weights = weights / weights.mean()
            criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, device=device))

            model = create_model(num_labels, tokenizer.pad_token_id)
            optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
            total_steps = (len(trn_loader) // GRAD_ACCUM) * EPOCHS
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=max(1, int(0.1 * total_steps)),
                num_training_steps=max(1, total_steps),
            )

            best_f1 = -1.0
            best_val_probs = None
            best_test_probs = None

            for epoch in range(1, EPOCHS + 1):
                model.train()
                optimizer.zero_grad()
                running_loss = 0.0
                ep_start = time.time()

                for step, batch in enumerate(trn_loader, start=1):
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    labels = batch["labels"].to(device)

                    with torch.amp.autocast("cuda", dtype=torch.float16):
                        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
                        loss = criterion(logits.float(), labels) / GRAD_ACCUM

                    loss.backward()
                    running_loss += loss.item() * GRAD_ACCUM

                    if step % GRAD_ACCUM == 0 or step == len(trn_loader):
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()
                        optimizer.zero_grad()
                        scheduler.step()

                    if step % 150 == 0 or step == len(trn_loader):
                        print(
                            f"[Fold {fold} | Ep {epoch}/{EPOCHS}] pas {step}/{len(trn_loader)} "
                            f"| loss={running_loss/step:.4f} | {time.time()-ep_start:.0f}s",
                            flush=True,
                        )

                val_probs = predict_probs(model, val_loader, device)
                val_preds = val_probs.argmax(axis=1)
                val_f1 = f1_score(y_all[val_idx], val_preds, average="macro")
                elapsed = time.time() - ep_start

                print(
                    f"[Fold {fold} | Epoca {epoch}/{EPOCHS} GATA] "
                    f"loss={running_loss/len(trn_loader):.4f} | val_macro_f1={val_f1:.4f} | timp={elapsed:.1f}s",
                    flush=True,
                )

                if val_f1 > best_f1:
                    best_f1 = val_f1
                    best_val_probs = val_probs
                    best_test_probs = predict_probs(model, test_loader, device)

            np.save(oof_path, best_val_probs)
            np.save(test_path, best_test_probs)
            print(
                f"[Fold {fold}] Salvat cu best_macro_f1={best_f1:.4f} (durata: {(time.time()-t0)/60:.1f} min)",
                flush=True,
            )

            del model, optimizer, scheduler
            gc.collect()
            torch.cuda.empty_cache()

    all_done = all(
        os.path.exists(f"predictions/llm_folds/oof_fold_{f}.npy")
        and os.path.exists(f"predictions/llm_folds/test_fold_{f}.npy")
        for f in range(5)
    )
    if all_done:
        oof_full = np.zeros((len(train_df), num_labels), dtype=np.float32)
        test_full = np.zeros((len(test_df), num_labels), dtype=np.float32)

        for fold in range(5):
            val_idx = np.where(folds == fold)[0]
            oof_full[val_idx] = np.load(f"predictions/llm_folds/oof_fold_{fold}.npy")
            test_full += np.load(f"predictions/llm_folds/test_fold_{fold}.npy") / 5.0

        overall_f1 = f1_score(y_all, oof_full.argmax(axis=1), average="macro")
        print(f"\n=== FINAL 5-FOLD OOF MACRO-F1: {overall_f1:.4f} ===", flush=True)

        np.save("predictions/oof_llm.npy", oof_full)
        np.save("predictions/test_llm.npy", test_full)
        print("Salvat predictions/oof_llm.npy:", oof_full.shape, flush=True)
        print("Salvat predictions/test_llm.npy:", test_full.shape, flush=True)


if __name__ == "__main__":
    main()
