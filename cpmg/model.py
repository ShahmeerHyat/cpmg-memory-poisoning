"""CPMG cross-encoder: DeBERTa-v3-base sequence classifier with provenance special tokens."""
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from cpmg.features import add_prov_tokens

BASE = "microsoft/deberta-v3-base"


def load_tokenizer(path=BASE):
    tok = AutoTokenizer.from_pretrained(path)
    if path == BASE:
        add_prov_tokens(tok)
    return tok


def load_model(tok, path=BASE, grad_ckpt=False):
    model = AutoModelForSequenceClassification.from_pretrained(path, num_labels=2, dtype=torch.float32)
    if path == BASE:
        model.resize_token_embeddings(len(tok))
    if grad_ckpt:
        model.gradient_checkpointing_enable()
    return model


def collate(batch, pad_id):
    n = max(len(b["ids"]) for b in batch)
    ids = torch.full((len(batch), n), pad_id, dtype=torch.long)
    att = torch.zeros((len(batch), n), dtype=torch.long)
    for i, b in enumerate(batch):
        ids[i, : len(b["ids"])] = torch.tensor(b["ids"])
        att[i, : len(b["ids"])] = 1
    out = {"input_ids": ids, "attention_mask": att}
    if "label" in batch[0]:
        out["labels"] = torch.tensor([b["label"] for b in batch])
    return out


@torch.no_grad()
def score(model, encoded, pad_id, device, batch_size=32):
    """Risk score = P(poisoned). `encoded` is a list of token-id lists. Sorted by length for speed."""
    model.eval()
    order = sorted(range(len(encoded)), key=lambda i: len(encoded[i]))
    out = [0.0] * len(encoded)
    for s in range(0, len(order), batch_size):
        idx = order[s: s + batch_size]
        b = collate([{"ids": encoded[i]} for i in idx], pad_id)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(input_ids=b["input_ids"].to(device), attention_mask=b["attention_mask"].to(device)).logits
        p = torch.softmax(logits.float(), -1)[:, 1].cpu().tolist()
        for i, v in zip(idx, p):
            out[i] = v
    return out
