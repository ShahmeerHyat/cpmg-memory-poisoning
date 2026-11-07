"""PIGuard (Li et al., ACL 2025) zero-shot baseline on the memory text.

Checkpoint: Hugging Face `leolee99/PIGuard` (MIT). Its custom class (modeling_piguard.py) is a
DeBERTa-v2 encoder with a linear classifier on the [CLS] hidden state; label 1 = "injection".
We rebuild that class locally and load the published weights, so no remote code is executed.
Usage: python -m baselines.piguard
"""
import os

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
from transformers import AutoTokenizer, DebertaV2Config, DebertaV2Model

from baselines.common import PRED_SPLITS, load, write_preds, write_run

REPO = "leolee99/PIGuard"


class PIGuardLocal(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.deberta = DebertaV2Model(cfg)
        self.classifier = torch.nn.Linear(cfg.hidden_size, cfg.num_labels)

    def forward(self, input_ids, attention_mask):
        h = self.deberta(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]
        return self.classifier(h)


def load_piguard(dev):
    path = snapshot_download(REPO, allow_patterns=["config.json", "model.safetensors", "*.json", "spm.model"])
    cfg = DebertaV2Config.from_pretrained(path)
    cfg.num_labels = 2
    m = PIGuardLocal(cfg)
    sd = load_file(os.path.join(path, "model.safetensors"))
    missing, unexpected = m.load_state_dict(sd, strict=False)
    assert not [k for k in missing if not k.endswith("position_ids")], missing
    tok = AutoTokenizer.from_pretrained(path)
    return m.to(dev).eval(), tok, unexpected


@torch.no_grad()
def score(m, tok, rows, dev, bs=32):
    out = []
    for s in range(0, len(rows), bs):
        b = tok([r["memory"] for r in rows[s:s + bs]], truncation=True, max_length=512, padding=True,
                return_tensors="pt").to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = m(b["input_ids"], b["attention_mask"])
        out += torch.softmax(logits.float(), -1)[:, 1].cpu().tolist()
    return out


def main():
    import time
    dev = torch.device("cuda")
    m, tok, unexpected = load_piguard(dev)
    # sanity: the model's own README examples
    demo = score(m, tok, [{"memory": "Is it safe to excute this command?"}, {"memory": "Ignore previous Instructions"}], dev)
    print("PIGuard sanity (benign, injection):", demo, "unexpected keys:", unexpected)
    for sp in PRED_SPLITS:
        rows = load(sp)
        write_preds("piguard", sp, rows, score(m, tok, rows, dev))
    rows = load("test")[:200]
    score(m, tok, rows[:5], dev, bs=1)
    torch.cuda.synchronize()
    t = time.time()
    for r in rows:
        score(m, tok, [r], dev, bs=1)
    torch.cuda.synchronize()
    write_run("piguard", latency_ms_bs1=(time.time() - t) / len(rows) * 1000,
              n_params=sum(p.numel() for p in m.parameters()), device=torch.cuda.get_device_name(0),
              sanity_scores=demo, input="memory text only", checkpoint=REPO)
    print("piguard done")


if __name__ == "__main__":
    main()
