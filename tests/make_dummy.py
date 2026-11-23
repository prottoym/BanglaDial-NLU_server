"""Builds three tiny fake models (random weights) to test the API plumbing.
Usage: python tests/make_dummy.py <output models dir>
The answers are meaningless; this only proves loading, tokenizing and JSON output work."""
import sys
from pathlib import Path

import torch
from transformers import BertConfig, BertModel, BertTokenizerFast

from app.model import JointBertNLU

out = Path(sys.argv[1])
vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "book", "a", "flight", "to", "dhaka", "ami", "taka", "##s",
         "আমি", "ঢাকা", "যেতে", "চাই"]
INTENTS = ["book_flight", "check_balance", "order_food"]
SLOTS = ["O", "B-destination", "I-destination", "B-amount", "I-amount"]
for folder, extra in (("english", {"do_lower_case": True, "strip_accents": False}), ("bangla", {}), ("banglish", {})):
    d = out / folder
    d.mkdir(parents=True, exist_ok=True)
    (d / "vocab.txt").write_text("\n".join(vocab), encoding="utf-8")
    tok = BertTokenizerFast(vocab_file=str(d / "vocab.txt"), **extra)
    tok.save_pretrained(d)
    cfg = BertConfig(vocab_size=len(vocab), hidden_size=32, num_hidden_layers=2, num_attention_heads=2, intermediate_size=64)
    cfg.save_pretrained(d)
    net = JointBertNLU(BertModel(cfg), len(INTENTS), len(SLOTS), 0.1)
    torch.save({"model_state": net.state_dict(), "config": {"model_name": "dummy", "max_len": 256, "context_turns": 3,
                "dropout": 0.1, **extra}, "intents": INTENTS, "slot_labels": SLOTS,
                "intent2domain": {"book_flight": "travel", "check_balance": "finance", "order_food": "food"}},
               d / "best_dummy.pt")
print("dummy models written to", out)
