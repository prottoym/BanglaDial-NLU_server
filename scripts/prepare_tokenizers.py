"""
One-time helper: saves each backbone's tokenizer + config next to the checkpoint,
so the server can start without internet. Run it once, with internet, after
copying the .pt files into models/<language>/.

    python scripts/prepare_tokenizers.py
"""
import sys
from pathlib import Path

import torch
from transformers import BertConfig, BertTokenizerFast

ROOT = Path(__file__).resolve().parent.parent
for folder in ("bangla", "english", "banglish"):
    d = ROOT / "models" / folder
    ckpts = sorted(d.glob("*.pt"))
    if not ckpts:
        print(f"[skip] {folder}: no .pt file")
        continue
    cfg = torch.load(ckpts[0], map_location="cpu", weights_only=False)["config"]
    name = cfg["model_name"]
    kw = {k: cfg[k] for k in ("do_lower_case", "strip_accents") if k in cfg}
    BertTokenizerFast.from_pretrained(name, **kw).save_pretrained(d)
    BertConfig.from_pretrained(name).save_pretrained(d)
    print(f"[ok] {folder}: saved tokenizer + config for {name}")
