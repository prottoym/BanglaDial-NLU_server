"""
The joint intent + slot model, rebuilt exactly as in the training notebooks.

Notebook facts this file reproduces:
  * JointBertNLU = BertModel + intent head (768 -> 256 -> GELU -> n_intents)
    + slot head (768 -> n_slots), intent from pooler_output, slots per token.
  * Input = [CLS] ctx1 [SEP] ctx2 [SEP] ... CURRENT TURN [SEP], max_len 256,
    the sentence is split on spaces and tokenized with is_split_into_words.
  * Slot tag is read from the FIRST sub-token of every word (BIO tags).
  * The checkpoint stores the label lists, so no extra label file is needed.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from transformers import BertConfig, BertModel, BertTokenizerFast


class JointBertNLU(nn.Module):
    def __init__(self, bert: BertModel, n_intents: int, n_slots: int, dropout: float):
        super().__init__()
        self.bert = bert
        h = bert.config.hidden_size
        self.intent_head = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(h, 256), nn.GELU(), nn.Dropout(dropout), nn.Linear(256, n_intents)
        )
        self.slot_head = nn.Sequential(nn.Dropout(dropout), nn.Linear(h, n_slots))

    def forward(self, input_ids, attention_mask, token_type_ids):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        return self.intent_head(out.pooler_output), self.slot_head(out.last_hidden_state)


def extract_entities(tags: list[str]) -> list[tuple[str, int, int]]:
    """BIO tags -> (slot name, first word, last word + 1). Same logic as the notebooks."""
    ents, cur = [], None
    for i, t in enumerate(tags):
        if t.startswith("B-"):
            if cur:
                ents.append(tuple(cur))
            cur = [t[2:], i, i + 1]
        elif t.startswith("I-"):
            if cur and cur[0] == t[2:]:
                cur[2] = i + 1
            else:
                if cur:
                    ents.append(tuple(cur))
                cur = [t[2:], i, i + 1]
        else:
            if cur:
                ents.append(tuple(cur))
                cur = None
    if cur:
        ents.append(tuple(cur))
    return ents


@dataclass
class Prediction:
    intent: str
    confidence: float
    domain: str
    top_intents: list[tuple[str, float]]
    slots: list[dict]  # {"slot", "value", "start", "end"}  (character positions in the text)


class NLUModel:
    """One loaded model: tokenizer + network + label maps."""

    def __init__(self, folder: Path, checkpoint: Path, device: str | None = None):
        self.folder = folder
        self.checkpoint_name = checkpoint.name
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
        cfg = ckpt["config"]
        self.cfg = cfg
        self.intents: list[str] = list(ckpt["intents"])
        self.slot_labels: list[str] = list(ckpt["slot_labels"])
        self.intent2domain: dict[str, str] = dict(ckpt.get("intent2domain", {}))
        self.max_len = int(cfg.get("max_len", 256))
        self.context_turns = int(cfg.get("context_turns", 3))
        backbone = cfg["model_name"]

        # Use files saved next to the checkpoint when present (works offline),
        # otherwise download the backbone's tokenizer/config from Hugging Face.
        local = (folder / "tokenizer_config.json").exists() or (folder / "vocab.txt").exists()
        source = str(folder) if local else backbone
        tok_kwargs = {}
        if "do_lower_case" in cfg:  # only the English notebook sets these
            tok_kwargs["do_lower_case"] = cfg["do_lower_case"]
        if "strip_accents" in cfg:
            tok_kwargs["strip_accents"] = cfg["strip_accents"]
        self.tokenizer = BertTokenizerFast.from_pretrained(source, **tok_kwargs)
        self.tokenizer.model_max_length = 10**6
        bert_cfg = BertConfig.from_pretrained(str(folder) if (folder / "config.json").exists() else backbone)

        # BertModel(cfg) builds the empty network; the real weights come from the checkpoint.
        net = JointBertNLU(BertModel(bert_cfg), len(self.intents), len(self.slot_labels), float(cfg.get("dropout", 0.1)))
        net.load_state_dict(ckpt["model_state"])
        self.net = net.to(self.device).eval()
        self.backbone = backbone

    # ---- encoding: same layout as the notebook's encode_example ----
    def _encode(self, ctx_ids: list[list[int]], cur_ids: list[int], cur_wids: list[int | None]):
        cls_id, sep_id = self.tokenizer.cls_token_id, self.tokenizer.sep_token_id
        ctx = [list(c) for c in ctx_ids if c]
        max_len = self.max_len
        while True:  # oldest context is trimmed first
            excess = 2 + len(ctx) + sum(len(c) for c in ctx) + len(cur_ids) - max_len
            if excess <= 0 or not ctx:
                break
            if len(ctx[0]) <= excess:
                ctx.pop(0)
            else:
                ctx[0] = ctx[0][excess:]
        if 2 + len(cur_ids) > max_len:
            keep = max_len - 2
            cur_ids, cur_wids = cur_ids[:keep], cur_wids[:keep]
        ids, tt = [cls_id], [0]
        for c in ctx:
            ids += c + [sep_id]
            tt += [0] * (len(c) + 1)
        start = len(ids)
        ids += cur_ids
        tt += [1] * len(cur_ids)
        prev, label_words, first_pos = None, [], []
        for off, wid in enumerate(cur_wids):
            if wid is not None and wid != prev:
                label_words.append(wid)
                first_pos.append(start + off)
            prev = wid
        ids.append(sep_id)
        tt.append(1)
        return ids, tt, label_words, first_pos

    def _words(self, words: list[str]):
        enc = self.tokenizer([words], is_split_into_words=True, add_special_tokens=False, truncation=False)
        return enc["input_ids"][0], enc.word_ids(0)

    @torch.no_grad()
    def predict(self, text: str, history: list[str] | None = None) -> Prediction:
        spans = [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]
        words = [text[a:b] for a, b in spans]
        if not words:
            raise ValueError("empty utterance")

        hist = [h for h in (history or []) if h and h.split()]
        hist = hist[len(hist) - self.context_turns:] if self.context_turns > 0 else []
        ctx_ids = [self._words(h.split())[0] for h in hist]
        cur_ids, cur_wids = self._words(words)
        ids, tt, label_words, first_pos = self._encode(ctx_ids, cur_ids, cur_wids)

        t = lambda x: torch.tensor([x], dtype=torch.long, device=self.device)
        mask = torch.ones(1, len(ids), dtype=torch.long, device=self.device)
        il, sl = self.net(t(ids), mask, t(tt))

        probs = torch.softmax(il.float()[0], -1)
        k = min(3, probs.numel())
        top_p, top_i = probs.topk(k)
        top = [(self.intents[int(i)], float(p)) for p, i in zip(top_p, top_i)]
        intent = top[0][0]

        tags = [self.slot_labels[int(sl[0, p].argmax())] for p in first_pos]
        slots = []
        for name, a, b in extract_entities(tags):
            w0, w1 = label_words[a], label_words[b - 1]
            start, end = spans[w0][0], spans[w1][1]
            slots.append({"slot": name, "value": text[start:end], "start": start, "end": end})

        return Prediction(intent, top[0][1], self.intent2domain.get(intent, "UNKNOWN"), top, slots)


def find_checkpoint(folder: Path) -> Path | None:
    files = sorted(folder.glob("*.pt")) + sorted(folder.glob("*.pth"))
    if not files:
        return None
    best = [f for f in files if f.name.startswith("best")]
    return (best or files)[0]
