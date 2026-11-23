"""Loads every model it can find and hands out the right one for a language."""
from __future__ import annotations

import logging

from .config import DEVICE, LANGUAGE_MODELS, MODELS_DIR
from .model import NLUModel, find_checkpoint

log = logging.getLogger("nlp")


class Registry:
    def __init__(self) -> None:
        self.models: dict[str, NLUModel] = {}
        self.errors: dict[str, str] = {}

    def load_all(self) -> None:
        for code, (folder_name, _label) in LANGUAGE_MODELS.items():
            folder = MODELS_DIR / folder_name
            ckpt = find_checkpoint(folder) if folder.is_dir() else None
            if ckpt is None:
                self.errors[code] = f"no .pt checkpoint in {folder}"
                log.warning("[%s] %s", code, self.errors[code])
                continue
            try:
                self.models[code] = NLUModel(folder, ckpt, DEVICE)
                log.info("[%s] loaded %s", code, ckpt.name)
            except Exception as exc:  # keep the other models usable
                self.errors[code] = f"{type(exc).__name__}: {exc}"
                log.exception("[%s] failed to load", code)

    def pick(self, language: str) -> tuple[str, NLUModel, bool]:
        """Returns (language code actually used, model, fell_back)."""
        if language in self.models:
            return language, self.models[language], False
        if not self.models:
            raise LookupError("No model is loaded. Put the .pt files in the models folder and restart.")
        code = next(iter(self.models))
        return code, self.models[code], True
