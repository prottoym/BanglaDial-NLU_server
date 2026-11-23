import logging
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .config import CORS_ORIGINS, LANGUAGE_MODELS, MAX_CHARS
from .language import detect_language
from .registry import Registry

logging.basicConfig(level=logging.INFO)

registry = Registry()

# Slot colours the interface knows. A slot name always gets the same colour.
PALETTE = ["emerald", "amber", "blue", "violet", "cyan", "orange", "rose"]
DOMAIN_ICONS = {
    "healthcare": "heart", "health": "heart", "ecommerce": "cart", "e_commerce": "cart", "e-commerce": "cart",
    "education": "graduation", "finance": "bank", "banking": "bank", "gov_service": "landmark", "government": "landmark",
}


def color_for(name: str) -> str:
    return PALETTE[sum(ord(c) for c in name) % len(PALETTE)]


def icon_for(domain: str) -> str:
    key = domain.lower().replace(" ", "_")
    for k, v in DOMAIN_ICONS.items():
        if k in key:
            return v
    return "message"


def humanize(name: str) -> str:
    return name.replace("_", " ").replace("-", " ").strip().lower()


@asynccontextmanager
async def lifespan(_: FastAPI):
    registry.load_all()
    yield


app = FastAPI(title="Multilingual NLU API", version="1.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=CORS_ORIGINS, allow_methods=["GET", "POST"], allow_headers=["*"])


class AnalyzeRequest(BaseModel):
    text: str = Field(min_length=1, max_length=MAX_CHARS)
    language: Literal["auto", "en", "bn", "bl"] = "auto"
    history: list[str] = Field(default_factory=list, max_length=10)  # optional earlier turns


@app.get("/health")
def health():
    return {
        "status": "ok" if registry.models else "no_models",
        "models": {
            code: {
                "loaded": code in registry.models,
                "backbone": LANGUAGE_MODELS[code][1],
                "checkpoint": registry.models[code].checkpoint_name if code in registry.models else None,
                "intents": len(registry.models[code].intents) if code in registry.models else None,
                "slot_labels": len(registry.models[code].slot_labels) if code in registry.models else None,
                "device": str(registry.models[code].device) if code in registry.models else None,
                "error": registry.errors.get(code),
            }
            for code in LANGUAGE_MODELS
        },
    }


@app.post("/analyze")
def analyze(req: AnalyzeRequest):
    text = req.text.strip()
    if not text:
        raise HTTPException(422, "Text is empty.")

    detected = detect_language(text) if req.language == "auto" else req.language
    try:
        used, model, fell_back = registry.pick(detected)
    except LookupError as exc:
        raise HTTPException(503, str(exc))

    try:
        pred = model.predict(text, req.history)
    except ValueError as exc:
        raise HTTPException(422, str(exc))

    return {
        "originalText": text,
        "detectedLanguage": detected,
        "intent": {
            "name": pred.intent,
            "description": f"Domain: {humanize(pred.domain)}. Predicted by {LANGUAGE_MODELS[used][1]}.",
            "confidence": round(pred.confidence, 4),
            "icon": icon_for(pred.domain),
        },
        "slots": [
            {"value": s["value"], "type": s["slot"], "color": color_for(s["slot"]), "start": s["start"], "end": s["end"]}
            for s in pred.slots
        ],
        "isMock": False,
        # extra fields (the interface ignores what it does not know)
        "domain": pred.domain,
        "topIntents": [{"name": n, "confidence": round(p, 4)} for n, p in pred.top_intents],
        "model": {"language": used, "backbone": model.backbone, "fallback": fell_back},
    }
