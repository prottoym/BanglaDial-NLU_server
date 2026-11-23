"""Settings and the fixed facts that come from the training notebooks."""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

MODELS_DIR = Path(os.getenv("MODELS_DIR", "./models")).resolve()
CORS_ORIGINS = [o.strip().rstrip("/") for o in os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",") if o.strip()]
DEVICE = os.getenv("DEVICE", "").strip() or None

# frontend language code -> (sub-folder in MODELS_DIR, readable backbone name)
LANGUAGE_MODELS = {
    "bn": ("bangla", "Bangla BERT (sagorsarker/bangla-bert-base)"),
    "en": ("english", "bert-base-uncased"),
    "bl": ("banglish", "mBERT (bert-base-multilingual-cased)"),
}

MAX_CHARS = 1000  # longest sentence accepted by the API
