# Multilingual NLU Backend

FastAPI server that runs your three trained models and answers the frontend.

| Language | Model folder | Backbone (from your notebooks) |
| --- | --- | --- |
| Bangla (`bn`) | `models/bangla/` | `sagorsarker/bangla-bert-base` |
| English (`en`) | `models/english/` | `bert-base-uncased` |
| Banglish (`bl`) | `models/banglish/` | `google-bert/bert-base-multilingual-cased` |

All three notebooks use the same network (`JointBertNLU`: BERT + intent head + BIO slot head), so one loader serves all of them. Each checkpoint already contains its own intent list, slot list and settings, so no label file is needed.

## 1. Put the model files in place

After training, each notebook saved the best checkpoint as
`seed_XX/best_joint_bert_base_uncased.pt` (the name differs per notebook, it ends in `.pt`).
Pick the best seed for each model and copy it:

```
models/bangla/best_xxx.pt
models/english/best_joint_bert_base_uncased.pt
models/banglish/best_xxx.pt
```

A language without a file is simply not loaded (`GET /health` says which). If a language has no model, its sentences go to another loaded model and the answer says `"fallback": true`.

## 2. Install and run

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env              # Windows: copy .env.example .env
python scripts/prepare_tokenizers.py   # once, needs internet: saves tokenizers next to the .pt files
uvicorn app.main:app --port 8000
```

Open http://localhost:8000/docs to try it. A GPU is used automatically if PyTorch can see one.

## 3. Connect the frontend

In the frontend project's `.env.local`:

```
NEXT_PUBLIC_NLP_API_URL=http://localhost:8000
```

Restart `npm run dev`. The "Demo mode" notice disappears and results come from your models.

## API

`GET /health` shows which models are loaded.

`POST /analyze`

```json
{ "text": "Ami ekhoni application submit dite chai", "language": "auto", "history": [] }
```

`language` is `auto` (default), `en`, `bn` or `bl`. `history` is optional earlier turns (the models were trained with up to 3 previous turns of context). The answer follows the frontend's `AnalysisResult` shape: `originalText`, `detectedLanguage`, `intent {name, description, confidence, icon}`, `slots [{value, type, color, start, end}]`, plus `domain`, `topIntents` and `model`.

## Good to know

- Language detection is a simple rule (Bengali letters = Bangla, a list of common romanised words = Banglish, else English). It can be wrong on short Banglish; send `language` to override.
- Checkpoints are loaded with `torch.load(..., weights_only=False)` because they store a config dict. Only load files you trained yourself.
- Smoke test without real models: `python tests/make_dummy.py /tmp/dummy_models`, then `MODELS_DIR=/tmp/dummy_models uvicorn app.main:app`. The answers are random, it only checks the plumbing.
- Deploy: any host that can run Python and PyTorch (Render, Railway, a VPS). Set `CORS_ORIGINS` to your frontend URL. Vercel serverless is too small for these models.
