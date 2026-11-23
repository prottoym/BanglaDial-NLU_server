"""Language detection for the three supported kinds of input.

  bn  Bengali script
  bl  Banglish = Bangla written in Latin letters (detected with a word list)
  en  everything else

Banglish detection is a simple heuristic, not a model. The API also accepts
a `language` field so the interface can say it explicitly.
"""
import re

_BENGALI = re.compile(r"[ঀ-৿]")
_WORD = re.compile(r"[a-z]+")

# Very common romanised Bangla words that are rare in English.
_BANGLISH_WORDS = {
    "ami", "amar", "amake", "tumi", "tomar", "tomake", "apni", "apnar", "apnake", "amra", "amader", "tara", "tader",
    "ekta", "ekhon", "ekhoni", "ekhane", "okhane", "kothay", "kothai", "keno", "kivabe", "kibhabe", "kemon", "koto",
    "kobe", "kon", "ki", "kono", "kichu", "kisu", "ache", "achhe", "nai", "nei", "hobe", "hoy", "hoye", "hoyeche",
    "chai", "chaichi", "chaw", "chao", "dite", "dibo", "dao", "den", "nite", "nibo", "korte", "korbo", "kori", "korun",
    "koro", "korchi", "jabo", "jete", "jacchi", "jai", "ashbo", "asbo", "pathao", "pathaite", "paban", "pabo", "pari",
    "parbo", "parben", "lagbe", "lagche", "bolo", "bolun", "bolte", "dekhte", "dekhao", "dekhbo", "jante", "janan",
    "taka", "tk", "bhalo", "valo", "kharap", "dorkar", "shuru", "shesh", "ojon", "jonno", "theke", "porjonto", "moddhe",
    "ebong", "ar", "na", "naki", "ba", "kintu", "tai", "tahole", "ekhon", "aj", "aaj", "kal", "porshu", "ei", "oi",
    "eta", "ota", "ekti", "duto", "tin", "char", "pach", "choy", "shat", "at", "noy", "dosh", "bank", "hospital",
    "doctor", "appointment", "nibo", "dite", "chacchi", "chachchi", "bujhte", "bujhi", "jani", "janina", "sathe",
    "babohar", "babohar", "khoj", "khuji", "khujchi", "bondho", "chalu", "abedon", "application", "joma", "uthano",
}


def detect_language(text: str) -> str:
    letters = [c for c in text if c.isalpha()]
    if letters:
        bengali = sum(1 for c in letters if _BENGALI.match(c))
        if bengali / len(letters) >= 0.3:
            return "bn"
    words = _WORD.findall(text.lower())
    if words:
        hits = sum(1 for w in words if w in _BANGLISH_WORDS)
        if hits >= 2 or hits / len(words) >= 0.3:
            return "bl"
    return "en"
