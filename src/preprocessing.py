"""
Arabic text preprocessing used throughout the traditional (TF-IDF) pipeline.
المعالجة المسبقة للنص العربي: نفس القواعد تُطبَّق على كل التجارب.

Rules (fixed for all experiments):
  1. Unicode NFKC normalization
  2. Remove diacritics (tashkeel) and Tatweel (ـ)
  3. Normalize Alef forms (أ إ آ ٱ -> ا) and Alef Maqsura (ى -> ي);
     keep ة, ؤ, ئ unchanged
  4. Remove URLs, e-mails, numbers, punctuation and non-Arabic characters
  5. Remove NLTK Arabic stop words (normalized with the same rules)
  No stemming or lemmatization.
"""
import re
import unicodedata

DIACRITICS = re.compile(r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]")
URLS_EMAILS = re.compile(r"https?://\S+|www\.\S+|\S+@\S+")
DIGITS = re.compile(r"[0-9\u0660-\u0669\u06F0-\u06F9]+")
NON_ARABIC = re.compile(r"[^\u0621-\u064A0-9\u0660-\u0669\u06F0-\u06F9\s]")

# Token pattern used by every TfidfVectorizer in this project
TOKEN_PATTERN = r"(?u)\b\w+\b"


def normalize(text: str) -> str:
    """Character-level normalization (rules 1-3)."""
    text = unicodedata.normalize("NFKC", str(text))
    text = DIACRITICS.sub("", text)
    text = text.replace("\u0640", "")      # Tatweel / التطويل
    text = re.sub("[أإآٱ]", "ا", text)     # Alef forms / أشكال الألف
    text = text.replace("ى", "ي")          # Alef Maqsura / الألف المقصورة
    return text


def load_stopwords() -> set:
    """NLTK Arabic stop words, normalized with the same rules as the text."""
    import nltk
    try:
        from nltk.corpus import stopwords
        words = stopwords.words("arabic")
    except LookupError:
        nltk.download("stopwords", quiet=True)
        from nltk.corpus import stopwords
        words = stopwords.words("arabic")
    return {normalize(w) for w in words}


def preprocess(text: str, stopwords: set,
               remove_stopwords: bool = True, remove_numbers: bool = True) -> str:
    """Full preprocessing (rules 1-5).
    The two flags exist only for the ablation study; the primary
    configuration keeps both True."""
    text = normalize(text)
    text = URLS_EMAILS.sub(" ", text)
    if remove_numbers:
        text = DIGITS.sub(" ", text)
    text = NON_ARABIC.sub(" ", text)
    tokens = text.split()
    if remove_stopwords:
        tokens = [t for t in tokens if t not in stopwords]
    return " ".join(tokens)
