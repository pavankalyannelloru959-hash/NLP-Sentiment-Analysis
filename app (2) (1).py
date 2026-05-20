"""
app.py — SentimentIQ Backend
Connects: best_model.pkl · tfidf_vectorizer.pkl · model_metadata.pkl · dataset.xlsx
Serves:   sentiment_dashboard.html

Run:
    python app.py
Then open: http://localhost:5000
"""

import os
import re
import pickle
import warnings
import numpy as np
import pandas as pd

from flask import Flask, request, jsonify, send_file, render_template_string
from flask_cors import CORS

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# 1.  Configuration
# ─────────────────────────────────────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH    = os.path.join(BASE_DIR, "best_model.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "tfidf_vectorizer.pkl")
METADATA_PATH = os.path.join(BASE_DIR, "model_metadata.pkl")
DATASET_PATH  = os.path.join(BASE_DIR, "dataset.xlsx")
HTML_PATH     = os.path.join(BASE_DIR, "sentiment_dashboard.html")

# ─────────────────────────────────────────────────────────────────────────────
# 2.  Load Models at Startup
# ─────────────────────────────────────────────────────────────────────────────

print("⏳ Loading model artifacts …")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

with open(VECTORIZER_PATH, "rb") as f:
    vectorizer = pickle.load(f)

with open(METADATA_PATH, "rb") as f:
    metadata = pickle.load(f)

CLASSES = metadata.get("classes", ["Negative", "Neutral", "Positive"])
print(f"✅ Model loaded  : {metadata.get('best_model_name', 'LinearSVC')}")
print(f"   Accuracy      : {metadata.get('accuracy', 0):.4f}")
print(f"   F1-Score      : {metadata.get('f1_score', 0):.4f}")
print(f"   TF-IDF feats  : {metadata.get('tfidf_features', 'N/A')}")
print(f"   Classes       : {CLASSES}")

# ─────────────────────────────────────────────────────────────────────────────
# 3.  NLP Preprocessing  (mirrors the notebook's clean_text exactly)
# ─────────────────────────────────────────────────────────────────────────────

STOPWORDS = {
    'i','me','my','myself','we','our','ours','ourselves','you','your','yours',
    'yourself','yourselves','he','him','his','himself','she','her','hers',
    'herself','it','its','itself','they','them','their','theirs','themselves',
    'what','which','who','whom','this','that','these','those','am','is','are',
    'was','were','be','been','being','have','has','had','having','do','does',
    'did','doing','a','an','the','and','but','if','or','because','as','until',
    'while','of','at','by','for','with','about','against','between','into',
    'through','during','before','after','above','below','to','from','up','down',
    'in','out','on','off','over','under','again','further','then','once','here',
    'there','when','where','why','how','all','both','each','few','more','most',
    'other','some','such','no','nor','not','only','own','same','so','than',
    'too','very','s','t','can','will','just','don','should','now','d','ll',
    'm','o','re','ve','y','ain','aren','couldn','didn','doesn','hadn','hasn',
    'haven','isn','ma','mightn','mustn','needn','shan','shouldn','wasn','weren',
    'won','wouldn','also','get','got','would','could','like','even','still','us',
    'want','one','two','three','four','five','going','go','come','came','take',
    'much','many','lot','bit','really','things','thing','say','said','make',
    'made','good','bad','great','nice','well','always','never','every','time',
    'day','days','week','month','year','buy','bought','phone','mobile','product',
    'samsung','amazon','review','star','rating',
}

LEMMA_MAP = {
    'running':'run','runs':'run','ran':'run',
    'buying':'buy','buys':'buy','bought':'buy',
    'using':'use','uses':'use','used':'use',
    'giving':'give','gives':'give','gave':'give',
    'getting':'get','gets':'get','gotten':'get',
    'working':'work','works':'work','worked':'work',
    'loving':'love','loves':'love','loved':'love',
    'hating':'hate','hates':'hate','hated':'hate',
    'heating':'heat','heats':'heat','heated':'heat',
    'lagging':'lag','lags':'lag','lagged':'lag',
    'hanging':'hang','hangs':'hang','hanged':'hang',
    'charging':'charge','charges':'charge','charged':'charge',
    'disappointed':'disappoint','disappointing':'disappoint',
    'cameras':'camera','phones':'phone','mobiles':'mobile',
    'pictures':'picture','photos':'photo','images':'image',
    'batteries':'battery','displays':'display','processors':'processor',
    'reviews':'review','customers':'customer','products':'product',
    'issues':'issue','problems':'problem','features':'feature',
    'worse':'bad','worst':'bad','better':'good','best':'good',
    'terrible':'bad','horrible':'bad','awful':'bad',
    'excellent':'excellent','outstanding':'excellent','amazing':'amazing',
    'awesome':'awesome','fantastic':'fantastic','wonderful':'wonderful',
}

POS_WORDS = {
    'good','great','excellent','amazing','awesome','fantastic','wonderful',
    'outstanding','brilliant','perfect','best','love','happy','satisfied',
    'impressive','quality','fast','smooth','long','clear','bright',
    'comfortable','easy','nice','enjoy','recommend','worth','value',
    'genuine','reliable','solid','strong','powerful','beautiful','clean',
    'crisp','quick','efficient','pleasant','superb','exceptional',
    'delighted','thrilled','pleased','incredible',
}

NEG_WORDS = {
    'bad','poor','worst','terrible','horrible','awful','disappoint',
    'slow','hang','lag','heavy','blurry','cheap','waste','useless',
    'pathetic','garbage','rubbish','junk','defective','broken','issue',
    'problem','fake','false','mislead','fraud','cheat','regret',
    'returning','disappointed','frustrated','angry','annoyed','inferior',
    'mediocre','substandard','faulty','glitch','crash','error','fail',
    'unreliable','expensive','overpriced',
}


def clean_text(text: str) -> str:
    """Mirror of the notebook's clean_text function."""
    if not isinstance(text, str) or not text.strip():
        return ""
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 2]
    tokens = [LEMMA_MAP.get(t, t) for t in tokens]
    return ' '.join(tokens)


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Prediction Helper
# ─────────────────────────────────────────────────────────────────────────────

def predict_review(text: str) -> dict:
    """
    Run the full NLP pipeline and return a prediction dict with:
      label, confidence, class_scores, tokens, pos_signals, neg_signals
    """
    cleaned = clean_text(text)

    if not cleaned.strip():
        return {
            "label": "Neutral",
            "confidence": 0.50,
            "class_scores": {"Positive": 0.33, "Neutral": 0.34, "Negative": 0.33},
            "tokens": [],
            "pos_signals": [],
            "neg_signals": [],
            "cleaned_text": cleaned,
        }

    # Vectorise
    X = vectorizer.transform([cleaned])

    # Predict label
    label = model.predict(X)[0]

    # Confidence via decision_function scores → softmax-like normalisation
    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)[0]          # shape (n_classes,)
        # Shift to positive then normalise
        scores = scores - scores.min()
        total  = scores.sum() if scores.sum() != 0 else 1.0
        proba  = scores / total
        class_scores = {cls: float(round(proba[i], 4)) for i, cls in enumerate(CLASSES)}
        confidence   = float(round(proba[list(CLASSES).index(label)], 4))
        confidence   = max(confidence, 0.50)            # floor at 50 %
    else:
        # Fallback for models that have predict_proba
        proba_arr = model.predict_proba(X)[0]
        class_scores = {cls: float(round(proba_arr[i], 4)) for i, cls in enumerate(CLASSES)}
        confidence   = float(round(max(proba_arr), 4))

    # Keyword signals from the raw text tokens
    raw_tokens  = re.sub(r'[^a-z\s]', ' ', text.lower()).split()
    pos_signals = list({LEMMA_MAP.get(w, w) for w in raw_tokens if LEMMA_MAP.get(w, w) in POS_WORDS})
    neg_signals = list({LEMMA_MAP.get(w, w) for w in raw_tokens if LEMMA_MAP.get(w, w) in NEG_WORDS})

    return {
        "label":        label,
        "confidence":   confidence,
        "class_scores": class_scores,
        "tokens":       cleaned.split()[:30],
        "pos_signals":  pos_signals[:8],
        "neg_signals":  neg_signals[:8],
        "cleaned_text": cleaned,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Dataset stats (cached at startup)
# ─────────────────────────────────────────────────────────────────────────────

DATASET_STATS = {}

try:
    df = pd.read_excel(DATASET_PATH)

    def _rating_to_sentiment(r):
        if r >= 4:
            return "Positive"
        elif r == 3:
            return "Neutral"
        else:
            return "Negative"

    df["sentiment"] = df["rating"].apply(_rating_to_sentiment)

    sent_counts = df["sentiment"].value_counts().to_dict()
    rating_counts = df["rating"].value_counts().sort_index().to_dict()

    DATASET_STATS = {
        "total_reviews":   int(len(df)),
        "sentiment_counts": {str(k): int(v) for k, v in sent_counts.items()},
        "rating_counts":    {str(k): int(v) for k, v in rating_counts.items()},
        "avg_rating":       float(round(df["rating"].mean(), 2)),
    }
    print(f"✅ Dataset loaded  : {len(df):,} reviews")
except Exception as e:
    print(f"⚠️  Dataset load warning: {e}")
    DATASET_STATS = {
        "total_reviews": 1440,
        "sentiment_counts": {"Positive": 729, "Neutral": 199, "Negative": 512},
        "rating_counts": {"1": 386, "2": 126, "3": 199, "4": 310, "5": 419},
        "avg_rating": 3.17,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Flask App & Routes
# ─────────────────────────────────────────────────────────────────────────────

app = Flask(__name__)
CORS(app)   # Allow requests from the HTML file opened locally


# ── Serve the dashboard ──────────────────────────────────────────────────────
@app.route("/")
def index():
    """Serve the HTML dashboard."""
    with open(HTML_PATH, "r", encoding="utf-8") as f:
        html = f.read()

    # Inject the API base URL into the page so JS knows where to call
    inject = """
<script>
  /* ── Injected by app.py ─────────────────────────── */
  window.API_BASE = "";          // same origin
  window.MODEL_CONNECTED = true; // flag for the JS layer
  /* ─────────────────────────────────────────────── */
</script>
"""
    html = html.replace("</head>", inject + "\n</head>", 1)
    return html


# ── Single prediction ────────────────────────────────────────────────────────
@app.route("/api/predict", methods=["POST"])
def api_predict():
    """
    POST /api/predict
    Body (JSON): { "text": "review text here" }
    Returns   : prediction dict
    """
    data = request.get_json(force=True, silent=True) or {}
    text = data.get("text", "").strip()

    if not text:
        return jsonify({"error": "No review text provided."}), 400

    result = predict_review(text)
    return jsonify(result)


# ── Batch prediction (JSON array) ────────────────────────────────────────────
@app.route("/api/predict/batch", methods=["POST"])
def api_predict_batch():
    """
    POST /api/predict/batch
    Body (JSON): { "reviews": ["text1", "text2", …] }
    Returns   : [ { label, confidence, … }, … ]
    """
    data    = request.get_json(force=True, silent=True) or {}
    reviews = data.get("reviews", [])

    if not reviews or not isinstance(reviews, list):
        return jsonify({"error": "Provide a JSON array under 'reviews'."}), 400

    results = [predict_review(str(r)) for r in reviews[:200]]   # cap at 200
    return jsonify(results)


# ── Batch prediction (file upload) ───────────────────────────────────────────
@app.route("/api/predict/upload", methods=["POST"])
def api_predict_upload():
    """
    POST /api/predict/upload
    Multipart form-data: file = .csv or .xlsx
    The file must have a column named 'review' (case-insensitive).
    Returns: JSON list of predictions.
    """
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded."}), 400

    f    = request.files["file"]
    name = f.filename.lower()

    try:
        if name.endswith(".csv"):
            df_upload = pd.read_csv(f)
        elif name.endswith((".xlsx", ".xls")):
            df_upload = pd.read_excel(f)
        else:
            return jsonify({"error": "Only .csv and .xlsx files are supported."}), 400
    except Exception as e:
        return jsonify({"error": f"Could not read file: {e}"}), 400

    # Locate the review column (case-insensitive)
    col_map = {c.lower(): c for c in df_upload.columns}
    review_col = col_map.get("review") or col_map.get("text") or col_map.get("body")

    if not review_col:
        review_col = df_upload.columns[0]   # fall back to first column

    rows = df_upload[review_col].fillna("").astype(str).tolist()[:200]

    results = []
    for i, text in enumerate(rows):
        r = predict_review(text)
        results.append({
            "index":      i + 1,
            "review":     text[:120] + ("…" if len(text) > 120 else ""),
            "label":      r["label"],
            "confidence": round(r["confidence"] * 100, 1),
        })

    return jsonify(results)


# ── Model metadata ───────────────────────────────────────────────────────────
@app.route("/api/model/info", methods=["GET"])
def api_model_info():
    """Return model metadata and dataset stats."""
    return jsonify({
        "model_name":     metadata.get("best_model_name", "LinearSVC"),
        "accuracy":       metadata.get("accuracy", 0),
        "f1_score":       metadata.get("f1_score", 0),
        "tfidf_features": metadata.get("tfidf_features", 0),
        "classes":        CLASSES,
        "dataset":        DATASET_STATS,
    })


# ── Dataset stats ────────────────────────────────────────────────────────────
@app.route("/api/dataset/stats", methods=["GET"])
def api_dataset_stats():
    """Return EDA stats for the dashboard charts."""
    return jsonify(DATASET_STATS)


# ── Health check ─────────────────────────────────────────────────────────────
@app.route("/api/health", methods=["GET"])
def api_health():
    return jsonify({"status": "ok", "model": metadata.get("best_model_name")})


# ─────────────────────────────────────────────────────────────────────────────
# 7.  Run
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "═" * 55)
    print("  SentimentIQ — Flask Backend")
    print("═" * 55)
    print("  Dashboard  →  http://localhost:5000")
    print("  Predict    →  POST /api/predict")
    print("  Batch JSON →  POST /api/predict/batch")
    print("  Batch File →  POST /api/predict/upload")
    print("  Model Info →  GET  /api/model/info")
    print("  Stats      →  GET  /api/dataset/stats")
    print("  Health     →  GET  /api/health")
    print("═" * 55 + "\n")

    app.run(host="0.0.0.0", port=5000, debug=True)
