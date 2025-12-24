from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager
import joblib
from typing import List
import os
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# NLTK resources (Backend startup par zaroori hain)
nltk.download('stopwords')
nltk.download('wordnet')

# ==========================================
# GLOBAL VARIABLES
# ==========================================
model = None
vectorizer = None
MODEL_PATH = 'sentiment_model.pkl'
VECTORIZER_PATH = 'vectorizer.pkl'

# ==========================================
# PREPROCESSING FUNCTION (Sync with Training)
# ==========================================
def preprocess_text(text):
    """Raw review ko clean karna taake model sahi samajh sakay"""
    text = str(text).lower()
    text = text.replace('1st', 'first').replace('grnd', 'ground')
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    stop_words = set(stopwords.words('english'))
    # Sentiment ke liye zaroori alfaz ko nahi nikalna
    negation_words = {'not', 'no', 'never', 'but', 'however', 'neither', 'nor', 'against'}
    final_stop_words = stop_words - negation_words
    
    lemmatizer = WordNetLemmatizer()
    words = text.split()
    cleaned_words = [lemmatizer.lemmatize(w) for w in words if w not in final_stop_words]
    return " ".join(cleaned_words)

# ==========================================
# MODEL LOADING FUNCTION
# ==========================================
def load_model():
    global model, vectorizer
    if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
        return False
    try:
        model = joblib.load(MODEL_PATH)
        vectorizer = joblib.load(VECTORIZER_PATH)
        print("✅ Model aur Vectorizer loaded!")
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

# ==========================================
# LIFESPAN HANDLER
# ==========================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("\n🚀 Starting Sentiment Analysis API...")
    if load_model():
        print("✅ API is ready!")
    else:
        print("⚠️ Model files missing or corrupted.")
    yield
    print("\n👋 Shutting down API...")

# ==========================================
# FASTAPI APP SETUP
# ==========================================
app = FastAPI(
    title="Sentiment Analysis API",
    description="Healthcare Review Sentiment Analysis",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# PYDANTIC MODELS (Same as yours)
# ==========================================
class ReviewRequest(BaseModel):
    review: str

class SentimentResponse(BaseModel):
    review: str
    sentiment: str
    confidence: str = "high"

class BatchReviewRequest(BaseModel):
    reviews: List[str]

class BatchSentimentResponse(BaseModel):
    results: List[SentimentResponse]
    total_reviews: int

# ==========================================
# API ENDPOINTS
# ==========================================
@app.get("/")
async def root():
    return {"status": "active", "model_loaded": model is not None}

@app.post("/predict", response_model=SentimentResponse)
async def predict_sentiment(request: ReviewRequest):
    if model is None or vectorizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if not request.review.strip():
        raise HTTPException(status_code=400, detail="Review is empty")

    try:
        # STEP 1: Preprocess (Important!)
        cleaned_review = preprocess_text(request.review)
        
        # STEP 2: Vectorize
        review_vec = vectorizer.transform([cleaned_review])
        
        # STEP 3: Predict
        sentiment = model.predict(review_vec)[0]
        
        return SentimentResponse(review=request.review, sentiment=sentiment)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch", response_model=BatchSentimentResponse)
async def predict_batch_sentiment(request: BatchReviewRequest):
    if model is None or vectorizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    valid_reviews = [r for r in request.reviews if r and r.strip()]
    
    try:
        # Batch preprocessing
        cleaned_list = [preprocess_text(r) for r in valid_reviews]
        vecs = vectorizer.transform(cleaned_list)
        sentiments = model.predict(vecs)
        
        results = [
            SentimentResponse(review=rev, sentiment=sent)
            for rev, sent in zip(valid_reviews, sentiments)
        ]
        return BatchSentimentResponse(results=results, total_reviews=len(results))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)