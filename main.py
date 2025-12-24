from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager
import joblib
from typing import List
import os

# ==========================================
# GLOBAL VARIABLES
# ==========================================
model = None
vectorizer = None
MODEL_PATH = 'sentiment_model.pkl'
VECTORIZER_PATH = 'vectorizer.pkl'

# ==========================================
# MODEL LOADING FUNCTION
# ==========================================
def load_model():
    """Saved model aur vectorizer ko load karo"""
    global model, vectorizer

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"❌ Model file nahi mili: {MODEL_PATH}")

    if not os.path.exists(VECTORIZER_PATH):
        raise FileNotFoundError(f"❌ Vectorizer file nahi mili: {VECTORIZER_PATH}")

    try:
        model = joblib.load(MODEL_PATH)
        vectorizer = joblib.load(VECTORIZER_PATH)
        print("✅ Model aur Vectorizer successfully load ho gaye!")
        return True
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False

# ==========================================
# LIFESPAN HANDLER (NEW METHOD)
# ==========================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup aur Shutdown events handle karo
    - yield se pehle: startup logic
    - yield ke baad: shutdown logic
    """
    # ===== STARTUP =====
    print("\n" + "="*60)
    print("🚀 Starting Sentiment Analysis API...")
    print("="*60)
    
    try:
        if load_model():
            print("✅ API is ready to accept requests!")
        else:
            print("⚠️  Warning: Model load nahi hua")
    except FileNotFoundError as e:
        print(f"⚠️  {e}")
        print("💡 Tip: Pehle model train karo aur save karo")
    except Exception as e:
        print(f"⚠️  Error during startup: {e}")
    
    print("="*60 + "\n")
    
    yield  # Application runs here
    
    # ===== SHUTDOWN =====
    print("\n" + "="*60)
    print("👋 Shutting down API...")
    print("="*60 + "\n")

# ==========================================
# FASTAPI APP SETUP
# ==========================================
app = FastAPI(
    title="Sentiment Analysis API",
    description="Healthcare Review Sentiment Analysis",
    version="1.0.0",
    lifespan=lifespan  # 👈 Lifespan handler yahan add karo
)

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# PYDANTIC MODELS
# ==========================================
class ReviewRequest(BaseModel):
    review: str

    class Config:
        json_schema_extra = {
            "example": {
                "review": "The doctor was very helpful and explained everything clearly"
            }
        }

class SentimentResponse(BaseModel):
    review: str
    sentiment: str
    confidence: str = "high"

class BatchReviewRequest(BaseModel):
    reviews: List[str]

    class Config:
        json_schema_extra = {
            "example": {
                "reviews": [
                    "Great service!",
                    "Very disappointed",
                    "It was okay"
                ]
            }
        }

class BatchSentimentResponse(BaseModel):
    results: List[SentimentResponse]
    total_reviews: int

# ==========================================
# API ENDPOINTS
# ==========================================
@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "Sentiment Analysis API is running! 🚀",
        "status": "active",
        "model_loaded": model is not None,
        "endpoints": {
            "predict": "/predict (POST)",
            "batch_predict": "/predict/batch (POST)",
            "health": "/health (GET)"
        }
    }

@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy" if model is not None else "unhealthy",
        "model_loaded": model is not None,
        "vectorizer_loaded": vectorizer is not None,
        "model_file_exists": os.path.exists(MODEL_PATH),
        "vectorizer_file_exists": os.path.exists(VECTORIZER_PATH)
    }

@app.post("/predict", response_model=SentimentResponse)
async def predict_sentiment(request: ReviewRequest):
    """Single review ka sentiment predict karo"""
    
    if model is None or vectorizer is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded. Please check server logs."
        )

    if not request.review or len(request.review.strip()) == 0:
        raise HTTPException(
            status_code=400,
            detail="Review cannot be empty"
        )

    try:
        review_vec = vectorizer.transform([request.review])
        sentiment = model.predict(review_vec)[0]
        
        return SentimentResponse(
            review=request.review,
            sentiment=sentiment
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )

@app.post("/predict/batch", response_model=BatchSentimentResponse)
async def predict_batch_sentiment(request: BatchReviewRequest):
    """Multiple reviews ka sentiment predict karo"""
    
    if model is None or vectorizer is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded. Please check server logs."
        )

    if not request.reviews or len(request.reviews) == 0:
        raise HTTPException(
            status_code=400,
            detail="Reviews list cannot be empty"
        )

    valid_reviews = [r for r in request.reviews if r and len(r.strip()) > 0]

    if len(valid_reviews) == 0:
        raise HTTPException(
            status_code=400,
            detail="All reviews are empty"
        )

    try:
        reviews_vec = vectorizer.transform(valid_reviews)
        sentiments = model.predict(reviews_vec)
        
        results = [
            SentimentResponse(review=review, sentiment=sentiment)
            for review, sentiment in zip(valid_reviews, sentiments)
        ]
        
        return BatchSentimentResponse(
            results=results,
            total_reviews=len(results)
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Batch prediction failed: {str(e)}"
        )

# ==========================================
# RUN
# ==========================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)