# Sentiment Analysis API

## Base URL
```
https://sentiment-ffqrr4mk2-zohraamna3s-projects.vercel.app
```

---

## Endpoints

### 1. Status Check
**GET** `https://sentiment-ffqrr4mk2-zohraamna3s-projects.vercel.app/`

**Response:**
```json
{
    "status": "active",
    "model_loaded": true
}
```

---

### 2. Single Review Prediction
**POST** `https://sentiment-ffqrr4mk2-zohraamna3s-projects.vercel.app/predict`

**Headers:**
```
Content-Type: application/json
```

**Body:**
```json
{
    "review": "The doctor was very helpful"
}
```

**Response:**
```json
{
    "review": "The doctor was very helpful",
    "sentiment": "positive",
    "confidence": "high"
}
```

---

### 3. Batch Prediction
**POST** `https://sentiment-ffqrr4mk2-zohraamna3s-projects.vercel.app/predict/batch`

**Headers:**
```
Content-Type: application/json
```

**Body:**
```json
{
    "reviews": [
        "Doctor was excellent",
        "Worst experience ever",
        "Average service"
    ]
}
```

**Response:**
```json
{
    "results": [
        {"review": "Doctor was excellent", "sentiment": "positive", "confidence": "high"},
        {"review": "Worst experience ever", "sentiment": "negative", "confidence": "high"},
        {"review": "Average service", "sentiment": "neutral", "confidence": "high"}
    ],
    "total_reviews": 3
}
```

---

### 4. Interactive API Docs
**Swagger UI:** `https://sentiment-ffqrr4mk2-zohraamna3s-projects.vercel.app/docs`

---

## Sentiment Values
| Sentiment | Description |
|-----------|-------------|
| `positive` | Positive review |
| `negative` | Negative review |
| `neutral`  | Neutral review |

---

## Tech Stack
- **Framework:** FastAPI
- **ML Model:** Scikit-learn
- **Deployment:** Vercel
- **Language:** Python
