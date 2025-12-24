# Sentiment Analysis API

## Base URL
```
https://sentiment-api-production-a47d.up.railway.app
```

---

## Endpoints

### 1. Single Review Prediction

**POST** `https://sentiment-api-production-a47d.up.railway.app/predict`

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
    "sentiment": "positive"
}
```

---

### 2. Batch Prediction

**POST** `https://sentiment-api-production-a47d.up.railway.app/predict/batch`

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
        {"review": "Doctor was excellent", "sentiment": "positive"},
        {"review": "Worst experience ever", "sentiment": "negative"},
        {"review": "Average service", "sentiment": "neutral"}
    ],
    "total_reviews": 3
}
```

---

### 3. Health Check

**GET** `https://sentiment-api-production-a47d.up.railway.app/health`

**Response:**
```json
{
    "status": "healthy",
    "model_loaded": true,
    "vectorizer_loaded": true,
    "model_file_exists": true,
    "vectorizer_file_exists": true
}
```

---

## Sentiment Values
| Sentiment | Description |
|-----------|-------------|
| `positive` | Positive review |
| `negative` | Negative review |
| `neutral` | Neutral review |
