# Sentiment Analysis API

## Base URL
```
https://sentiment-api-production-a47d.up.railway.app
```

---

## Endpoints

### 1. Status Check

**GET** `https://sentiment-api-production-a47d.up.railway.app/`

**Response:**
```json
{
    "status": "active",
    "model_loaded": true
}
```

---

### 2. Single Review Prediction

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
    "sentiment": "positive",
    "confidence": "high"
}
```

---

### 3. Batch Prediction

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
        {"review": "Doctor was excellent", "sentiment": "positive", "confidence": "high"},
        {"review": "Worst experience ever", "sentiment": "negative", "confidence": "high"},
        {"review": "Average service", "sentiment": "neutral", "confidence": "high"}
    ],
    "total_reviews": 3
}
```

---

## Sentiment Values

| Sentiment | Description |
|-----------|-------------|
| `positive` | Positive review |
| `negative` | Negative review |
| `neutral` | Neutral review |
