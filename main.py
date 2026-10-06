"""
Step 2: a small backend that serves the saved model as an API.

Run on your own computer:   uvicorn main:app --reload
Then open in a browser:     http://127.0.0.1:8000/docs

Endpoints
  GET /                          is the API running? what model is loaded?
  GET /predict?lag_1=..&lag_2=.. predict from two prices you type in
  GET /predict/live?symbol=TSLA  fetch the latest prices from Finnhub, then predict
"""
import json
import os
from datetime import datetime, timedelta, timezone

import joblib
import pandas as pd
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Load the saved model ONCE, when the server starts
model = joblib.load("model.joblib")
with open("model_info.json") as f:
    info = json.load(f)

app = FastAPI(
    title="Stock price prediction API",
    description="Predicts the next closing price from the last two closes. "
    "A teaching example, not investment advice.",
)

# Allow web pages on other addresses (any frontend) to call this API
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET"])


def predict_next(lag_1: float, lag_2: float) -> float:
    X = pd.DataFrame({"lag_1": [lag_1], "lag_2": [lag_2]})
    return round(float(model.predict(X)[0]), 2)


def finnhub_get(endpoint: str, params: dict) -> dict:
    key = os.environ.get("FINNHUB_API_KEY")
    if not key:
        raise HTTPException(status_code=500, detail="FINNHUB_API_KEY is not set on the server.")

    try:
        response = requests.get(
            f"https://finnhub.io/api/v1/{endpoint}",
            params={**params, "token": key},
            timeout=15,
        )
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"Finnhub request failed: {exc}") from exc

    try:
        reply = response.json()
    except ValueError as exc:
        raise HTTPException(status_code=502, detail="Finnhub returned an invalid response.") from exc

    if not response.ok:
        message = reply.get("error", "Finnhub request failed.") if isinstance(reply, dict) else "Finnhub request failed."
        raise HTTPException(status_code=502, detail=message)
    if not isinstance(reply, dict):
        raise HTTPException(status_code=502, detail="Finnhub returned an unexpected response.")
    if reply.get("error"):
        raise HTTPException(status_code=502, detail=reply["error"])

    return reply


@app.get("/")
def home():
    return {"message": "The API is running. Open /docs to try it.", **info}


@app.get("/predict")
def predict(lag_1: float, lag_2: float):
    """Predict tomorrow's close. lag_1 = latest close, lag_2 = the close before it."""
    return {
        "lag_1": lag_1,
        "lag_2": lag_2,
        "predicted_next_close": predict_next(lag_1, lag_2),
        "naive_forecast": lag_1,
    }


@app.get("/search")
def search_symbol(keywords: str):
    """Search Finnhub for stock symbols matching a company keyword."""
    reply = finnhub_get("search", {"q": keywords})
    matches = reply.get("result") or []
    cleaned = [
        {
            "symbol": match.get("symbol") or match.get("displaySymbol"),
            "name": match.get("description"),
            "type": match.get("type"),
        }
        for match in matches
        if isinstance(match, dict)
    ]
    return {"keywords": keywords, "matches": cleaned}


@app.get("/predict/live")
def predict_live(symbol: str = "TSLA"):
    """Get the latest two daily closes from Finnhub, then predict the next one."""
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=14)
    reply = finnhub_get(
        "stock/candle",
        {
            "symbol": symbol,
            "resolution": "D",
            "from": int(start.timestamp()),
            "to": int(end.timestamp()),
        },
    )

    closes = reply.get("c")
    timestamps = reply.get("t")
    if reply.get("s") != "ok" or not isinstance(closes, list) or not isinstance(timestamps, list):
        raise HTTPException(status_code=502, detail=f"Finnhub returned no daily candles for '{symbol}'.")
    if len(closes) < 2 or len(timestamps) != len(closes):
        raise HTTPException(status_code=502, detail=f"Finnhub returned fewer than two daily closes for '{symbol}'.")

    try:
        last_close = float(closes[-1])
        previous_close = float(closes[-2])
        latest_trading_day = datetime.fromtimestamp(timestamps[-1], timezone.utc).date().isoformat()
    except (TypeError, ValueError, OverflowError) as exc:
        raise HTTPException(status_code=502, detail="Finnhub returned invalid daily candle data.") from exc

    return {
        "symbol": symbol.upper(),
        "latest_trading_day": latest_trading_day,
        "last_close": last_close,
        "previous_close": previous_close,
        "predicted_next_close": predict_next(last_close, previous_close),
        "naive_forecast": last_close,
        "model": info["model"],
        "trained_on": info["trained_on"],
    }
