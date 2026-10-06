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
import math
import os
from datetime import datetime, timezone

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
    """Use Finnhub's latest quote and previous close to predict the next close."""
    reply = finnhub_get("quote", {"symbol": symbol})

    try:
        last_close = float(reply["c"])
        previous_close = float(reply["pc"])
        quote_timestamp = int(reply["t"])
        latest_trading_day = datetime.fromtimestamp(quote_timestamp, timezone.utc).date().isoformat()
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise HTTPException(status_code=502, detail=f"Finnhub returned invalid quote data for '{symbol}'.") from exc
    if not math.isfinite(last_close) or not math.isfinite(previous_close) or last_close <= 0 or previous_close <= 0:
        raise HTTPException(status_code=502, detail=f"Finnhub returned no usable quote data for '{symbol}'.")

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
