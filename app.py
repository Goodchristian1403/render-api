import os

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="Stock Prediction", page_icon="📈", layout="wide")
st.title("Stock Price Prediction")
st.caption("Frontend for the FastAPI stock prediction backend")

api_url = st.sidebar.text_input("API URL", value=API_URL)

try:
    health = requests.get(f"{api_url}/", timeout=10)
    health.raise_for_status()
    st.sidebar.success("API connection is working")
except requests.RequestException as exc:
    st.sidebar.warning(f"Could not reach API: {exc}")

manual_tab, live_tab, search_tab = st.tabs(["Manual inputs", "Live symbol data", "Search symbols"])

with manual_tab:
    lag_1 = st.number_input("Latest close", value=100.0, step=0.01)
    lag_2 = st.number_input("Previous close", value=98.0, step=0.01)

    if st.button("Predict from prices", type="primary"):
        try:
            response = requests.get(
                f"{api_url}/predict",
                params={"lag_1": lag_1, "lag_2": lag_2},
                timeout=15,
            )
            if response.status_code >= 400:
                detail = response.json().get("detail", response.text)
                st.error(f"API request failed ({response.status_code}): {detail}")
            else:
                data = response.json()
                st.subheader("Prediction")
                st.metric("Predicted next close", f"${data['predicted_next_close']:.2f}")
                st.json(data)
        except requests.RequestException as exc:
            st.error(f"Prediction request failed: {exc}")

with live_tab:
    symbol = st.text_input("Stock symbol", value="TSLA")

    if st.button("Predict live"):
        try:
            response = requests.get(
                f"{api_url}/predict/live",
                params={"symbol": symbol},
                timeout=20,
            )
            if response.status_code >= 400:
                detail = response.json().get("detail", response.text)
                st.error(f"Live prediction failed ({response.status_code}): {detail}")
            else:
                data = response.json()
                st.subheader(f"Live prediction for {data.get('symbol', symbol)}")
                col1, col2, col3 = st.columns(3)
                col1.metric("Last close", f"${data['last_close']:.2f}")
                col2.metric("Previous close", f"${data['previous_close']:.2f}")
                col3.metric("Predicted next close", f"${data['predicted_next_close']:.2f}")

                st.json(data)
        except requests.RequestException as exc:
            st.error(f"Live prediction request failed: {exc}")

with search_tab:
    keyword = st.text_input("Search company keyword", value="apple")

    if st.button("Search symbols"):
        try:
            response = requests.get(
                f"{api_url}/search",
                params={"keywords": keyword},
                timeout=20,
            )
            if response.status_code >= 400:
                detail = response.json().get("detail", response.text)
                st.error(f"Symbol search failed ({response.status_code}): {detail}")
            else:
                data = response.json()
                matches = data.get("matches", [])
                st.subheader(f"Matches for '{data.get('keywords', keyword)}'")

                if not matches:
                    st.info("No matching symbols were found.")
                else:
                    for match in matches[:10]:
                        with st.container():
                            st.markdown(
                                f"**{match['symbol']}** — {match['name']} "
                                f"({match.get('currency', 'N/A')}, {match.get('region', 'N/A')})"
                            )
                            st.caption(
                                f"Type: {match.get('type', 'N/A')} | "
                                f"Market: {match.get('market_open', 'N/A')} - {match.get('market_close', 'N/A')} | "
                                f"Match score: {match.get('match_score', 'N/A')}"
                            )
                            if st.button(f"Use {match['symbol']}", key=f"symbol_{match['symbol']}"):
                                st.session_state["selected_symbol"] = match["symbol"]
                                st.success(f"Selected symbol: {match['symbol']}")

                    if "selected_symbol" in st.session_state:
                        st.markdown(f"Selected ticker: **{st.session_state['selected_symbol']}**")
                        st.code(f"Use this in the live prediction tab: {st.session_state['selected_symbol']}")
        except requests.RequestException as exc:
            st.error(f"Symbol search request failed: {exc}")
