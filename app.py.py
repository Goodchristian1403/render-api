import requests, streamlit as st
api_url = "https://render-api-2y5x.onrender.com"
symbol = st.text_input("Stock symbol", "TSLA")
if st.button("Predict next close"):
    r = requests.get(f"{api_url}/predict/live",
                     params={"symbol": symbol})
    data = r.json()
    st.metric("Predicted next close",
