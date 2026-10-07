import os
import requests
import streamlit as st

API_URL = "http://localhost:8000/agent"

st.set_page_config(page_title ="Support Agent Chat")
st.title("Customer Support RAG Assistant")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("Ask a support question...")

if prompt:
    st.session_state.messages.append({"role" : "user", "content" :prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    try:
        response = requests.post(API_URL, json={"question":prompt}, timeout=20)
        response.raise_for_status()
        payload = response.json()
        answer = payload.get("answer", "No answer return by the agent")
    except requests.RequestException as ex:
        answer = f" The agent is not available right now. Error {ex}"

    st.session_state.messages.append({"role" : "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.markdown(answer)


with st.sidebar:
    st.subheader("Agent health")  # Small heading for the health-status section
    try:  # The health check itself can fail if the backend is down
        health_response = requests.get("http://localhost:8000/health", timeout=10)  # GET the FastAPI /health endpoint
        if health_response.ok:  # True when the status code is < 400 (the service responded successfully)
            st.success("FastAPI service is responding")  # Green success banner
            st.json(health_response.json())  # Pretty-print the full health JSON (per-service status details)
        else:  # The service replied but with an error status
            st.warning("FastAPI service responded with an error")  # Yellow warning banner
    except requests.RequestException:  # Could not reach the service at all (not started, wrong port, etc.)
        st.error("Service is not running yet. Start the FastAPI app first.")  # Red error banner with a hint