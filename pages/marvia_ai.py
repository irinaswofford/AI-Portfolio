import streamlit as st
import streamlit.components.v1 as components

# Load secrets
WEBHOOK_URL = st.secrets["WEBHOOK_URL"]
CLIENT_KEY = st.secrets["FAX_CLIENT_KEY"]

# Use f-string to replace variables in HTML
components.html(
    f"""
    <script 
        src="https://irinaswofford.github.io/assistant-widget/widget.js"
        data-webhook="{WEBHOOK_URL}"
        data-client-key="{CLIENT_KEY}"
        data-brand-color="#8b5cf6"
        data-agent-name="Marvia AI"
        data-welcome="Hi! How can I help you today?"
        data-theme="light"
    ></script>
    """,
    width=1000,
    height=700
)
