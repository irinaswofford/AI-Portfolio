import streamlit as st
import streamlit.components.v1 as components
WEBHOOK_URL = st.secrets["WEBHOOK_URL"] #uncomment for production
components.html("""
<script 
    src="https://irinaswofford.github.io/assistant-widget/widget.js"
    data-webhook="WEBHOOK_URL"
    data-client-key="test_client_002"
    data-brand-color="#8b5cf6"
    data-agent-name="Marvia AI"
    data-welcome="Hi! How can I help you today?"
    data-theme="light"
></script>
""", width=1000,height=700)
