import streamlit as st
import os
import pickle
import base64
import torch
import logging

from portfolio_data import portfolio_data
from googleapiclient.discovery import build
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import Flow
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from transformers import T5Tokenizer, T5ForConditionalGeneration
from dotenv import load_dotenv

# ---------------- CONFIG ----------------
os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"
os.environ["STREAMLIT_WATCH_FILE_SYSTEM"] = "false"

logging.basicConfig(level=logging.INFO)

load_dotenv()
CSE_ID = os.getenv('CSE_ID')
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

# Fix PyTorch warning
try:
    torch.classes.__path__ = []
except AttributeError:
    pass

# ---------------- UI CLEAN ----------------
st.markdown("""
<style>
div[data-testid="stSidebarNav"], div[data-testid="stSidebarHeader"] {
    display:none;
}
</style>
""", unsafe_allow_html=True)

# ---------------- SESSION ----------------
if 'page' not in st.session_state:
    st.session_state.page = 'Home'

# ---------------- SIDEBAR ----------------
st.sidebar.title("Portfolio")
st.session_state.page = st.sidebar.radio(
    "Navigate",
    ["Home"]
)

# ---------------- MODEL ----------------
@st.cache_resource
def get_t5_model():
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    model = T5ForConditionalGeneration.from_pretrained("t5-small")
    return tokenizer, model

tokenizer, model = get_t5_model()

# ---------------- GOOGLE AUTH ----------------
TOKEN_FILE = "token.pickle"

client_config = {
    "web": {
        "client_id": st.secrets["client_id"],
        "client_secret": st.secrets["client_secret"],
        "auth_uri": st.secrets["auth_uri"],
        "token_uri": st.secrets["token_uri"],
        "redirect_uri": st.secrets["redirect_uri"]
    }
}

SCOPES = [
    "https://www.googleapis.com/auth/gmail.compose",
    "https://www.googleapis.com/auth/userinfo.email",
    "openid"
]

def get_auth_code_from_url():
    try:
        code_value = st.query_params.get("code")
        if isinstance(code_value, list):
            return code_value[0]
        return code_value
    except:
        return None

def get_user_credentials():
    creds = None

    if os.path.exists(TOKEN_FILE):
        try:
            with open(TOKEN_FILE, "rb") as f:
                creds = pickle.load(f)
        except:
            creds = None

    if creds and creds.valid:
        return creds

    if creds and creds.expired and creds.refresh_token:
        try:
            creds.refresh(Request())
            with open(TOKEN_FILE, "wb") as f:
                pickle.dump(creds, f)
            return creds
        except:
            creds = None

    flow = Flow.from_client_config(
        client_config,
        scopes=SCOPES,
        redirect_uri=st.secrets["redirect_uri"]
    )

    auth_url, _ = flow.authorization_url(prompt="consent")

    st.info(f"[Login with Google]({auth_url})")

    code = get_auth_code_from_url()

    if code:
        try:
            flow.fetch_token(code=code)
            creds = flow.credentials

            with open(TOKEN_FILE, "wb") as f:
                pickle.dump(creds, f)

            st.query_params.clear()
            st.rerun()

        except Exception as e:
            st.error(f"Auth error: {e}")

    return creds

# ---------------- AI ----------------
def generate_ai_answer(query):
    inputs = tokenizer(query, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=100)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def google_search(query):
    try:
        service = build("customsearch", "v1", developerKey=GOOGLE_API_KEY)
        res = service.cse().list(q=query, cx=CSE_ID).execute()
        return res["items"][0]["snippet"]
    except:
        return "No results"

# ---------------- EMAIL ----------------
def create_draft(creds, to, subject, body):
    try:
        service = build("gmail", "v1", credentials=creds)

        message = MIMEMultipart()
        message["to"] = to
        message["subject"] = subject
        message.attach(MIMEText(body))

        raw = base64.urlsafe_b64encode(message.as_bytes()).decode()

        draft = service.users().drafts().create(
            userId="me",
            body={"message": {"raw": raw}}
        ).execute()

        st.success("Draft created!")
        return draft["id"]

    except Exception as e:
        st.error(str(e))

# ---------------- ASSISTANT ----------------
class PortfolioAssistant:
    def __init__(self, data):
        self.data = data

    def get_response(self, query):
        for q in self.data["portfolio_questions"]:
            if query.lower() in q["question"].lower():
                return q["response"]
        return None

def handle_query(query, email):
    assistant = PortfolioAssistant(portfolio_data)
    response = assistant.get_response(query)

    if response:
        return response

    ai = generate_ai_answer(query)
    search = google_search(query)

    combined = f"{ai}\n\n{search}"

    if email:
        creds = get_user_credentials()
        if creds:
            create_draft(creds, email, "Response", combined)
            return "Email draft created"

    return "Provide email for response"

# ---------------- UI ----------------
def main():
    st.title("AI Portfolio Assistant")

    if "email" not in st.session_state:
        st.session_state.email = ""

    query = st.text_input("Ask something")

    if query:
        result = handle_query(query, st.session_state.email)
        st.write(result)

        if "email" in result.lower():
            st.session_state.email = st.text_input("Enter email")

main()
