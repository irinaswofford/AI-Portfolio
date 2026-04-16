# import streamlit as st
# import os
# import pickle
# import base64
# import torch
# import logging

# from portfolio_data import portfolio_data
# from googleapiclient.discovery import build
# from google.auth.transport.requests import Request
# from google_auth_oauthlib.flow import Flow
# from email.mime.text import MIMEText
# from email.mime.multipart import MIMEMultipart
# from transformers import T5Tokenizer, T5ForConditionalGeneration
# from dotenv import load_dotenv

# # ---------------- CONFIG ----------------
# os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"
# os.environ["STREAMLIT_WATCH_FILE_SYSTEM"] = "false"

# logging.basicConfig(level=logging.INFO)

# load_dotenv()
# CSE_ID = os.getenv('CSE_ID')
# GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

# # Fix PyTorch warning
# try:
#     torch.classes.__path__ = []
# except AttributeError:
#     pass

# # ---------------- UI CLEAN ----------------
# st.markdown("""
# <style>
# div[data-testid="stSidebarNav"], div[data-testid="stSidebarHeader"] {
#     display:none;
# }
# </style>
# """, unsafe_allow_html=True)

# # ---------------- SESSION ----------------
# if 'page' not in st.session_state:
#     st.session_state.page = 'Home'

# # ---------------- SIDEBAR ----------------
# st.sidebar.title("Portfolio")
# st.session_state.page = st.sidebar.radio(
#     "Navigate",
#     ["Home"]
# )

# # ---------------- MODEL ----------------
# @st.cache_resource
# def get_t5_model():
#     tokenizer = T5Tokenizer.from_pretrained("t5-small")
#     model = T5ForConditionalGeneration.from_pretrained("t5-small")
#     return tokenizer, model

# tokenizer, model = get_t5_model()

# # ---------------- GOOGLE AUTH ----------------
# TOKEN_FILE = "token.pickle"

# client_config = {
#     "web": {
#         "client_id": st.secrets["client_id"],
#         "client_secret": st.secrets["client_secret"],
#         "auth_uri": st.secrets["auth_uri"],
#         "token_uri": st.secrets["token_uri"],
#         "redirect_uri": st.secrets["redirect_uri"]
#     }
# }

# SCOPES = [
#     "https://www.googleapis.com/auth/gmail.compose",
#     "https://www.googleapis.com/auth/userinfo.email",
#     "openid"
# ]

# def get_auth_code_from_url():
#     try:
#         code_value = st.query_params.get("code")
#         if isinstance(code_value, list):
#             return code_value[0]
#         return code_value
#     except:
#         return None

# def get_user_credentials():
#     creds = None

#     if os.path.exists(TOKEN_FILE):
#         try:
#             with open(TOKEN_FILE, "rb") as f:
#                 creds = pickle.load(f)
#         except:
#             creds = None

#     if creds and creds.valid:
#         return creds

#     if creds and creds.expired and creds.refresh_token:
#         try:
#             creds.refresh(Request())
#             with open(TOKEN_FILE, "wb") as f:
#                 pickle.dump(creds, f)
#             return creds
#         except:
#             creds = None

#     flow = Flow.from_client_config(
#         client_config,
#         scopes=SCOPES,
#         redirect_uri=st.secrets["redirect_uri"]
#     )

#     auth_url, _ = flow.authorization_url(prompt="consent")

#     st.info(f"[Login with Google]({auth_url})")

#     code = get_auth_code_from_url()

#     if code:
#         try:
#             flow.fetch_token(code=code)
#             creds = flow.credentials

#             with open(TOKEN_FILE, "wb") as f:
#                 pickle.dump(creds, f)

#             st.query_params.clear()
#             st.rerun()

#         except Exception as e:
#             st.error(f"Auth error: {e}")

#     return creds

# # ---------------- AI ----------------
# def generate_ai_answer(query):
#     inputs = tokenizer(query, return_tensors="pt")
#     outputs = model.generate(**inputs, max_length=100)
#     return tokenizer.decode(outputs[0], skip_special_tokens=True)

# def google_search(query):
#     try:
#         service = build("customsearch", "v1", developerKey=GOOGLE_API_KEY)
#         res = service.cse().list(q=query, cx=CSE_ID).execute()
#         return res["items"][0]["snippet"]
#     except:
#         return "No results"

# # ---------------- EMAIL ----------------
# def create_draft(creds, to, subject, body):
#     try:
#         service = build("gmail", "v1", credentials=creds)

#         message = MIMEMultipart()
#         message["to"] = to
#         message["subject"] = subject
#         message.attach(MIMEText(body))

#         raw = base64.urlsafe_b64encode(message.as_bytes()).decode()

#         draft = service.users().drafts().create(
#             userId="me",
#             body={"message": {"raw": raw}}
#         ).execute()

#         st.success("Draft created!")
#         return draft["id"]

#     except Exception as e:
#         st.error(str(e))

# # ---------------- ASSISTANT ----------------
# class PortfolioAssistant:
#     def __init__(self, data):
#         self.data = data

#     def get_response(self, query):
#         for q in self.data["portfolio_questions"]:
#             if query.lower() in q["question"].lower():
#                 return q["response"]
#         return None

# def handle_query(query, email):
#     assistant = PortfolioAssistant(portfolio_data)
#     response = assistant.get_response(query)

#     if response:
#         return response

#     ai = generate_ai_answer(query)
#     search = google_search(query)

#     combined = f"{ai}\n\n{search}"

#     if email:
#         creds = get_user_credentials()
#         if creds:
#             create_draft(creds, email, "Response", combined)
#             return "Email draft created"

#     return "Provide email for response"

# # ---------------- UI ----------------
# def main():
#     st.title("AI Portfolio Assistant")

#     if "email" not in st.session_state:
#         st.session_state.email = ""

#     query = st.text_input("Ask something")

#     if query:
#         result = handle_query(query, st.session_state.email)
#         st.write(result)

#         if "email" in result.lower():
#             st.session_state.email = st.text_input("Enter email")

# main()

import streamlit as st
import os
import pickle
import base64
import torch
from portfolio_data import portfolio_data 
from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import Flow
from google.oauth2 import id_token
from email.mime.text import MIMEText
from transformers import T5Tokenizer, T5ForConditionalGeneration
from langgraph.graph import StateGraph, END 
from dotenv import load_dotenv
from email.mime.multipart import MIMEMultipart
import logging

# --- Global Configurations ---
os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

load_dotenv()
CSE_ID = os.getenv('CSE_ID')
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
os.environ["STREAMLIT_WATCH_FILE_SYSTEM"] = "false" 

# Suppress a specific PyTorch warning
try:
    torch.classes.__path__ = []
except AttributeError:
    pass

# UI Styling
hide_elements = """
<style>
   div[data-testid="stSidebarNav"], div[data-testid="stSidebarHeader"] { display:none; }
</style>
"""
st.markdown(hide_elements, unsafe_allow_html=True)

# Initialize session state
if 'page' not in st.session_state:
    st.session_state.page = 'Home'

st.sidebar.title("AI/ML Projects")
selected_page = st.sidebar.radio(
    "Choose an option:",
    [
        "Home", "AI Project Mangement experience", 
        "Robotic Process Automation and Natural Language Processing",
        "Recurrent Neural Network-Long Short Term Memory Networks",
        "Supervised learning", "Unsupervised learning",
        "Conversational AI fine-tuned with Retrieval Augmented Generation",
        "Natural Language Processing & Generative AI",
        "Computer Vision - Image Text Extraction",
        "Computer Vision - Object Detection",
        "Sales Agent- Agentic Framework",
        "NLP and Generative AI: Speech-to-Text AI Voice Agent",
        "Computer Vision, Generative AI: A Text-to-Speech, Audio, and Video Generator",
        "Customer Chatbot Fine Tunned with ChatGPT Turbo"
    ],
    key="unique_radio_key",
)
st.session_state.page = selected_page

@st.cache_resource
def get_t5_model():
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    model = T5ForConditionalGeneration.from_pretrained("t5-small")
    return tokenizer, model

tokenizer, model = get_t5_model()

# OAuth Config
TOKEN_FILE = st.secrets["GOOGLE_TOKEN_PATH"]
client_config = {
    "web": {
        "client_id": st.secrets["client_id"],
        "client_secret": st.secrets["client_secret"],
        "auth_uri": st.secrets["auth_uri"],
        "token_uri": st.secrets["token_uri"],
        "redirect_uri": st.secrets["redirect_uri"]
    }
}
SCOPES = ["https://www.googleapis.com/auth/gmail.compose", "https://www.googleapis.com/auth/userinfo.email", "openid"]

def get_auth_code_from_url():
    try:
        query_params = st.query_params
        code_value = query_params.get("code")
        if isinstance(code_value, list):
            return code_value[0] if code_value else None
        return code_value
    except Exception as e:
        st.error(f"❌ Error extracting code: {e}")
        return None

def get_user_credentials():
    creds = None
    if os.path.exists(TOKEN_FILE):
        try:
            with open(TOKEN_FILE, 'rb') as token_file_obj:
                creds = pickle.load(token_file_obj)
        except Exception as e:
            os.remove(TOKEN_FILE)
            creds = None

    if creds and creds.valid:
        return creds

    if creds and creds.expired and creds.refresh_token:
        try:
            creds.refresh(Request())
            with open(TOKEN_FILE, 'wb') as f:
                pickle.dump(creds, f)
            return creds
        except Exception as e:
            creds = None

    if not creds:
        flow = Flow.from_client_config(client_config, scopes=SCOPES, redirect_uri=st.secrets["redirect_uri"])
        auth_url, _ = flow.authorization_url(prompt='consent', access_type='offline')
        
        st.info(f"### 🔐 [Click here to sign in with Google]({auth_url})")
        
        auth_code = get_auth_code_from_url()
        if auth_code:
            try:
                flow.fetch_token(code=auth_code)
                creds = flow.credentials
                with open(TOKEN_FILE, 'wb') as f:
                    pickle.dump(creds, f)
                st.query_params.pop("code", None)
                st.rerun()
            except Exception as e:
                st.error(f"❌ Failed to fetch token: {e}")
    return creds

def create_gmail_draft(creds, recipient, subject, body):
    try:
        service = build("gmail", "v1", credentials=creds)
        message = MIMEMultipart()
        message["to"] = recipient
        message["subject"] = subject
        message.attach(MIMEText(body, "plain"))
        raw_message = base64.urlsafe_b64encode(message.as_bytes()).decode()
        draft_body = {"message": {"raw": raw_message}}
        draft = service.users().drafts().create(userId="me", body=draft_body).execute()
        st.success(f"✅ Email draft created! ID: {draft['id']}")
        return True
    except Exception as e:
        st.error(f"❌ Failed to create draft: {e}")
        return False

# --- Page Routing Logic ---
if st.session_state.page == "Home":
    st.title("Hi, I am Irina Swofford")
    st.write("Welcome to my AI Portfolio.")
    
    # Initialize interaction state
    if "user_query" not in st.session_state: st.session_state.user_query = ""
    if "user_email" not in st.session_state: st.session_state.user_email = ""
    
    user_query = st.text_input("Ask me anything about my experience:")
    if user_query:
        # Business logic for query would go here
        st.write(f"Processing: {user_query}")

# Helper to load sub-pages
def load_page(path):
    if os.path.exists(path):
        with open(path, "r") as f:
            code = compile(f.read(), path, 'exec')
            exec(code, globals())
    else:
        st.error(f"Page {path} not found.")

# Routing
pages_map = {
    "AI Project Mangement experience": "pages/project_roadmap.py",
    "Robotic Process Automation and Natural Language Processing": "pages/analyse_workflow.py",
    "Recurrent Neural Network-Long Short Term Memory Networks": "pages/lstm.py",
    "Supervised learning": "pages/features-sales.py",
    "Unsupervised learning": "pages/segmentation.py",
    "Conversational AI fine-tuned with Retrieval Augmented Generation": "pages/conversational_ai.py",
    "Natural Language Processing & Generative AI": "pages/survey_proofreading.py",
    "Computer Vision - Image Text Extraction": "pages/text-extraction-image.py",
    "Computer Vision - Object Detection": "pages/object_detection_size.py",
    "Sales Agent- Agentic Framework": "pages/sales_agent.py",
    "NLP and Generative AI: Speech-to-Text AI Voice Agent": "pages/speach-to-text.py",
    "Computer Vision, Generative AI: A Text-to-Speech, Audio, and Video Generator": "pages/text-to-speach.py",
}

if st.session_state.page in pages_map:
    load_page(pages_map[st.session_state.page])
