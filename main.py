# import streamlit as st
# import os
# import pickle
# import base64
# import torch
# from portfolio_data import portfolio_data 
# from googleapiclient.discovery import build
# from google.oauth2.credentials import Credentials
# from google.auth.transport.requests import Request
# from google_auth_oauthlib.flow import Flow
# from google.oauth2 import id_token
# from email.mime.text import MIMEText
# from transformers import T5Tokenizer, T5ForConditionalGeneration
# from langgraph.graph import StateGraph, END 
# from dotenv import load_dotenv
# from email.mime.multipart import MIMEMultipart
# import logging

# # --- Global Configurations ---
# os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# load_dotenv()
# CSE_ID = os.getenv('CSE_ID')
# GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
# os.environ["STREAMLIT_WATCH_FILE_SYSTEM"] = "false" 

# # Suppress a specific PyTorch warning
# try:
#     torch.classes.__path__ = []
# except AttributeError:
#     pass

# # UI Styling
# hide_elements = """
# <style>
#    div[data-testid="stSidebarNav"], div[data-testid="stSidebarHeader"] { display:none; }
# </style>
# """
# st.markdown(hide_elements, unsafe_allow_html=True)

# # Initialize session state
# if 'page' not in st.session_state:
#     st.session_state.page = 'Home'

# st.sidebar.title("AI/ML Projects")
# selected_page = st.sidebar.radio(
#     "Choose an option:",
#     [
#         "Home", "AI Project Mangement experience", 
#         "Robotic Process Automation and Natural Language Processing",
#         "Recurrent Neural Network-Long Short Term Memory Networks",
#         "Supervised learning", "Unsupervised learning",
#         "Conversational AI fine-tuned with Retrieval Augmented Generation",
#         "Natural Language Processing & Generative AI",
#         "Computer Vision - Image Text Extraction",
#         "Computer Vision - Object Detection",
#         "Sales Agent- Agentic Framework",
#         "NLP and Generative AI: Speech-to-Text AI Voice Agent",
#         "Computer Vision, Generative AI: A Text-to-Speech, Audio, and Video Generator",
#         "Customer Chatbot Fine Tunned with ChatGPT Turbo"
#     ],
#     key="unique_radio_key",
# )
# st.session_state.page = selected_page

# @st.cache_resource
# def get_t5_model():
#     tokenizer = T5Tokenizer.from_pretrained("t5-small")
#     model = T5ForConditionalGeneration.from_pretrained("t5-small")
#     return tokenizer, model

# tokenizer, model = get_t5_model()

# # OAuth Config
# TOKEN_FILE = st.secrets["GOOGLE_TOKEN_PATH"]
# client_config = {
#     "web": {
#         "client_id": st.secrets["client_id"],
#         "client_secret": st.secrets["client_secret"],
#         "auth_uri": st.secrets["auth_uri"],
#         "token_uri": st.secrets["token_uri"],
#         "redirect_uri": st.secrets["redirect_uri"]
#     }
# }
# SCOPES = ["https://www.googleapis.com/auth/gmail.compose", "https://www.googleapis.com/auth/userinfo.email", "openid"]

# def get_auth_code_from_url():
#     try:
#         query_params = st.query_params
#         code_value = query_params.get("code")
#         if isinstance(code_value, list):
#             return code_value[0] if code_value else None
#         return code_value
#     except Exception as e:
#         st.error(f"❌ Error extracting code: {e}")
#         return None

# def get_user_credentials():
#     creds = None
#     if os.path.exists(TOKEN_FILE):
#         try:
#             with open(TOKEN_FILE, 'rb') as token_file_obj:
#                 creds = pickle.load(token_file_obj)
#         except Exception as e:
#             os.remove(TOKEN_FILE)
#             creds = None

#     if creds and creds.valid:
#         return creds

#     if creds and creds.expired and creds.refresh_token:
#         try:
#             creds.refresh(Request())
#             with open(TOKEN_FILE, 'wb') as f:
#                 pickle.dump(creds, f)
#             return creds
#         except Exception as e:
#             creds = None

#     if not creds:
#         flow = Flow.from_client_config(client_config, scopes=SCOPES, redirect_uri=st.secrets["redirect_uri"])
#         auth_url, _ = flow.authorization_url(prompt='consent', access_type='offline')
        
#         st.info(f"### 🔐 [Click here to sign in with Google]({auth_url})")
        
#         auth_code = get_auth_code_from_url()
#         if auth_code:
#             try:
#                 flow.fetch_token(code=auth_code)
#                 creds = flow.credentials
#                 with open(TOKEN_FILE, 'wb') as f:
#                     pickle.dump(creds, f)
#                 st.query_params.pop("code", None)
#                 st.rerun()
#             except Exception as e:
#                 st.error(f"❌ Failed to fetch token: {e}")
#     return creds

# def create_gmail_draft(creds, recipient, subject, body):
#     try:
#         service = build("gmail", "v1", credentials=creds)
#         message = MIMEMultipart()
#         message["to"] = recipient
#         message["subject"] = subject
#         message.attach(MIMEText(body, "plain"))
#         raw_message = base64.urlsafe_b64encode(message.as_bytes()).decode()
#         draft_body = {"message": {"raw": raw_message}}
#         draft = service.users().drafts().create(userId="me", body=draft_body).execute()
#         st.success(f"✅ Email draft created! ID: {draft['id']}")
#         return True
#     except Exception as e:
#         st.error(f"❌ Failed to create draft: {e}")
#         return False

# # --- Page Routing Logic ---
# if st.session_state.page == "Home":
#     st.title("Hi, I am Irina Swofford")
#     st.write("Welcome to my AI Portfolio.")
    
#     # Initialize interaction state
#     if "user_query" not in st.session_state: st.session_state.user_query = ""
#     if "user_email" not in st.session_state: st.session_state.user_email = ""
    
#     user_query = st.text_input("Ask me anything about my experience:")
#     if user_query:
#         # Business logic for query would go here
#         st.write(f"Processing: {user_query}")

# # Helper to load sub-pages
# def load_page(path):
#     if os.path.exists(path):
#         with open(path, "r") as f:
#             code = compile(f.read(), path, 'exec')
#             exec(code, globals())
#     else:
#         st.error(f"Page {path} not found.")

# # Routing
# pages_map = {
#     "AI Project Mangement experience": "pages/project_roadmap.py",
#     "Robotic Process Automation and Natural Language Processing": "pages/analyse_workflow.py",
#     "Recurrent Neural Network-Long Short Term Memory Networks": "pages/lstm.py",
#     "Supervised learning": "pages/features-sales.py",
#     "Unsupervised learning": "pages/segmentation.py",
#     "Conversational AI fine-tuned with Retrieval Augmented Generation": "pages/conversational_ai.py",
#     "Natural Language Processing & Generative AI": "pages/survey_proofreading.py",
#     "Computer Vision - Image Text Extraction": "pages/text-extraction-image.py",
#     "Computer Vision - Object Detection": "pages/object_detection_size.py",
#     "Sales Agent- Agentic Framework": "pages/sales_agent.py",
#     "NLP and Generative AI: Speech-to-Text AI Voice Agent": "pages/speach-to-text.py",
#     "Computer Vision, Generative AI: A Text-to-Speech, Audio, and Video Generator": "pages/text-to-speach.py",
# }

# if st.session_state.page in pages_map:
#     load_page(pages_map[st.session_state.page])


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

CSE_ID = os.getenv("CSE_ID")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# ---------------- SAFE TORCH PATCH ----------------
try:
    if hasattr(torch, "classes") and hasattr(torch.classes, "__path__"):
        torch.classes.__path__ = []
except Exception:
    pass

# ---------------- UI ----------------
st.markdown(
    """
    <style>
    div[data-testid="stSidebarNav"], div[data-testid="stSidebarHeader"] {
        display:none;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------- SESSION ----------------
if "page" not in st.session_state:
    st.session_state.page = "Home"

# ---------------- SIDEBAR ----------------
st.sidebar.title("AI/ML Projects")

selected_page = st.sidebar.radio(
    "Choose an option:",
    [
        "Home",
        "AI Project Mangement experience",
        "Robotic Process Automation and Natural Language Processing",
        "Recurrent Neural Network-Long Short Term Memory Networks",
        "Supervised learning",
        "Unsupervised learning",
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

# ---------------- MODEL ----------------
@st.cache_resource
def get_t5_model():
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    model = T5ForConditionalGeneration.from_pretrained("t5-small")
    return tokenizer, model

tokenizer, model = get_t5_model()

# ---------------- OAUTH ----------------
TOKEN_FILE = st.secrets.get("GOOGLE_TOKEN_PATH", "token.pkl")

client_config = {
    "web": {
        "client_id": st.secrets.get("client_id"),
        "client_secret": st.secrets.get("client_secret"),
        "auth_uri": st.secrets.get("auth_uri"),
        "token_uri": st.secrets.get("token_uri"),
        "redirect_uri": st.secrets.get("redirect_uri")
    }
}

SCOPES = [
    "https://www.googleapis.com/auth/gmail.compose",
    "https://www.googleapis.com/auth/userinfo.email",
    "openid"
]

# ---------------- AUTH ----------------
def get_auth_code_from_url():
    try:
        return st.query_params.get("code")
    except Exception:
        return None


def get_user_credentials():
    creds = None

    if os.path.exists(TOKEN_FILE):
        try:
            with open(TOKEN_FILE, "rb") as f:
                creds = pickle.load(f)
        except Exception:
            os.remove(TOKEN_FILE)
            creds = None

    if creds and creds.valid:
        return creds

    if creds and creds.expired and creds.refresh_token:
        try:
            creds.refresh(Request())
            with open(TOKEN_FILE, "wb") as f:
                pickle.dump(creds, f)
            return creds
        except Exception:
            creds = None

    if not creds:
        flow = Flow.from_client_config(
            client_config,
            scopes=SCOPES,
            redirect_uri=st.secrets.get("redirect_uri"),
        )

        auth_url, _ = flow.authorization_url(
            prompt="consent",
            access_type="offline"
        )

        st.info("### 🔐 Login with Google")
        st.markdown(auth_url)

        code = get_auth_code_from_url()

        if code:
            try:
                flow.fetch_token(code=code)
                creds = flow.credentials

                with open(TOKEN_FILE, "wb") as f:
                    pickle.dump(creds, f)

                st.rerun()

            except Exception as e:
                st.error(f"Auth error: {e}")

    return creds


# ---------------- GMAIL ----------------
def create_gmail_draft(creds, recipient, subject, body):
    try:
        service = build("gmail", "v1", credentials=creds)

        message = MIMEMultipart()
        message["to"] = recipient
        message["subject"] = subject
        message.attach(MIMEText(body, "plain"))

        raw = base64.urlsafe_b64encode(message.as_bytes()).decode()

        draft = service.users().drafts().create(
            userId="me",
            body={"message": {"raw": raw}}
        ).execute()

        st.success(f"Draft created: {draft['id']}")
        return True

    except Exception as e:
        st.error(f"Gmail error: {e}")
        return False


# ---------------- HOME PAGE ----------------
if st.session_state.page == "Home":
    st.title("Hi, I’m Irina Swofford")
    st.subheader("AI Engineer • ML Developer • Project Manager")

    st.write("""
I specialize in building end-to-end AI/ML systems that combine machine learning, automation, and generative AI with real-world business applications.

My focus is on transforming complex technical problems into scalable, practical solutions that improve decision-making, automate workflows, and enhance user experiences.

---

### 🔹 Portfolio Overview

This portfolio demonstrates hands-on experience across multiple AI domains:

- Machine Learning: supervised & unsupervised learning, predictive modeling  
- Deep Learning: LSTM networks and sequence modeling  
- Natural Language Processing: text classification, RAG, conversational AI  
- Computer Vision: object detection and OCR systems  
- Generative AI: chatbot systems and content generation pipelines  
- AI Agents & Automation: workflow automation and agent-based systems  

---

### 🔹 What I build

- Production-ready AI systems (not just experiments)
- End-to-end ML pipelines from data to deployment
- API-integrated AI applications
- Automation systems for real business workflows
- AI tools that are explainable and usable by non-technical users  

---

### 🔹 Explore Projects

Use the sidebar to navigate through each AI domain and see how systems are designed, trained, and deployed.

You can also ask:

- “Explain your NLP projects”
- “Show your deep learning work”
- “What AI agents did you build?”
""")

    if "user_query" not in st.session_state:
        st.session_state.user_query = ""

    user_query = st.text_input("Ask me anything about my experience:")

    if user_query:
        st.write(f"Processing: {user_query}")


# ---------------- PAGE LOADER ----------------
def load_page(path):
    if os.path.exists(path):
        with open(path, "r") as f:
            code = compile(f.read(), path, "exec")
            exec(code, globals())
    else:
        st.error(f"Page not found: {path}")


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


