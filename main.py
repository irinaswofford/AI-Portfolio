# import streamlit as st
# import os
# import pickle
# import base64
# import logging
# import torch

# from portfolio_data import portfolio_data
# from googleapiclient.discovery import build
# from google.auth.transport.requests import Request
# from google_auth_oauthlib.flow import Flow
# from email.mime.text import MIMEText
# from email.mime.multipart import MIMEMultipart
# from transformers import T5Tokenizer, T5ForConditionalGeneration
# from dotenv import load_dotenv

# # --- Global Configurations ---
# os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"
# os.environ["STREAMLIT_WATCH_FILE_SYSTEM"] = "false"
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# load_dotenv()
# CSE_ID = os.getenv('CSE_ID')
# GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

# # --- Safe Torch Patch ---
# try:
#     if hasattr(torch, "classes") and hasattr(torch.classes, "__path__"):
#         torch.classes.__path__ = []
# except Exception:
#     pass

# # --- UI Customization ---
# st.markdown("""
#     <style>
#     div[data-testid="stSidebarNav"], div[data-testid="stSidebarHeader"] { display:none; }
#     </style>
#     """, unsafe_allow_html=True)

# # --- Session State Initialization ---
# if "page" not in st.session_state:
#     st.session_state.page = "Home"
# if "user_query" not in st.session_state:
#     st.session_state.user_query = ""
# if "user_email" not in st.session_state:
#     st.session_state.user_email = ""
# if "email_sent" not in st.session_state:
#     st.session_state.email_sent = False

# # --- Navigation Sidebar ---
# st.sidebar.title("AI/ML Projects & Management")
# selected_page = st.sidebar.radio(
#     "Choose an option:",
#     [
#         "Home", 
#         "AI Project Mangement experience", 
#         "Robotic Process Automation and Natural Language Processing",
#         "Recurrent Neural Network-Long Short Term Memory Networks",
#         "Supervised learning", 
#         "Unsupervised learning",
#         "Conversational AI fine-tuned with Retrieval Augmented Generation",
#         "Natural Language Processing & Generative AI",
#         "Computer Vision - Image Text Extraction",
#         "Computer Vision - Object Detection",
#         "Sales Agent- Agentic Framework",
#         "NLP and Generative AI: Speech-to-Text AI Voice Agent",
#         "Computer Vision, Generative AI: A Text-to-Speech, Audio, and Video Generator",
#         "Customer Chatbot Fine Tunned with ChatGPT Turbo"
#     ],
#     key="unique_radio_key"
# )
# st.session_state.page = selected_page

# # --- Model Loading ---
# @st.cache_resource
# def get_t5_model():
#     tokenizer = T5Tokenizer.from_pretrained("t5-small")
#     model = T5ForConditionalGeneration.from_pretrained("t5-small")
#     return tokenizer, model

# tokenizer, model = get_t5_model()

# # --- Google OAuth Logic ---
# TOKEN_FILE = st.secrets.get("GOOGLE_TOKEN_PATH", "token.pkl")
# client_config = {
#     "web": {
#         "client_id": st.secrets.get("client_id"),
#         "client_secret": st.secrets.get("client_secret"),
#         "auth_uri": st.secrets.get("auth_uri"),
#         "token_uri": st.secrets.get("token_uri"),
#         "redirect_uri": st.secrets.get("redirect_uri")
#     }
# }
# SCOPES = ["https://www.googleapis.com/auth/gmail.compose", "https://www.googleapis.com/auth/userinfo.email", "openid"]

# def get_user_credentials():
#     creds = None
#     if os.path.exists(TOKEN_FILE):
#         try:
#             with open(TOKEN_FILE, "rb") as f:
#                 creds = pickle.load(f)
#         except Exception:
#             creds = None

#     if creds and creds.valid:
#         return creds

#     if creds and creds.expired and creds.refresh_token:
#         try:
#             creds.refresh(Request())
#             with open(TOKEN_FILE, "wb") as f:
#                 pickle.dump(creds, f)
#             return creds
#         except Exception:
#             creds = None

#     if not creds:
#         flow = Flow.from_client_config(client_config, scopes=SCOPES, redirect_uri=st.secrets.get("redirect_uri"))
#         auth_url, _ = flow.authorization_url(prompt="consent", access_type="offline")
        
#         # Capture authorization code from URL
#         code = st.query_params.get("code")
#         if code:
#             try:
#                 # Exchange code for tokens
#                 flow.fetch_token(code=code)
#                 creds = flow.credentials
#                 with open(TOKEN_FILE, "wb") as f:
#                     pickle.dump(creds, f)
#                 # CLEAR query params to prevent re-using the same code on refresh
#                 st.query_params.clear()
#                 st.rerun()
#             except Exception as e:
#                 st.error(f"Authentication Failed: {e}")
#                 st.query_params.clear()
#         else:
#             st.info("🔐 To enable the 'Out-of-Scope' email feature, authentication is required.")
#             st.link_button("Sign in with Google", auth_url)
#             st.stop() # Halt execution until user interacts with the link
    
#     return creds

# def create_gmail_draft(creds, recipient, subject, body):
#     try:
#         service = build("gmail", "v1", credentials=creds)
#         message = MIMEMultipart()
#         message["to"] = recipient
#         message["subject"] = subject
#         message.attach(MIMEText(body, "plain"))
#         raw = base64.urlsafe_b64encode(message.as_bytes()).decode()
#         service.users().drafts().create(userId="me", body={"message": {"raw": raw}}).execute()
#         st.success("✅ Email draft successfully created! Irina will review it soon.")
#         return True
#     except Exception as e:
#         st.error(f"Gmail Error: {e}")
#         return False

# # --- Assistant Logic (Alessandra) ---
# class PortfolioAssistant:
#     def __init__(self, data):
#         self.data = data

#     def get_response(self, query):
#         for q_data in self.data["portfolio_questions"]:
#             if query.lower() in q_data["question"].lower():
#                 return q_data["response"]
#         return None

# def handle_user_query(query):
#     assistant = PortfolioAssistant(portfolio_data)
#     response = assistant.get_response(query)

#     if response:
#         st.write(f"**Alessandra:** {response}")
#     else:
#         st.warning("Alessandra: This query is out-of-scope. I can generate a draft for Irina to review.")
#         email = st.text_input("Enter your email for follow-up:", key="assistant_email_input")
#         if email:
#             creds = get_user_credentials()
#             if creds:
#                 # Generate a simple AI response for the draft
#                 inputs = tokenizer(query, return_tensors="pt", truncation=True, max_length=512)
#                 outputs = model.generate(**inputs, max_length=100)
#                 ai_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                
#                 body = f"Query: {query}\n\nAI Suggested Response: {ai_text}\n\n(This is a draft for Irina to review.)"
#                 create_gmail_draft(creds, email, f"Portfolio Inquiry: {query[:30]}", body)

# # --- Home Page Interface ---
# if st.session_state.page == "Home":
#     st.title("Hi, I am Irina Swofford, and this is Alessandra, my portfolio assistant")
    
#     st.write("""
#     I specialize in both AI engineering and project management, with a strong ability to communicate complex AI/ML concepts in an understandable way for both technical and non-technical stakeholders. 
#     My goal is to turn challenges into actionable insights by focusing on problem-solving, improving operational efficiency, and staying ahead of emerging AI trends. 
#     By combining my technical expertise with strategic project management, I ensure that both AI and business objectives are successfully achieved.
#     """)

#     st.markdown("""
#     My portfolio assistant, powered by AI, helps navigate through the various sections of this portfolio.

#     ### How my portfolio AI assistant works:
#     - **In-Scope Questions:** Example: **"How do you stay organized as a project manager?"** -> Alessandra responds directly.
#     - **Out-of-Scope Questions:** Example: **"How do you build a rocket?"** -> Alessandra prompts for an email and creates a Gmail draft for me to review.
#     """)

#     query = st.text_input("Ask Alessandra anything about my projects or experience:", key="home_query_box")
#     if query:
#         handle_user_query(query)

# # --- Dynamic Page Routing ---
# def load_page(path):
#     if os.path.exists(path):
#         with open(path, "r") as f:
#             exec(compile(f.read(), path, "exec"), globals())
#     else:
#         st.error(f"Page file not found: {path}")

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
#     "Customer Chatbot Fine Tunned with ChatGPT Turbo": "pages/chatbot_fine_tuned.py"
# }

# if st.session_state.page in pages_map:
#     load_page(pages_map[st.session_state.page])
import streamlit as st
import os
import pickle
import base64
import logging
import torch

from portfolio_data import portfolio_data
from googleapiclient.discovery import build
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import Flow
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from transformers import T5Tokenizer, T5ForConditionalGeneration
from dotenv import load_dotenv

# --- CONFIG ---
os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"
os.environ["STREAMLIT_WATCH_FILE_SYSTEM"] = "false"
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
load_dotenv()

CSE_ID = os.getenv("CSE_ID")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# --- SAFE TORCH PATCH ---
try:
    if hasattr(torch, "classes") and hasattr(torch.classes, "__path__"):
        torch.classes.__path__ = []
except Exception:
    pass

# --- UI ---
st.markdown("""
<style>
div[data-testid="stSidebarNav"], div[data-testid="stSidebarHeader"] {
    display:none;
}
</style>
""", unsafe_allow_html=True)

# --- SESSION STATE ---
if "page" not in st.session_state:
    st.session_state.page = "Home"

# --- SIDEBAR ---
st.sidebar.title("AI/ML Projects & Management")

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
    key="unique_radio_key"
)

st.session_state.page = selected_page

# --- MODEL ---
@st.cache_resource
def get_t5_model():
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    model = T5ForConditionalGeneration.from_pretrained("t5-small")
    return tokenizer, model

tokenizer, model = get_t5_model()

# --- OAUTH CONFIG (IMPORTANT FIX AREA) ---
TOKEN_FILE = st.secrets.get("GOOGLE_TOKEN_PATH", "token.pkl")

# IMPORTANT: must EXACTLY match Google Cloud Console redirect URI
REDIRECT_URI = st.secrets.get("redirect_uri")

client_config = {
    "web": {
        "client_id": st.secrets.get("client_id"),
        "client_secret": st.secrets.get("client_secret"),
        "auth_uri": "https://accounts.google.com/o/oauth2/auth",
        "token_uri": "https://oauth2.googleapis.com/token",
        "redirect_uri": REDIRECT_URI,
    }
}

SCOPES = [
    "https://www.googleapis.com/auth/gmail.compose",
    "https://www.googleapis.com/auth/userinfo.email",
    "openid"
]

# --- FIXED OAUTH FLOW (NO redirect mismatch chaos) ---
def get_user_credentials():
    creds = None

    # load token
    if os.path.exists(TOKEN_FILE):
        try:
            with open(TOKEN_FILE, "rb") as f:
                creds = pickle.load(f)
        except Exception:
            creds = None

    # valid token
    if creds and creds.valid:
        return creds

    # refresh token
    if creds and creds.expired and creds.refresh_token:
        try:
            creds.refresh(Request())
            with open(TOKEN_FILE, "wb") as f:
                pickle.dump(creds, f)
            return creds
        except Exception:
            creds = None

    # NEW AUTH FLOW
    flow = Flow.from_client_config(
        client_config,
        scopes=SCOPES,
        redirect_uri=REDIRECT_URI
    )

    auth_url, _ = flow.authorization_url(
        prompt="consent",
        access_type="offline",
        include_granted_scopes="true"
    )

    # STEP 1: show login button
    st.markdown("### 🔐 Google Authentication Required")
    st.link_button("Sign in with Google", auth_url)

    # STEP 2: capture code ONLY if returned
    code = st.query_params.get("code")

    if not code:
        st.stop()

    try:
        flow.fetch_token(code=code)
        creds = flow.credentials

        with open(TOKEN_FILE, "wb") as f:
            pickle.dump(creds, f)

        # safer than clear()
        st.query_params.clear()
        st.rerun()

    except Exception as e:
        st.error(f"OAuth Error: {e}")
        return None

    return creds

# --- GMAIL ---
def create_gmail_draft(creds, recipient, subject, body):
    try:
        service = build("gmail", "v1", credentials=creds)

        message = MIMEMultipart()
        message["to"] = recipient
        message["subject"] = subject
        message.attach(MIMEText(body, "plain"))

        raw = base64.urlsafe_b64encode(message.as_bytes()).decode()

        service.users().drafts().create(
            userId="me",
            body={"message": {"raw": raw}}
        ).execute()

        st.success("✅ Gmail draft created successfully!")
        return True

    except Exception as e:
        st.error(f"Gmail Error: {e}")
        return False

# --- PORTFOLIO ASSISTANT ---
class PortfolioAssistant:
    def __init__(self, data):
        self.data = data

    def get_response(self, query):
        for item in self.data["portfolio_questions"]:
            if query.lower() in item["question"].lower():
                return item["response"]
        return None

def handle_user_query(query):
    assistant = PortfolioAssistant(portfolio_data)
    response = assistant.get_response(query)

    if response:
        st.write(f"**Alessandra:** {response}")
        return

    st.warning("Out-of-scope query detected. Email required for follow-up.")

    email = st.text_input("Enter your email:", key="email_input")

    if email:
        creds = get_user_credentials()

        if not creds:
            st.error("Google authentication required.")
            return

        inputs = tokenizer(query, return_tensors="pt", truncation=True)
        outputs = model.generate(**inputs, max_length=120)
        ai_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        body = f"""
Query: {query}

AI Suggested Response:
{ai_text}

(Review required by Irina)
"""

        create_gmail_draft(creds, email, f"Portfolio: {query[:40]}", body)

# --- HOME ---
if st.session_state.page == "Home":
    st.title("Hi, I am Irina Swofford — Portfolio Assistant")

    st.write("""
This portfolio demonstrates AI engineering, ML systems, NLP, CV, and agent-based automation.
""")

    query = st.text_input("Ask Alessandra about my work:")

    if query:
        handle_user_query(query)

# --- PAGE LOADER ---
def load_page(path):
    if os.path.exists(path):
        with open(path, "r") as f:
            exec(compile(f.read(), path, "exec"), globals())
    else:
        st.error("Page not found")

pages_map = {
    "AI Project Mangement experience": "pages/project_roadmap.py",
    "Robotic Process Automation and Natural Language Processing": "pages/analyse_workflow.py",
    "Recurrent Neural Network-Long Short Term Memory Networks": "pages/lstm.py",
    "Supervised learning": "pages/features-sales.py",
    "Unsupervised learning": "pages/segmentation.py",
    "Conversational AI": "pages/conversational_ai.py",
    "Natural Language Processing & Generative AI": "pages/survey_proofreading.py",
    "Computer Vision - Image Text Extraction": "pages/text-extraction-image.py",
    "Computer Vision - Object Detection": "pages/object_detection_size.py",
    "Sales Agent- Agentic Framework": "pages/sales_agent.py",
    "Speech AI": "pages/speach-to-text.py",
    "Text to Speech AI": "pages/text-to-speach.py",
    "Customer Chatbot": "pages/chatbot_fine_tuned.py"
}

if st.session_state.page in pages_map:
    load_page(pages_map[st.session_state.page])
