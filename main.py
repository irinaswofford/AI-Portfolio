import streamlit as st
import os
import pickle
import base64
import torch
import logging

from portfolio_data import portfolio_data
from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import Flow
from google.oauth2 import id_token
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from transformers import T5Tokenizer, T5ForConditionalGeneration
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv

# Allow insecure transport for local development
os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"

# Logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Load environment variables
load_dotenv()
CSE_ID = os.getenv("CSE_ID")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Prevent Streamlit from reloading unnecessarily
os.environ["STREAMLIT_WATCH_FILE_SYSTEM"] = "false"

# Suppress PyTorch warning
try:
    torch.classes.__path__ = []
except AttributeError:
    pass

# Hide Streamlit sidebar elements
hide_elements = """
<style>
    div[data-testid="stSidebarNav"], div[data-testid="stSidebarHeader"] {
        display:none;
    }
    .stSidebar h1 {
        padding: 4.25rem 0px 3rem;
    }
</style>
"""
st.markdown(hide_elements, unsafe_allow_html=True)

# Initialize session state
if "page" not in st.session_state:
    st.session_state.page = "Home"

st.sidebar.title("AI/ML Projects and Project Management Experience")

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
        "AI News Agent",
        "Customer Chatbot Fine Tunned with ChatGPT Turbo",
    ],
    key="unique_radio_key",
)

st.session_state.page = selected_page


# -----------------------------
# Cached T5 Model Loader
# -----------------------------
@st.cache_resource
def get_t5_model():
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    model = T5ForConditionalGeneration.from_pretrained("t5-small")
    return tokenizer, model


tokenizer, model = get_t5_model()

# -----------------------------
# LangGraph State Schema
# -----------------------------
state_schema = frozenset(
    [
        ("start", "user_query"),
        ("user_query", "response"),
        ("response", END),
        ("start", "file_upload"),
        ("file_upload", END),
    ]
)

graph = StateGraph(state_schema=state_schema)

# -----------------------------
# Google OAuth Configuration
# -----------------------------
TOKEN_FILE = st.secrets["GOOGLE_TOKEN_PATH"]

client_config = {
    "web": {
        "client_id": st.secrets["client_id"],
        "client_secret": st.secrets["client_secret"],
        "auth_uri": st.secrets["auth_uri"],
        "token_uri": st.secrets["token_uri"],
        "redirect_uri": st.secrets["redirect_uri"],
    }
}

SCOPES = [
    "https://www.googleapis.com/auth/gmail.compose",
    "https://www.googleapis.com/auth/userinfo.email",
    "openid",
]


def get_auth_code_from_url():
    try:
        query_params = st.query_params
        code_value = query_params.get("code")

        if isinstance(code_value, list):
            return code_value[0] if code_value else None
        return code_value

    except Exception as e:
        st.error(f"Error extracting code: {e}")
        return None


def get_user_credentials():
    creds = None

    # Load existing token
    if os.path.exists(TOKEN_FILE):
        try:
            with open(TOKEN_FILE, "rb") as token_file_obj:
                creds = pickle.load(token_file_obj)
        except Exception:
            try:
                os.remove(TOKEN_FILE)
            except OSError:
                pass
            creds = None

    # Valid credentials
    if creds and creds.valid:
        st.toast("Logged in successfully!", icon="✅")
        return creds

    # Refresh expired token
    if creds and creds.expired and creds.refresh_token:
        try:
            creds.refresh(Request())
            with open(TOKEN_FILE, "wb") as f:
                pickle.dump(creds, f)
            return creds
        except Exception:
            creds = None

    # New authentication
    if not creds:
        try:
            flow = Flow.from_client_config(client_config, scopes=SCOPES, redirect_uri=st.secrets["redirect_uri"])
            auth_url, _ = flow.authorization_url(prompt="consent", access_type="offline")

            st.info(
                f"### Google Authentication Required\n"
                f"[Click here to sign in]({auth_url})"
            )

            auth_code = get_auth_code_from_url()

            if auth_code:
                flow.fetch_token(code=auth_code)
                creds = flow.credentials

                with open(TOKEN_FILE, "wb") as f:
                    pickle.dump(creds, f)

                st.query_params.pop("code", None)
                st.rerun()

            else:
                st.rerun()

        except Exception as e:
            st.error(f"Error during authentication: {e}")
            creds = None

    return creds


# -----------------------------
# Gmail Draft Creation
# -----------------------------
def create_message(sender, to, subject, message_text):
    message = MIMEText(message_text)
    message["to"] = to
    message["from"] = sender
    message["subject"] = subject
    raw = base64.urlsafe_b64encode(message.as_bytes())
    return raw.decode()


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

        st.success(f"Draft created! ID: {draft['id']}")
        return f"Draft created successfully with ID: {draft['id']}"

    except Exception as e:
        st.error(f"Failed to create draft: {e}")
        return f"Failed to create draft: {e}"


# -----------------------------
# Search + AI Answer
# -----------------------------
def GoogleSearch(query):
    try:
        service = build("customsearch", "v1", developerKey=GOOGLE_API_KEY)
        results = service.cse().list(q=query, cx=CSE_ID, num=3).execute()

        if "items" not in results:
            return "No results found."

        snippets = [
            f"{item.get('title', 'No title')} - {item.get('snippet', 'No snippet')}\nURL: {item.get('link', '')}"
            for item in results["items"]
        ]

        return "\n\n".join(snippets)

    except Exception as e:
        return f"Search error: {e}"


def generate_ai_answer(query):
    try:
        inputs = tokenizer(query, return_tensors="pt", padding=True, truncation=True)
        outputs = model.generate(**inputs, max_length=150)
        return tokenizer.decode(outputs[0], skip_special_tokens=True)
    except Exception as e:
        return f"AI generation error: {e}"


# -----------------------------
# Portfolio Assistant Logic
# -----------------------------
class PortfolioAssistant:
    def __init__(self, portfolio_data):
        self.portfolio_data = portfolio_data

    def get_response(self, user_query):
        for q in self.portfolio_data["portfolio_questions"]:
            if user_query.lower() in q["question"].lower():
                return q["response"]
        return None


def handle_user_query(user_query, user_email, email_sent=False):
    assistant = PortfolioAssistant(portfolio_data)
    response = assistant.get_response(user_query)

    if response:
        return {
            "input": user_query,
            "output": f"Portfolio Response: {response}",
            "prompt_email": False,
            "email_sent": email_sent,
        }

    ai_answer = generate_ai_answer(user_query)
    search_result = GoogleSearch(user_query)
    combined = f"AI Answer:\n{ai_answer}\n\nSearch Results:\n{search_result}"

    if user_email and not email_sent:
        creds = get_user_credentials()
        if not creds:
            return {
                "input": user_query,
                "output": "Google authentication failed.",
                "prompt_email": False,
                "email_sent": False,
            }

        subject = f"Response to your query: {user_query}"
        body = f"Your query:\n{user_query}\n\n{combined}"

        email_status = create_gmail_draft(creds, user_email, subject, body)

        return {
            "input": user_query,
            "output": email_status,
            "prompt_email": False,
            "email_sent": True,
        }

    if not user_email:
        return {
            "input": user_query,
            "output": "Please provide your email for follow-up.",
            "prompt_email": True,
            "email_sent": False,
        }

    return {
        "input": user_query,
        "output": "Email already sent.",
        "prompt_email": False,
        "email_sent": True,
    }


# -----------------------------
# Page Loader
# -----------------------------
def load_page(page_name):
    with open(page_name, "r") as f:
        code = compile(f.read(), page_name, "exec")
        exec(code, globals())


# -----------------------------
# Main Page Routing
# -----------------------------
if st.session_state.page == "Home":

    def create_streamlit_interface():
        st.title("Hi, I am Irina Swofford, and this is Alessandra, my portfolio assistant")

        st.write(
            """
            I specialize in AI engineering and project management, translating complex AI/ML concepts
            into clear insights for both technical and non-technical audiences.
            """
        )

        st.markdown(
            """
            ### How my portfolio AI assistant works:
            - **In-Scope Questions:** Answered directly in the UI.
            - **Out-of-Scope Questions:**  
              Prompts for your email → Generates a Gmail draft → I review and send it.
            """
        )

        if "user_query" not in st.session_state:
            st.session_state.user_query = ""
        if "user_email" not in st.session_state:
            st.session_state.user_email = ""
        if "email_sent" not in st.session_state:
            st.session_state.email_sent = False
        if "response_data" not in st.session_state:
            st.session_state.response_data = None

        user_query = st.text_input("Ask me anything:")

        if user_query:
            st.session_state.user_query = user_query
            st.session_state.response_data = handle_user_query(
                user_query,
                st.session_state.user_email,
                st.session_state.email_sent,
            )

        if st.session_state.response_data:
            st.write(st.session_state.response_data["output"])

            if st.session_state.response_data.get("prompt_email"):
                st.session_state.user_email = st.text_input("Enter your email:")
                if st.session_state.user_email:
                    st.session_state.response_data = handle_user_query(
                        st.session_state.user_query,
                        st.session_state.user_email,
                        st.session_state.email_sent,
                    )
                    st.write(st.session_state.response_data["output"])
                    st.session_state.email_sent = st.session_state.response_data["email_sent"]

    create_streamlit_interface()

else:
    # Load other pages
    page_map = {
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
        "AI News Agent": "pages/daily_ai_news_agent.py",
        "Customer Chatbot Fine Tunned with ChatGPT Turbo": "pages/chatbot_fine_tuned.py",
    }

    if st.session_state.page in page_map:
        load_page(page_map[st.session_state.page])
