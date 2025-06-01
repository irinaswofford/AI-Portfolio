import logging
import pickle
import streamlit as st
import base64
import os
import torch
from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import Flow
from google.oauth2 import id_token
from google_auth_oauthlib import get_user_credentials
from email.mime.text import MIMEText
from transformers import T5Tokenizer, T5ForConditionalGeneration
from langgraph.graph import StateGraph, END
from portfolio_data import portfolio_data
from dotenv import load_dotenv

# --- Global Configurations ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

TOKEN_FILE = st.secrets["GOOGLE_TOKEN_PATH"]
REDIRECT_URI = "https://ai-portfolio-ftadvcasiaw55zhdgujya2.streamlit.app/oauth2callback/"
SCOPES = ["https://www.googleapis.com/auth/gmail.send"]

load_dotenv()
CSE_ID = os.getenv('CSE_ID')
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
os.environ["STREAMLIT_WATCH_FILE_SYSTEM"] = "false"

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

if 'page' not in st.session_state:
    st.session_state.page = 'Home'

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

state_schema = frozenset([
    ("start", "user_query"),
    ("user_query", "response"),
    ("response", END),
    ("start", "file_upload"),
    ("file_upload", END)
])

graph = StateGraph(state_schema=state_schema)

def handle_oauth2_redirect():
    return None

def login_callback():
    credentials = get_user_credentials(
        client_id=st.secrets.client_id,
        client_secret=st.secrets.client_secret,
        scopes=[
            'openid',
            'https://www.googleapis.com/auth/userinfo.email',
            'https://www.googleapis.com/auth/userinfo.profile',
            "https://www.googleapis.com/auth/calendar.events.readonly",
        ],
        minimum_port=9000,
        maximum_port=9001,
    )
    st.session_state.credentials = credentials


def st_redirect(url: str) -> None:
    nav_script = f"""
        <meta http-equiv="refresh" content="0; url='{url}'">
    """
    st.markdown(nav_script, unsafe_allow_html=True)

st.button(':key: Login', type='primary', on_click=login_callback)

if 'credentials' in st.session_state:
    id_info = id_token.verify_token(
        st.session_state.credentials.id_token,
        Request(),
    )
    st.json(id_info)

def authenticate_user():
    creds = None
    if os.path.exists(TOKEN_FILE):
        try:
            with open(TOKEN_FILE, 'rb') as token_file_obj:
                creds = pickle.load(token_file_obj)
        except Exception:
            try:
                os.remove(TOKEN_FILE)
            except OSError:
                pass
            creds = None

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
            except Exception:
                creds = None

        if not creds:
            try:
                REDIRECT_URI = st.secrets["redirect_uri"]
                client_config = {
                    "web": {
                        "client_id": st.secrets["client_id"],
                        "project_id": st.secrets["project_id"],
                        "auth_uri": st.secrets["auth_uri"],
                        "token_uri": st.secrets["token_uri"],
                        "auth_provider_x509_cert_url": st.secrets["auth_provider_x509_cert_url"],
                        "client_secret": st.secrets["client_secret"],
                        "redirect_uris": [REDIRECT_URI]
                    }
                }

                flow = Flow.from_client_config(
                    client_config,
                    scopes=[
                        "https://www.googleapis.com/auth/userinfo.profile",
                        "https://www.googleapis.com/auth/userinfo.email",
                        "openid"
                    ]
                )
                flow.redirect_uri = REDIRECT_URI

                auth_url, _ = flow.authorization_url(
                    prompt='consent',
                    access_type='offline',
                    include_granted_scopes='true'
                )

                st.info(f"### 🔐 Google Authentication Required:\n\nPlease click [here to sign in with Google]({auth_url})")
                st.markdown("---")

                auth_code = handle_oauth2_redirect()

                if auth_code:
                    try:
                        flow.fetch_token(code=auth_code)
                        creds = flow.credentials
                        with open(TOKEN_FILE, 'wb') as token_file_obj:
                            pickle.dump(creds, token_file_obj)
                        st.success("Authentication successful! Credentials saved.")
                        st.rerun()
                    except Exception:
                        st.error("❌ Error fetching token.")
                        creds = None
            except Exception:
                st.error("❌ Failed to set up authentication flow.")
                return None
    return creds

def send_email(creds, to_email, subject, message_text):
    service = build('gmail', 'v1', credentials=creds)
    message = {
        "raw": create_message("me", to_email, subject, message_text)
    }
    service.users().messages().send(userId="me", body=message).execute()

def create_message(sender, to, subject, message_text):
    message = MIMEText(message_text)
    message['to'] = to
    message['from'] = sender
    message['subject'] = subject
    raw = base64.urlsafe_b64encode(message.as_bytes())
    return raw.decode()

def GoogleSearch(query):
    try:
        service = build("customsearch", "v1", developerKey=GOOGLE_API_KEY)
        results = service.cse().list(q=query, cx=CSE_ID, num=3).execute()
        snippets = []
        if "items" in results:
            for item in results["items"]:
                snippets.append(
                    f"{item.get('title', 'No title')} - {item.get('snippet', 'No snippet available.')}\nURL: {item.get('link', 'No URL')}"
                )
            return "\n\n".join(snippets)
        else:
            return "No results found for your query."
    except Exception as e:
        if 'rateLimitExceeded' in str(e):
            return "Search service is temporarily unavailable due to quota limits. Please try again later."
        else:
            return f"Error performing search: {e}"

# --- AI Answer Generation ---
# Additional code for AI output could go here








# import logging
# import pickle
# import streamlit as st
# import base64
# from googleapiclient.discovery import build
# from google.oauth2.credentials import Credentials
# from google.auth.transport.requests import Request
# from email.mime.text import MIMEText
# from email.mime.multipart import MIMEMultipart
# from transformers import T5Tokenizer, T5ForConditionalGeneration
# from langgraph.graph import StateGraph, END
# from portfolio_data import portfolio_data
# from dotenv import load_dotenv


# import os


# # --- Global Configurations ---
# # Configure logging for better debugging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# # Define a token file name for persistence.
# # IMPORTANT: On Streamlit Cloud's ephemeral filesystem, this file might not persist
# # across deployments or prolonged inactivity. It's stored in /tmp/ which is writable.
# TOKEN_FILE = st.secrets["GOOGLE_TOKEN_PATH"] # FIX: Changed path to /tmp/

# # Define the exact redirect URI your Streamlit app uses.
# # This MUST EXACTLY MATCH one of the "Authorized redirect URIs" in your
# # Google Cloud Console "Web application" OAuth 2.0 Client ID settings.
# # The trailing slash is often crucial for consistency.
# REDIRECT_URI = "https://ai-portfolio-ftadvcasiaw55zhdgujya2.streamlit.app/oauth2callback/"

# # Define the required OAuth scopes
# # Use 'gmail.compose' if you only create drafts, 'gmail.send' if you actually send emails.
# SCOPES = ["https://www.googleapis.com/auth/gmail.send"] # Or "https://www.googleapis.com/auth/gmail.compose"

# # Load environment variables from .env file
# load_dotenv()

# # Get the environment variables
# # Ensure GOOGLE_API_KEY is also loaded if used in Google Search
# CSE_ID = os.getenv('CSE_ID')
# GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY') # Assuming you have this for Google Search

# os.environ["STREAMLIT_WATCH_FILE_SYSTEM"] = "false"
# from google_auth_oauthlib.flow import Flow # Already imported, moving up for clarity

# # Import torch explicitly (transformers will import it implicitly)
# import torch

# # THE FIX: Workaround for Streamlit/Torch file watcher conflict
# # This line should be placed after 'import torch'
# try:
#     torch.classes.__path__ = []
# except AttributeError:
#     # Handle cases where __path__ might not be modifiable or present
#     pass

# # Hide Streamlit sidebar elements
# hide_elements = """
# <style>
#     div[data-testid="stSidebarNav"], div[data-testid="stSidebarHeader"] {
#         display:none;
#     }
#     .stSidebar h1 {
#         padding: 4.25rem 0px 3rem;
#     }
# </style>
# """
# st.markdown(hide_elements, unsafe_allow_html=True)

# # Initialize session state for page selection
# if 'page' not in st.session_state:
#     st.session_state.page = 'Home'

# # Sidebar radio button for page selection
# st.sidebar.title("AI/ML Projects and Project Management Experience")
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
#     key="unique_radio_key",
# )
# st.session_state.page = selected_page

# # Initialize T5 Model and Tokenizer
# @st.cache_resource
# def get_t5_model():
#     tokenizer = T5Tokenizer.from_pretrained("t5-small")
#     model = T5ForConditionalGeneration.from_pretrained("t5-small")
#     return tokenizer, model
# # Only initialize if needed, potentially within a page where it's used
# tokenizer, model = get_t5_model()

# state_schema = frozenset([
#     ("start", "user_query"),
#     ("user_query", "response"),
#     ("response", END),
#     ("start", "file_upload"),
#     ("file_upload", END)
# ])

# graph = StateGraph(state_schema=state_schema)
# def handle_oauth2_redirect():
#     # query_params = st.query_params
#     # if "code" in query_params:
#     #     auth_code = query_params["code"][0]
#     #     logging.info("Authorization code received via URL parameter.")
#     #     return auth_code

#         flow = InstalledAppFlow.from_client_secrets_file(
#             'credentials.json', ['https://www.googleapis.com/auth/gmail.compose']
#         )
#       creds = flow.run_local_server(port=0)
#     return None

# from google.oauth2 import id_token
# from google_auth_oauthlib import get_user_credentials


# def login_callback():
#     credentials = get_user_credentials(
#         client_id=st.secrets.client_id,
#         client_secret=st.secrets.client_secret,
#         scopes=[
#             'openid',
#             'https://www.googleapis.com/auth/userinfo.email',
#             'https://www.googleapis.com/auth/userinfo.profile',
#             "https://www.googleapis.com/auth/calendar.events.readonly",
#         ],
#         minimum_port=9000,
#         maximum_port=9001,
#     )
#     print(f"credentials; {credentials}")
#     st.session_state.credentials = credentials


# st.button(':key: Login',
#           type='primary',
#           on_click=login_callback)

# print(f'st.session_state: {st.session_state}')

# if 'credentials' in st.session_state:
#     id_info = id_token.verify_token(
#         st.session_state.credentials.id_token,
#         requests.Request(),
#     )
#     st.json(id_info)








# def authenticate_user():
#     creds = None
#     logging.info("Attempting to authenticate user.")

#     # 1. Try loading cached credentials from token.pickle
#     if os.path.exists(TOKEN_FILE):
#         try:
#             with open(TOKEN_FILE, 'rb') as token_file_obj:
#                 creds = pickle.load(token_file_obj)
#             logging.info("Cached token loaded successfully.")
#         except Exception as e:
#             logging.error(f"Error loading token.pickle: {e}. Removing corrupted file.")
#             st.warning("Could not load cached credentials. Will re-authenticate.")
#             try:
#                 os.remove(TOKEN_FILE)
#             except OSError:
#                 logging.warning(f"Could not remove corrupted token file: {TOKEN_FILE}")
#             creds = None

#     # 2. Refresh or re-authenticate if credentials are not valid/expired
#     if not creds or not creds.valid:
#         if creds and creds.expired and creds.refresh_token:
#             logging.info("Cached credentials expired, attempting to refresh.")
#             try:
#                 creds.refresh(Request())
#                 logging.info("Credentials refreshed successfully.")
#             except Exception as e:
#                 logging.error(f"Error refreshing credentials: {e}. Forcing full re-authentication.")
#                 st.warning("Your saved credentials could not be refreshed. Please re-authenticate.")
#                 creds = None

#         if not creds:
#             logging.info("Starting new authentication flow.")
#             try:
#                 # Load client config from secrets
#                 REDIRECT_URI = st.secrets["redirect_uri"]
#                 client_config = {
#                     "web": {
#                         "client_id": st.secrets["client_id"],
#                         "project_id": st.secrets["project_id"],
#                         "auth_uri": st.secrets["auth_uri"],
#                         "token_uri": st.secrets["token_uri"],
#                         "auth_provider_x509_cert_url": st.secrets["auth_provider_x509_cert_url"],
#                         "client_secret": st.secrets["client_secret"],
#                         "redirect_uris": [REDIRECT_URI]
#                     }
#                 }

#                 # Build flow from config
#                 flow = Flow.from_client_config(
#                     client_config,
#                     scopes=[
#                         "https://www.googleapis.com/auth/userinfo.profile",
#                         "https://www.googleapis.com/auth/userinfo.email",
#                         "openid"
#                     ]
#                 )
#                 flow.redirect_uri = REDIRECT_URI

#                 # Generate authorization URL
#                 auth_url, _ = flow.authorization_url(
#                     prompt='consent',
#                     access_type='offline',
#                     include_granted_scopes='true'
#                 )

#                 st.info(f"### 🔐 Google Authentication Required:\n\nPlease click [here to sign in with Google]({auth_url})")
#                 st.markdown("---")

#                 # Get code from redirected URL
#                 auth_code = handle_oauth2_redirect()

#                 if auth_code:
#                     try:
#                         logging.info("Authorization code received, fetching token.")
#                         flow.fetch_token(code=auth_code)
#                         creds = flow.credentials

#                         # Save credentials to token file
#                         with open(TOKEN_FILE, 'wb') as token_file_obj:
#                             pickle.dump(creds, token_file_obj)
#                         st.success("Authentication successful! Credentials saved.")
#                         logging.info("Credentials successfully acquired and saved.")
#                         st.rerun()
#                     except Exception as e:
#                         st.error(f"❌ Error fetching token: {e}. Please ensure the code is correct and try again.")
#                         logging.error(f"Token fetching error: {e}", exc_info=True)
#                         creds = None
#             except Exception as e:
#                 st.error(f"❌ Failed to set up authentication flow: {e}. Please check your Streamlit secrets and Google Cloud Console settings.")
#                 logging.error(f"Authentication flow setup error: {e}", exc_info=True)
#                 return None

#     return creds


# # --- Helper Functions for Gmail ---
# def send_email(creds, to_email, subject, message_text):
#     service = build('gmail', 'v1', credentials=creds)
#     message = {
#         "raw": create_message("me", to_email, subject, message_text)
#     }
#     service.users().messages().send(userId="me", body=message).execute()

# def create_message(sender, to, subject, message_text):
#     message = MIMEText(message_text)
#     message['to'] = to
#     message['from'] = sender
#     message['subject'] = subject
#     raw = base64.urlsafe_b64encode(message.as_bytes())
#     return raw.decode()

# # --- Google Custom Search Function ---
# def GoogleSearch(query):
#     try:
#         service = build("customsearch", "v1", developerKey=GOOGLE_API_KEY)
#         results = service.cse().list(q=query, cx=CSE_ID, num=3).execute()
#         snippets = []
#         if "items" in results:
#             for item in results["items"]:
#                 snippets.append(
#                     f"{item.get('title', 'No title')} - {item.get('snippet', 'No snippet available.')}\nURL: {item.get('link', 'No URL')}"
#                 )
#             return "\n\n".join(snippets)
#         else:
#             return "No results found for your query."
#     except Exception as e:
#         if 'rateLimitExceeded' in str(e):
#             return "Search service is temporarily unavailable due to quota limits. Please try again later."
#         else:
#             return f"Error performing search: {e}"

# --- AI Answer Generation ---
def generate_ai_answer(query):
    # Ensure tokenizer and model are loaded only when needed
    tokenizer, model = get_t5_model()
    try:
        inputs = tokenizer(query, return_tensors="pt", padding=True, truncation=True)
        outputs = model.generate(**inputs, max_length=150, num_return_sequences=1)
        ai_answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return ai_answer
    except Exception as e:
        return f"Error generating AI answer: {e}"


# --- Create Gmail Draft ---
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
        logging.info(f"Draft created with ID: {draft['id']}")
        return f"Draft created successfully with ID: {draft['id']}"
    except Exception as e:
        logging.error(f"Error creating draft: {e}", exc_info=True)
        return f"❌ Failed to create draft: {e}"

# --- Handle User Query ---
def handle_user_query(user_query, user_email, email_sent=False):
    assistant = PortfolioAssistant(portfolio_data)
    response = assistant.get_response(user_query)

    if response:
        return {
            "input": user_query,
            "output": f"Portfolio Response: {response}",
            "prompt_email": False,
            "email_sent": email_sent
        }
    else:
        ai_answer = generate_ai_answer(user_query)
        search_result = GoogleSearch(user_query)
        combined_response = f"AI Answer:\n{ai_answer}\n\nRelevant Search Results:\n{search_result}"

        if user_email and not email_sent:
            creds = authenticate_user()
            if creds is None:
                return {
                    "input": user_query,
                    "output": "Failed to authenticate Google credentials. Please check your configuration.",
                    "prompt_email": False,
                    "email_sent": False
                }
            subject = f"Response to your query: {user_query}"
            body = f"Your query: {user_query}\n\n{combined_response}"
            email_status = create_gmail_draft(creds, user_email, subject, body)

            if "Error" not in email_status:
                st.success(email_status)
            else:
                st.error(email_status)

            return {
                "input": user_query,
                "output": email_status,
                "prompt_email": False,
                "email_sent": True
            }
        elif not user_email:
            return {
                "input": user_query,
                "output": "Your query has been received. Please provide your email for follow-up.",
                "prompt_email": True,
                "email_sent": email_sent
            }
        else:
            return {
                "input": user_query,
                "output": "The email has already been sent. Thank you!",
                "prompt_email": False,
                "email_sent": email_sent
            }

# --- Portfolio Assistant Class ---
class PortfolioAssistant:
    def __init__(self, portfolio_data):
        self.portfolio_data = portfolio_data

    def get_response(self, user_query):
        for question_data in self.portfolio_data["portfolio_questions"]:
            if user_query.lower() in question_data["question"].lower():
                return question_data["response"]
        return None

# --- Page Loading Function (Simplified) ---
# Removed authentication logic from here as it's handled by authenticate_user()
def load_page(page_name):
    with open(page_name, "r") as f:
        code = compile(f.read(), page_name, 'exec')
        exec(code, globals())

# --- Main app pages ---
if st.session_state.page == "Home":
    def create_streamlit_interface():
        st.title("Hi, I am Irina Swofford, and this is Alessandra, my portfolio assistant")

        st.write("""
        I specialize in both AI engineering and project management, with a strong ability to communicate complex AI/ML concepts in an understandable way for both technical and non-technical stakeholders.
        My goal is to turn challenges into actionable insights by focusing on problem-solving, improving operational efficiency, and staying ahead of emerging AI trends.
        By combining my technical expertise with strategic project management, I ensure that both AI and business objectives are successfully achieved.
        """)

        st.markdown("""
        My portfolio assistant, powered by AI, helps navigate through the various sections of this portfolio.
        """)

        st.markdown("""
        ### How my portfolio AI assistant works:

        - **In-Scope Questions:**
          Example: If you ask me questions related to my portfolio, like **"How do you stay organized as a project manager?"**,
          the AI directly responds with an answer displayed in the UI.

        - **Out-of-Scope Questions:**
          Example: **"How do you build a rocket?"**
          - Prompts the user for an email.
          - Captures the email and generates a Gmail draft with the AI's response and Google search results.
          - I review the draft and send you an email. (Human-in-the-Loop)
        """)

        # Initialize session state for user interaction
        if "user_query" not in st.session_state:
            st.session_state.user_query = ""
        if "user_email" not in st.session_state:
            st.session_state.user_email = ""
        if "email_sent" not in st.session_state:
            st.session_state.email_sent = False
        if "response_data" not in st.session_state:
            st.session_state.response_data = None

        user_query = st.text_input("Ask me anything about skills, my management experience, or my AI projects:")

        if user_query:
            st.session_state.user_query = user_query
            st.session_state.response_data = handle_user_query(
                user_query,
                st.session_state.user_email,
                st.session_state.email_sent
            )

        if st.session_state.response_data:
            st.write(st.session_state.response_data["output"])

            if st.session_state.response_data.get("prompt_email"):
                st.session_state.user_email = st.text_input("Enter your email for follow-up:")

                if st.session_state.user_email:
                    st.session_state.response_data = handle_user_query(
                        st.session_state.user_query,
                        st.session_state.user_email,
                        st.session_state.email_sent
                    )
                    st.write(st.session_state.response_data["output"])
                    st.session_state.email_sent = st.session_state.response_data["email_sent"]

    create_streamlit_interface()

elif st.session_state.page == "AI Project Mangement experience":
    load_page("pages/project_roadmap.py")
elif st.session_state.page == "Robotic Process Automation and Natural Language Processing":
    load_page("pages/analyse_workflow.py")
elif st.session_state.page == "Recurrent Neural Network-Long Short Term Memory Networks":
    load_page("pages/lstm.py")
elif st.session_state.page == "Supervised learning":
    load_page("pages/features-sales.py")
elif st.session_state.page == "Unsupervised learning":
    load_page("pages/unfeatures-sales.py")
elif st.session_state.page == "Conversational AI fine-tuned with Retrieval Augmented Generation":
    load_page("pages/graph.py")
elif st.session_state.page == "Natural Language Processing & Generative AI":
    load_page("pages/llama.py")
elif st.session_state.page == "Computer Vision - Image Text Extraction":
    load_page("pages/computer-vision.py")
elif st.session_state.page == "Computer Vision - Object Detection":
    load_page("pages/computer-vision-detect.py")
elif st.session_state.page == "Sales Agent- Agentic Framework":
    load_page("pages/sales_agent.py")
elif st.session_state.page == "NLP and Generative AI: Speech-to-Text AI Voice Agent":
    load_page("pages/speech2text.py")
elif st.session_state.page == "Computer Vision, Generative AI: A Text-to-Speech, Audio, and Video Generator":
    load_page("pages/tts-video.py")
elif st.session_state.page == "Customer Chatbot Fine Tunned with ChatGPT Turbo":
    load_page("pages/chatbot.py")
