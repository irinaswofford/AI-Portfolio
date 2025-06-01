import streamlit as st
import os
import pickle
import base64
import torch
from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import Flow
from google.oauth2 import id_token
from email.mime.text import MIMEText
from transformers import T5Tokenizer, T5ForConditionalGeneration
from langgraph.graph import StateGraph, END # Assuming langgraph.graph provides END
from dotenv import load_dotenv
from email.mime.multipart import MIMEMultipart
import logging

# Ensure OAUTHLIB_INSECURE_TRANSPORT is set for local development if not using HTTPS
# For Streamlit Cloud, this is usually not needed as it uses HTTPS.
os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"

# --- Global Configurations ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

load_dotenv()
CSE_ID = os.getenv('CSE_ID')
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
os.environ["STREAMLIT_WATCH_FILE_SYSTEM"] = "false" # Prevent unnecessary reloads

# Suppress a specific PyTorch warning if torch.classes.__path__ is not writable
try:
    torch.classes.__path__ = []
except AttributeError:
    pass

# Hide Streamlit sidebar elements for a cleaner UI
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

# Initialize session state for page navigation
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
    """
    Loads the T5 tokenizer and model.
    Cached to prevent reloading on every rerun.
    """
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    model = T5ForConditionalGeneration.from_pretrained("t5-small")
    return tokenizer, model

tokenizer, model = get_t5_model()

# Define the state schema for the graph (used by langgraph)
state_schema = frozenset([
    ("start", "user_query"),
    ("user_query", "response"),
    ("response", END),
    ("start", "file_upload"),
    ("file_upload", END)
])

graph = StateGraph(state_schema=state_schema)

# Define the path for the token file, retrieved from Streamlit secrets
TOKEN_FILE = st.secrets["GOOGLE_TOKEN_PATH"]

# === Get Google Client Info from Streamlit Secrets ===
# This dictionary holds the OAuth 2.0 client configuration
client_config = {
    "web": {
        "client_id": st.secrets["client_id"],
        "client_secret": st.secrets["client_secret"],
        "auth_uri": st.secrets["auth_uri"],
        "token_uri": st.secrets["token_uri"],
        "redirect_uri": st.secrets["redirect_uri"]
    }
}

# Define the OAuth 2.0 scopes (permissions) your application requests
SCOPES = ["https://www.googleapis.com/auth/gmail.compose", "https://www.googleapis.com/auth/userinfo.email", "openid"]

def get_auth_code_from_url():
    """
    Extracts the 'code' query parameter from the current URL.
    This code is provided by Google after user consent.
    """
    try:
        query_params = st.query_params
        code = query_params.get("code", [None])[0]
        st.write(f"📦 Query code: {code}") # Debugging: Show the extracted code
        return code
    except Exception as e:
        st.error(f"❌ Error extracting code from query params: {e}")
        return None

def get_user_credentials():
    """
    Handles the Google authentication flow:
    1. Loads existing credentials from file.
    2. Refreshes expired credentials.
    3. Initiates new authentication if no valid credentials exist.
    """
    creds = None

    # Load existing token if available
    if os.path.exists(TOKEN_FILE):
        try:
            with open(TOKEN_FILE, 'rb') as token_file_obj:
                creds = pickle.load(token_file_obj)
            logging.info("Existing token loaded.")
        except Exception as e:
            logging.warning(f"Failed to load existing token: {e}. Attempting to remove corrupted file.")
            try:
                os.remove(TOKEN_FILE)
            except OSError:
                pass # Ignore if file doesn't exist to remove
            creds = None

    # If credentials exist and are valid, return them immediately
    if creds and creds.valid:
        logging.info("Credentials are valid.")
        st.toast("🎉 Logged in successfully with Google!", icon="✅")
        return creds

    # If credentials exist but are expired and have a refresh token, try to refresh them
    if creds and creds.expired and creds.refresh_token:
        try:
            logging.info("Attempting to refresh token.")
            creds.refresh(Request()) # Attempt to refresh the token
            with open(TOKEN_FILE, 'wb') as f:
                pickle.dump(creds, f) # Save the refreshed token
            st.success("🔁 Token refreshed")
            logging.info("Token refreshed successfully.")
            # After refresh, if valid, return them
            if creds.valid:
                return creds
        except Exception as e:
            st.error(f"⚠️ Failed to refresh token: {e}")
            logging.error(f"Failed to refresh token: {e}", exc_info=True)
            creds = None # Invalidate creds if refresh fails

    # If still no valid creds after load/refresh, initiate new authentication
    if not creds:
        logging.info("No valid credentials found. Initiating new authentication flow.")
        try:
            flow = Flow.from_client_config(
                client_config,
                scopes=SCOPES,
                redirect_uri=st.secrets["redirect_uri"]
            )

            auth_url, _ = flow.authorization_url(
                prompt='consent',
                access_type='offline', # Request refresh token
                include_granted_scopes='true'
            )

            st.info(f"### 🔐 Google Authentication Required:\n\nPlease click [here to sign in with Google]({auth_url})")
            st.markdown("---")

            auth_code = get_auth_code_from_url()

            # --- DEBUGGING STATEMENTS ---
            
            st.write(f"DEBUG: Redirect URI being used by Flow for token exchange: {st.secrets['redirect_uri']}")
            # --- END DEBUGGING STATEMENTS ---

            if auth_code:
                try:
                    logging.info(f"Attempting to fetch token with code: {auth_code[:10]}...") # Log first 10 chars
                    flow.fetch_token(code=auth_code) # Use 'code' parameter for fetch_token
                    creds = flow.credentials
                    with open(TOKEN_FILE, 'wb') as f:
                        pickle.dump(creds, f)
                    st.success("✅ Authentication successful! Credentials saved.")
                    logging.info("Authentication successful. Rerunning app.")

                    # --- CRITICAL ADDITION: Clear the 'code' from query parameters ---
                    # This prevents the app from trying to reuse the same code on rerun
                    # and helps with the "Malformed auth code" error.
                    current_query_params = st.query_params.to_dict()
                    if "code" in current_query_params:
                        del current_query_params["code"]
                        # To apply changes, you need to set st.query_params
                        # Streamlit 1.31+ allows direct assignment to st.query_params
                        st.query_params.clear() # Clear all existing
                        for k, v in current_query_params.items():
                            # Handle cases where query param value might be a list (e.g., from .get())
                            st.query_params[k] = v[0] if isinstance(v, list) else v
                    # --- END CRITICAL ADDITION ---

                    st.experimental_rerun() # Rerun the app to reflect the logged-in state
                except Exception as e:
                    st.error(f"❌ Failed to fetch token: {e}")
                    st.error(f"Full Exception (from get_user_credentials): {e}") # Provide full exception for more detail
                    logging.error(f"Failed to fetch token: {e}", exc_info=True)
                    creds = None # Ensure creds is None if token fetch fails, allowing re-prompt
            else:
                logging.info("No auth code found in URL. Waiting for user interaction.")

        except Exception as e:
            st.error(f"❌ Error initiating auth flow: {e}")
            logging.error(f"Error initiating auth flow: {e}", exc_info=True)
            creds = None

    return creds

def send_email(creds, to_email, subject, message_text):
    """Sends an email using the Gmail API."""
    service = build('gmail', 'v1', credentials=creds)
    message = {
        "raw": create_message("me", to_email, subject, message_text)
    }
    # Using drafts instead of direct send for human-in-the-loop
    # service.users().messages().send(userId="me", body=message).execute()
    # This function is not used in the current flow where create_gmail_draft is used.
    pass

def create_message(sender, to, subject, message_text):
    """Creates a MIMEText message for Gmail API."""
    message = MIMEText(message_text)
    message['to'] = to
    message['from'] = sender
    message['subject'] = subject
    raw = base64.urlsafe_b64encode(message.as_bytes())
    return raw.decode()

def GoogleSearch(query):
    """Performs a Google Custom Search and returns snippets."""
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

def generate_ai_answer(query):
    """Generates an AI answer using the T5 model."""
    tokenizer, model = get_t5_model() # Ensure tokenizer and model are loaded
    try:
        inputs = tokenizer(query, return_tensors="pt", padding=True, truncation=True)
        outputs = model.generate(**inputs, max_length=150, num_return_sequences=1)
        ai_answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return ai_answer
    except Exception as e:
        return f"Error generating AI answer: {e}"

def create_gmail_draft(creds, recipient, subject, body):
    """Creates a Gmail draft with the given content."""
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

# Placeholder for portfolio_data and PortfolioAssistant class
# You would need to ensure 'portfolio_data' is defined and accessible
# and 'PortfolioAssistant' class is implemented.
# For this example, I'll provide a minimal placeholder for PortfolioAssistant.
portfolio_data = {
    "portfolio_questions": [
        {"question": "how do you stay organized as a project manager?", "response": "I use agile methodologies and various project management tools to keep track of tasks and progress."},
        {"question": "what is your experience with AI projects?", "response": "I have extensive experience in leading AI projects from conception to deployment, focusing on practical applications and measurable results."},
    ]
}

class PortfolioAssistant:
    def __init__(self, portfolio_data):
        self.portfolio_data = portfolio_data

    def get_response(self, user_query):
        for question_data in self.portfolio_data["portfolio_questions"]:
            if user_query.lower() in question_data["question"].lower():
                return question_data["response"]
        return None

def handle_user_query(user_query, user_email, email_sent=False):
    """
    Handles user queries, providing direct answers for in-scope questions
    or generating an email draft for out-of-scope questions.
    """
    assistant = PortfolioAssistant(portfolio_data)
    response = assistant.get_response(user_query)

    if response:
        # In-scope question
        return {
            "input": user_query,
            "output": f"Portfolio Response: {response}",
            "prompt_email": False,
            "email_sent": email_sent
        }
    else:
        # Out-of-scope question
        ai_answer = generate_ai_answer(user_query)
        search_result = GoogleSearch(user_query)
        combined_response = f"AI Answer:\n{ai_answer}\n\nRelevant Search Results:\n{search_result}"

        if user_email and not email_sent:
            creds = get_user_credentials() # Attempt to get credentials
            if creds is None or not creds.valid:
                return {
                    "input": user_query,
                    "output": "Failed to authenticate Google credentials. Please sign in with Google to enable email functionality.",
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
            # Prompt for email if it's an out-of-scope question and no email provided yet
            return {
                "input": user_query,
                "output": "Your query has been received. Please provide your email for follow-up.",
                "prompt_email": True,
                "email_sent": email_sent
            }
        else:
            # If email already sent for this query
            return {
                "input": user_query,
                "output": "The email has already been sent. Thank you!",
                "prompt_email": False,
                "email_sent": email_sent
            }

# --- Page Loading Function (Simplified) ---
# This function is used to load content from other Python files (pages)
def load_page(page_name):
    # This is a simplified approach. In a real app, you might use
    # st.set_page_config or a more robust page routing mechanism.
    # For now, it executes the content of the specified page file.
    with open(page_name, "r") as f:
        # It's generally not recommended to exec arbitrary code from files
        # in a production Streamlit app due to security concerns.
        # For a portfolio, this might be acceptable if content is controlled.
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

        # Initialize session state variables for user interaction
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
            # Call handle_user_query to process the input
            st.session_state.response_data = handle_user_query(
                user_query,
                st.session_state.user_email,
                st.session_state.email_sent
            )

        if st.session_state.response_data:
            st.write(st.session_state.response_data["output"])

            # If the response indicates an email is needed, prompt for it
            if st.session_state.response_data.get("prompt_email"):
                st.session_state.user_email = st.text_input("Enter your email for follow-up:")

                if st.session_state.user_email:
                    # Re-handle the query with the provided email
                    st.session_state.response_data = handle_user_query(
                        st.session_state.user_query,
                        st.session_state.user_email,
                        st.session_state.email_sent
                    )
                    st.write(st.session_state.response_data["output"])
                    st.session_state.email_sent = st.session_state.response_data["email_sent"]

    create_streamlit_interface()

# --- Page routing based on sidebar selection ---
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
