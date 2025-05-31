import logging
import pickle
import streamlit as st
import base64
from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from transformers import T5Tokenizer, T5ForConditionalGeneration
from langgraph.graph import StateGraph, END
from portfolio_data import portfolio_data 
from dotenv import load_dotenv
import os
from google_auth_oauthlib.flow import InstalledAppFlow

# Import torch explicitly (transformers will import it implicitly)
import torch

# THE FIX: Workaround for Streamlit/Torch file watcher conflict
# This line should be placed after 'import torch'
try:
    torch.classes.__path__ = []
except AttributeError:
    # Handle cases where __path__ might not be modifiable or present
    # (though it usually is the problem in these errors)
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

# Initialize session state for page selection
if 'page' not in st.session_state:
    st.session_state.page = 'Home'

# Sidebar radio button for page selection
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

# Function to load and execute a page script
def load_page(page_name):
    with open(page_name, "r") as f:
        code = compile(f.read(), page_name, 'exec')
        exec(code, globals())

# Initialize T5 Model and Tokenizer
# tokenizer = T5Tokenizer.from_pretrained("t5-small")
# model = T5ForConditionalGeneration.from_pretrained("t5-small")



# Initialize T5 Model and Tokenizer
# It's good practice to cache these heavy objects if they don't change
@st.cache_resource
def get_t5_model():
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    model = T5ForConditionalGeneration.from_pretrained("t5-small")
    return tokenizer, model

state_schema = frozenset([
    ("start", "user_query"),
    ("user_query", "response"),
    ("response", END),
    ("start", "file_upload"),
    ("file_upload", END)
])
graph = StateGraph(state_schema=state_schema)

# Load environment variables from .env file
load_dotenv()

# Get the environment variables
CSE_ID = os.getenv('CSE_ID')
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

# Google Custom Search Function
def google_search(query):
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

# AI Answer Generation Function
def generate_ai_answer(query):
    try:
        inputs = tokenizer(query, return_tensors="pt", padding=True, truncation=True)
        outputs = model.generate(**inputs, max_length=150, num_return_sequences=1)
        ai_answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return ai_answer
    except Exception as e:
        return f"Error generating AI answer: {e}"

# Load Gmail API Credentials with error handling and Streamlit secrets check
def load_credentials():
    try:
        token_file = "token.pickle"
        scopes = ['https://www.googleapis.com/auth/gmail.compose']

        # Debug: print loaded secrets (comment out in production)
        # st.write("Loaded client_id:", st.secrets.get("client_id"))
        # st.write("Loaded client_secret:", st.secrets.get("client_secret"))

        # Load cached token if available and valid
        if os.path.exists(token_file):
            with open(token_file, 'rb') as token:
                creds = pickle.load(token)
            if creds and creds.valid:
                return creds
            elif creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
                with open(token_file, 'wb') as token:
                    pickle.dump(creds, token)
                return creds

        # Prepare client secrets dict from Streamlit secrets
        client_secrets_dict = {
            "installed": {
                "client_id": st.secrets["client_id"],
                "project_id": st.secrets["project_id"],
                "auth_uri": st.secrets["auth_uri"],
                "token_uri": st.secrets["token_uri"],
                "auth_provider_x509_cert_url": st.secrets["auth_provider_x509_cert_url"],
                "client_secret": st.secrets["client_secret"]
            }
        }

        flow = InstalledAppFlow.from_client_config(client_secrets_dict, scopes=scopes)
        creds = flow.run_local_server(port=0)

        with open(token_file, 'wb') as token:
            pickle.dump(creds, token)

        return creds

    except Exception as e:
        st.error(f"Failed to load credentials: {e}")
        return None

# Create Gmail Draft
def create_gmail_draft(creds, recipient, subject, body):
    try:
        service = build("gmail", "v1", credentials=creds)
        message = MIMEMultipart()
        message["to"] = recipient
        message["subject"] = subject
        message.attach(MIMEText(body, "plain"))

        encoded_message = {"raw": base64.urlsafe_b64encode(message.as_bytes()).decode("utf-8")}
        draft = service.users().drafts().create(userId="me", body={"message": encoded_message}).execute()
        return "Draft created successfully! Please check your Gmail drafts."
    except Exception as e:
        return f"Error creating draft: {e}"

# Handle User Query
def handle_user_query(user_query, user_email, email_sent=False):
    assistant = PortfolioAssistant(portfolio_data)
    response = assistant.get_response(user_query)

    if response:
        # In-scope query: Show only portfolio response
        return {
            "input": user_query,
            "output": f"Portfolio Response: {response}",
            "prompt_email": False,
            "email_sent": email_sent
        }
    else:
        # Out-of-scope query: Generate AI Answer and include search results
        ai_answer = generate_ai_answer(user_query)
        search_result = google_search(user_query)
        combined_response = f"AI Answer:\n{ai_answer}\n\nRelevant Search Results:\n{search_result}"

        if user_email and not email_sent:
            creds = load_credentials()
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

# Portfolio Assistant Class
class PortfolioAssistant:
    def __init__(self, portfolio_data):
        self.portfolio_data = portfolio_data

    def get_response(self, user_query):
        for question_data in self.portfolio_data["portfolio_questions"]:
            if user_query.lower() in question_data["question"].lower():
                return question_data["response"]
        return None

# Main app pages
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

        if st.session_state.response_data and not st.session_state.response_data["prompt_email"]:
            if st.button("Ask another question"):
                st.session_state.user_query = ""
                st.session_state.user_email = ""
                st.session_state.email_sent = False
                st.session_state.response_data = None

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
# from google_auth_oauthlib.flow import InstalledAppFlow

# # import sys
# # from pathlib import Path
# # Add AI-Sales-agent-main to the Python path
# # agent_path = Path(__file__).parent/ "AI-Sales-agent-main"
# # print('ssdd',agent_path)
# # sys.path.append(str(agent_path))


# # Inject custom CSS to hide the specific element
# hide_elements = """
# <style>
#     div[data-testid="stSidebarNav"] ,div[data-testid="stSidebarHeader"]{
#         display:none;
#     }
#     .stSidebar h1{
#         padding: 4.25rem 0px 3rem;
        
#     }
# </style>
# """
# st.markdown(hide_elements, unsafe_allow_html=True)

# # Continue with the rest of your Streamlit app...

# # Initialize session state for page selection
# if 'page' not in st.session_state:
#     st.session_state.page = 'Home'

# # Sidebar radio button for page selection

# st.sidebar.title("AI/ML Projects and Project Management Experience")
# selected_page = st.sidebar.radio("Choose an option:", ["Home", "AI Project Mangement experience", "Robotic Process Automation and Natural Language Processing", "Recurrent Neural Network-Long Short Term Memory Networks", "Supervised learning",
#                                                         "Unsupervised learning",
#                                                         "Conversational AI fine-tuned with Retrieval Augmented Generation",
#                                                         "Natural Language Processing & Generative AI" , 
#                                                         "Computer Vision - Image Text Extraction",   
#                                                         "Computer Vision - Object Detection",
#                                                         "Sales Agent- Agentic Framework",
#                                                         "NLP and Generative AI: Speech-to-Text AI Voice Agent", "Computer Vision, Generative AI: A Text-to-Speech, Audio, and Video Generator",
#                                                         "Customer Chatbot Fine Tunned with ChatGPT Turbo"], key="unique_radio_key",)

# # Update session state based on selection
# st.session_state.page = selected_page          

# # Function to load and execute a page script
# def load_page(page_name):
#     with open(page_name, "r") as f:
#         code = compile(f.read(), page_name, 'exec')
#         exec(code, globals())
# #end side bar 

# # Main Content Area 

# # Initialize T5 Model and Tokenizer
# tokenizer = T5Tokenizer.from_pretrained("t5-small")
# model = T5ForConditionalGeneration.from_pretrained("t5-small")

# state_schema = frozenset([
#             ("start", "user_query"),
#             ("user_query", "response"),
#             ("response", END),
#             ("start", "file_upload"),
#             ("file_upload", END)
#         ])
# graph = StateGraph(state_schema=state_schema)

 
# # Load environment variables from .env file
# load_dotenv()

# # Get the environment variables
# CSE_ID = os.getenv('CSE_ID')
# GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')


# # Google Custom Search Function
# def google_search(query):
#             try:
#                 service = build("customsearch", "v1", developerKey=GOOGLE_API_KEY)
#                 results = service.cse().list(q=query, cx=CSE_ID, num=3).execute()
#                 snippets = []
#                 if "items" in results:
#                     for item in results["items"]:
#                         snippets.append(
#                             f"{item.get('title', 'No title')} - {item.get('snippet', 'No snippet available.')}\nURL: {item.get('link', 'No URL')}"
#                         )
#                     return "\n\n".join(snippets)
#                 else:
#                     return "No results found for your query."
#             except Exception as e:
#                 if 'rateLimitExceeded' in str(e):
#                     return "Search service is temporarily unavailable due to quota limits. Please try again later."
#                 else:
#                     return f"Error performing search: {e}"

#         # AI Answer Generation Function
# def generate_ai_answer(query):
#             try:
#                 inputs = tokenizer(query, return_tensors="pt", padding=True, truncation=True)
#                 outputs = model.generate(**inputs, max_length=150, num_return_sequences=1)
#                 ai_answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
#                 return ai_answer
#             except Exception as e:
#                 return f"Error generating AI answer: {e}"

#         # Load Gmail API Credentials
# # def load_credentials():
# #     """Handles OAuth2 authentication and returns credentials."""
# #     creds = None
# #     token_file = 'token.pickle'

# #     # Check if the token.pickle file exists
# #     if os.path.exists(token_file):
# #         with open(token_file, 'rb') as token:
# #             creds = pickle.load(token)
# #         if creds and creds.valid:
# #             return creds
# #         elif creds and creds.expired and creds.refresh_token:
# #             creds.refresh(Request())
# #             with open(token_file, 'wb') as token:
# #                 pickle.dump(creds, token)
# #             return creds
# #     else:
# #         # If no valid credentials are found, perform OAuth
# #         flow = InstalledAppFlow.from_client_secrets_file(
# #             'credentials.json', ['https://www.googleapis.com/auth/gmail.compose']
# #         )
# #         creds = flow.run_local_server(port=0)
        
# #         # Save the credentials for the next run
# #         with open(token_file, 'wb') as token:
# #             pickle.dump(creds, token)

# #         return creds

# def load_credentials():
#     token_file = "token.pickle"
#     scopes = ['https://www.googleapis.com/auth/gmail.compose']

#     # Try loading cached credentials first
#     if os.path.exists(token_file):
#         with open(token_file, 'rb') as token:
#             creds = pickle.load(token)
#         if creds and creds.valid:
#             return creds
#         elif creds and creds.expired and creds.refresh_token:
#             creds.refresh(Request())
#             with open(token_file, 'wb') as token:
#                 pickle.dump(creds, token)
#             return creds

#     # No valid credentials found, create flow from secrets dict
#     client_secrets_dict = {
#         "installed": {
#             "client_id": st.secrets["client_id"],
#             "project_id": st.secrets["project_id"],
#             "auth_uri": st.secrets["auth_uri"],
#             "token_uri": st.secrets["token_uri"],
#             "auth_provider_x509_cert_url": st.secrets["auth_provider_x509_cert_url"],
#             "client_secret": st.secrets["client_secret"]
#         }
#     }

#     flow = InstalledAppFlow.from_client_config(client_secrets_dict, scopes=scopes)
#     creds = flow.run_local_server(port=0)

#     # Save credentials for next time
#     with open(token_file, 'wb') as token:
#         pickle.dump(creds, token)

#     return creds

#         # Create Gmail Draft
# def create_gmail_draft(creds, recipient, subject, body):
#             try:
#                 service = build("gmail", "v1", credentials=creds)
#                 message = MIMEMultipart()
#                 message["to"] = recipient
#                 message["subject"] = subject
#                 message.attach(MIMEText(body, "plain"))

#                 encoded_message = {"raw": base64.urlsafe_b64encode(message.as_bytes()).decode("utf-8")}
#                 draft = service.users().drafts().create(userId="me", body={"message": encoded_message}).execute()
#             except Exception as e:
#                 return f"Error creating draft: {e}"

#         # Handle User Query
# def handle_user_query(user_query, user_email, email_sent=False):
#             assistant = PortfolioAssistant(portfolio_data)
#             response = assistant.get_response(user_query)

#             if response:
#                 # In-scope query: Show only portfolio response
#                 return {
#                     "input": user_query,
#                     "output": f"Portfolio Response: {response}",
#                     "prompt_email": False,
#                     "email_sent": email_sent
#                 }
#             else:
#                 # Out-of-scope query: Generate AI Answer and include search results
#                 ai_answer = generate_ai_answer(user_query)
#                 search_result = google_search(user_query)
#                 combined_response = f"Relevant Search Results:\n{search_result}"

#                 if user_email and not email_sent:
#                     creds = load_credentials()
#                     subject = f"Response to your query: {user_query}"
#                     body = f"Your query: {user_query}\n\n{combined_response}"
#                     email_status = create_gmail_draft(creds, user_email, subject, body)
                    
#                     # Display success message using st.success
#                     st.success("Email sent successfully!")
            
#                     return {
#                         "input": user_query,
#                         "output": email_status,
#                         "prompt_email": False,
#                         "email_sent": True
#                     }
#                 elif not user_email:
#                     return {
#                         "input": user_query,
#                         "output": "Your query has been received. Please provide your email for follow-up.",
#                         "prompt_email": True,
#                         "email_sent": email_sent
#                     }
#                 else:
#                     return {
#                         "input": user_query,
#                         "output": "The email has already been sent. Thank you!",
#                         "prompt_email": False,
#                         "email_sent": email_sent
#                     }

#         # Portfolio Assistant Class
# class PortfolioAssistant:
#             def __init__(self, portfolio_data):
#                 self.portfolio_data = portfolio_data

#             def get_response(self, user_query):
#                 for question_data in self.portfolio_data["portfolio_questions"]:
#                     if user_query.lower() in question_data["question"].lower():
#                         return question_data["response"]

# # if st.session_state.page == "Home":
# #         # Streamlit Interface
# #     def create_streamlit_interface():

# #         st.title("Hi, I am Irina Swofford, and this is Alessandra, my portfolio assistant")
# #         st.write("""
# #         I specialize in both AI engineering and project management, with a strong ability to communicate complex AI/ML concepts in an understandable way for both technical and non-technical stakeholders. 
# #         My goal is to turn challenges into actionable insights by focusing on problem-solving, improving operational efficiency, and staying ahead of emerging AI trends.
# #         By combining my technical expertise with strategic project management, I ensure that both AI and business objectives are successfully achieved.
# #         """)
# #         st.markdown("""
# #         My portfolio assistant, powered by AI, helps navigate through the various sections of this portfolio.
# #         """)
        
# #         st.markdown("""
# #         ### How my portfolio AI assistant works:

# #         **In-Scope Questions:**  
# #         Example: If you ask me questions related to my portfolio, like **"How do you stay organized as a project manager?"**,  
# #         the AI directly responds with an answer displayed in the UI.

# #         **Out-of-Scope Questions:**  
# #         Example: **"How do you build a rocket?"**  
# #         - Prompts the user for an email.  
# #         - Captures the email and generates a Gmail draft with the AI's response and Google search results.  
# #         - I review the draft and send you an email. (Human-in-the-Loop)
# #         """)
        
# #         user_query = st.text_input("Ask me anything about skills, my management experience, or my AI projects:")

# #         email_sent = False
# #         user_email = None

# #         if user_query:
# #             response_data = handle_user_query(user_query, user_email, email_sent)
# #             st.write(response_data["output"])

# #             if response_data.get("prompt_email"):
# #                 user_email = st.text_input("Enter your email for follow-up:")
# #                 if user_email:
# #                     response_data = handle_user_query(user_query, user_email, email_sent)
# #                     st.write(response_data["output"])
# #                     email_sent = response_data["email_sent"]
# #                     # Main
# #                     # Sample data (sales pipeline data)
# #     if __name__ == "__main__":
# #         create_streamlit_interface()
# if st.session_state.page == "Home":
#     def create_streamlit_interface():
#         st.title("Hi, I am Irina Swofford, and this is Alessandra, my portfolio assistant")

#         st.write("""
#         I specialize in both AI engineering and project management, with a strong ability to communicate complex AI/ML concepts in an understandable way for both technical and non-technical stakeholders. 
#         My goal is to turn challenges into actionable insights by focusing on problem-solving, improving operational efficiency, and staying ahead of emerging AI trends.
#         By combining my technical expertise with strategic project management, I ensure that both AI and business objectives are successfully achieved.
#         """)

#         st.markdown("""
#         My portfolio assistant, powered by AI, helps navigate through the various sections of this portfolio.
#         """)

#         st.markdown("""
#         ### How my portfolio AI assistant works:

#         - **In-Scope Questions:**  
#           Example: If you ask me questions related to my portfolio, like **"How do you stay organized as a project manager?"**,  
#           the AI directly responds with an answer displayed in the UI.

#         - **Out-of-Scope Questions:**  
#           Example: **"How do you build a rocket?"**  
#           - Prompts the user for an email.  
#           - Captures the email and generates a Gmail draft with the AI's response and Google search results.  
#           - I review the draft and send you an email. (Human-in-the-Loop)
#         """)

#         # Initialize session state for user interaction
#         if "user_query" not in st.session_state:
#             st.session_state.user_query = ""
#         if "user_email" not in st.session_state:
#             st.session_state.user_email = ""
#         if "email_sent" not in st.session_state:
#             st.session_state.email_sent = False
#         if "response_data" not in st.session_state:
#             st.session_state.response_data = None

#         # Input from user
#         user_query = st.text_input("Ask me anything about skills, my management experience, or my AI projects:")

#         if user_query:
#             st.session_state.user_query = user_query
#             st.session_state.response_data = handle_user_query(
#                 user_query,
#                 st.session_state.user_email,
#                 st.session_state.email_sent
#             )

#         # Handle response
#         if st.session_state.response_data:
#             st.write(st.session_state.response_data["output"])

#             if st.session_state.response_data.get("prompt_email"):
#                 st.session_state.user_email = st.text_input("Enter your email for follow-up:")

#                 if st.session_state.user_email:
#                     st.session_state.response_data = handle_user_query(
#                         st.session_state.user_query,
#                         st.session_state.user_email,
#                         st.session_state.email_sent
#                     )
#                     st.write(st.session_state.response_data["output"])
#                     st.session_state.email_sent = st.session_state.response_data["email_sent"]

#         # Option to reset interaction
#         if st.session_state.response_data and not st.session_state.response_data["prompt_email"]:
#             if st.button("Ask another question"):
#                 st.session_state.user_query = ""
#                 st.session_state.user_email = ""
#                 st.session_state.email_sent = False
#                 st.session_state.response_data = None

#     # Call the interface function
#     create_streamlit_interface()
# elif  st.session_state.page =="AI Project Mangement experience":
#         load_page("pages/project_roadmap.py")
# elif  st.session_state.page =="Robotic Process Automation and Natural Language Processing":
#         load_page("pages/analyse_workflow.py")
# elif  st.session_state.page =="Recurrent Neural Network-Long Short Term Memory Networks":
#         load_page("pages/lstm.py")
# elif st.session_state.page == "Supervised learning":
#         load_page("pages/features-sales.py")
# elif st.session_state.page == "Unsupervised learning":
#         load_page("pages/segmentation.py")
# elif  st.session_state.page =="Conversational AI fine-tuned with Retrieval Augmented Generation":
#         load_page("pages/conversational_ai.py")
# elif  st.session_state.page =="Natural Language Processing & Generative AI":
#         load_page("pages/survey_proofreading.py")
# elif  st.session_state.page =="Computer Vision - Image Text Extraction":
#         load_page("pages/text-extraction-image.py")
# elif  st.session_state.page =="Computer Vision - Object Detection":
#         load_page("pages/object_detection_size.py")  
# elif  st.session_state.page =="Sales Agent- Agentic Framework":
#         load_page("pages/sales_agent.py") 
# elif  st.session_state.page =="NLP and Generative AI: Speech-to-Text AI Voice Agent":
#         load_page("pages/speach-to-text.py")   
# elif  st.session_state.page =="Computer Vision, Generative AI: A Text-to-Speech, Audio, and Video Generator":
#         load_page("pages/text-to-speach.py")          
# elif  st.session_state.page =="Customer Chatbot Fine Tunned with ChatGPT Turbo":
#         load_page("pages/chatbot_fine_tuned.py")  
