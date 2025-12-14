import os
import pickle
import base64
import json
from datetime import datetime
from typing import Optional

from googleapiclient.discovery import build
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

SCOPES = ["https://www.googleapis.com/auth/gmail.compose"]
TOKEN_FILE = "token.pickle"
CREDENTIALS_FILE = "credentials.json"
AUDIT_LOG = "draft_audit_log.json"

def _save_json(path: str, content: str):
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

def load_credentials() -> Credentials:
    """
    Loads Gmail OAuth2 credentials.

    Headless CI support:
    - If env CREDENTIALS_JSON and TOKEN_JSON are set, writes them to files and loads.
    Local dev support:
    - Falls back to existing token.pickle or local OAuth flow.

    Environment variables (optional for CI):
    - CREDENTIALS_JSON: contents of credentials.json (Google OAuth client)
    - TOKEN_JSON: pickled credentials base64 (optional), or serialized token JSON
    """
    # In CI, allow credentials via environment variables
    cred_json = os.getenv("CREDENTIALS_JSON")
    token_json = os.getenv("TOKEN_JSON")

    if cred_json:
        _save_json(CREDENTIALS_FILE, cred_json)

    creds: Optional[Credentials] = None

    # Prefer loading token from file if present
    if os.path.exists(TOKEN_FILE):
        with open(TOKEN_FILE, "rb") as token:
            creds = pickle.load(token)
    elif token_json:
        # Support token provided as base64-pickled creds or raw JSON
        try:
            # Try base64->pickle first
            creds = pickle.loads(base64.b64decode(token_json))
        except Exception:
            # Fallback: token_json is a raw Google Credentials JSON
            try:
                creds = Credentials.from_authorized_user_info(json.loads(token_json), SCOPES)
            except Exception:
                creds = None

        if creds:
            with open(TOKEN_FILE, "wb") as token:
                pickle.dump(creds, token)

    # Refresh or run local flow
    if creds and creds.valid:
        return creds
    if creds and creds.expired and creds.refresh_token:
        creds.refresh(Request())
        with open(TOKEN_FILE, "wb") as token:
            pickle.dump(creds, token)
        return creds

    # Local dev flow (requires browser)
    if os.path.exists(CREDENTIALS_FILE):
        flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_FILE, SCOPES)
        creds = flow.run_local_server(port=0)
        with open(TOKEN_FILE, "wb") as token:
            pickle.dump(creds, token)
        return creds

    raise RuntimeError("No valid Gmail credentials available. Provide CREDENTIALS_JSON/TOKEN_JSON or run local OAuth.")

def create_gmail_draft(creds: Credentials, recipient: str, subject: str, body: str, advisor_id: Optional[str] = None):
    """
    Creates a Gmail draft (never sends), appends advisory disclaimer,
    and logs an audit entry with advisor_id, recipient, subject, and timestamp.
    """
    try:
        service = build("gmail", "v1", credentials=creds)

        disclaimer = (
            "\n\n---\nThis analysis is provided by the AI Market News Agent. "
            "It is for informational purposes only and does not constitute investment advice. "
            "All decisions are the responsibility of the recipient."
        )
        header = "⚠️ This is for informational purposes only. Please review before acting.\n\n"
        full_body = f"{header}{body}{disclaimer}"

        message = MIMEMultipart()
        message["to"] = recipient
        message["subject"] = subject
        message.attach(MIMEText(full_body, "plain"))

        encoded_message = {"raw": base64.urlsafe_b64encode(message.as_bytes()).decode("utf-8")}
        draft = service.users().drafts().create(userId="me", body={"message": encoded_message}).execute()

        # Audit log entry
        log_entry = {
            "advisor_id": advisor_id,
            "recipient": recipient,
            "subject": subject,
            "timestamp": datetime.utcnow().isoformat(),
            "draft_id": draft.get("id")
        }
        with open(AUDIT_LOG, "a", encoding="utf-8") as log_file:
            log_file.write(json.dumps(log_entry) + "\n")

        return draft
    except Exception as e:
        return {"error": f"Error creating draft: {e}"}
