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

# -----------------------------
# Config
# -----------------------------
SCOPES = [
    "https://www.googleapis.com/auth/gmail.compose",
    "https://www.googleapis.com/auth/gmail.send"
]
TOKEN_FILE = "token.pickle"
CREDENTIALS_FILE = "credentials.json"
AUDIT_LOG = "draft_audit_log.json"


def _save_json(path: str, content: str):
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


# -----------------------------
# Load credentials
# -----------------------------
def load_credentials() -> Credentials:
    """
    Loads Gmail OAuth2 credentials.
    1. Checks token.pickle.
    2. Refreshes if expired.
    3. Falls back to local OAuth flow if necessary.
    Supports headless usage in GitHub Actions.
    """
    cred_json = os.getenv("CREDENTIALS_JSON")
    token_json = os.getenv("TOKEN_JSON")

    if cred_json:
        _save_json(CREDENTIALS_FILE, cred_json)

    creds: Optional[Credentials] = None

    # Try pickle first
    if os.path.exists(TOKEN_FILE):
        with open(TOKEN_FILE, "rb") as token:
            creds = pickle.load(token)

    # Try base64 token_json if provided
    elif token_json:
        try:
            creds = pickle.loads(base64.b64decode(token_json))
        except Exception:
            try:
                creds = Credentials.from_authorized_user_info(json.loads(token_json), SCOPES)
            except Exception:
                creds = None
        if creds:
            with open(TOKEN_FILE, "wb") as token:
                pickle.dump(creds, token)

    # Refresh if expired
    if creds and creds.expired and creds.refresh_token:
        creds.refresh(Request())
        with open(TOKEN_FILE, "wb") as token:
            pickle.dump(creds, token)
        return creds

    # Local OAuth flow if nothing valid
    if (not creds or not creds.valid) and os.path.exists(CREDENTIALS_FILE):
        flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_FILE, SCOPES)
        creds = flow.run_local_server(port=0)
        with open(TOKEN_FILE, "wb") as token:
            pickle.dump(creds, token)
        return creds

    if creds and creds.valid:
        return creds

    raise RuntimeError(
        "No valid Gmail credentials available. "
        "Provide CREDENTIALS_JSON/TOKEN_JSON or run local OAuth."
    )


# -----------------------------
# Draft or send message
# -----------------------------
def create_or_send_message(
    creds: Credentials,
    recipient: str,
    subject: str,
    body: str,
    advisor_id: Optional[str] = None
):
    """
    Creates a Gmail draft OR sends an email depending on SEND_EMAIL env var.
    - SEND_EMAIL=true → sends immediately
    - default → creates draft
    """
    try:
        service = build("gmail", "v1", credentials=creds)

        # Append disclaimer
        disclaimer = (
            "\n\n---\nThis analysis is provided by the AI Market News Agent. "
            "It is for informational purposes only and does not constitute investment advice. "
            "All decisions are the responsibility of the recipient."
        )
        header = "⚠️ This is for informational purposes only. Please review before acting.\n\n"
        full_body = f"{header}{body}{disclaimer}"

        # Construct email
        message = MIMEMultipart()
        message["to"] = recipient
        message["subject"] = subject
        message.attach(MIMEText(full_body, "plain"))

        encoded_message = {
            "raw": base64.urlsafe_b64encode(message.as_bytes()).decode("utf-8")
        }

        send_mode = os.getenv("SEND_EMAIL", "false").lower() == "true"

        if send_mode:
            result = service.users().messages().send(userId="me", body=encoded_message).execute()
        else:
            result = service.users().drafts().create(userId="me", body={"message": encoded_message}).execute()

        print("Gmail API result:", result)

        # Audit log
        log_entry = {
            "advisor_id": advisor_id,
            "recipient": recipient,
            "subject": subject,
            "timestamp": datetime.utcnow().isoformat(),
            "draft_id": result.get("id") if not send_mode else None,
            "message_id": result.get("id") if send_mode else None,
            "mode": "send" if send_mode else "draft"
        }
        with open(AUDIT_LOG, "a", encoding="utf-8") as log_file:
            log_file.write(json.dumps(log_entry) + "\n")

        return result

    except Exception as e:
        print("Error creating draft:", e)
        return {"error": str(e)}

