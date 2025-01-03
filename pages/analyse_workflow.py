import streamlit as st
import time
import pandas as pd
from textblob import TextBlob  # For sentiment analysis
import rpa as rpa  # RPA library for automating actions

# Simulate RPA (Robotic Process Automation) functionality
class RPA:
    def __init__(self):
        st.write("Initializing RPA environment...")

    def url(self, url):
        st.write(f"Navigating to URL: {url}")

    def type(self, element, value):
        st.write(f"Typing '{value}' into {element}")

    def click(self, element):
        st.write(f"Clicking the element: {element}")

    def wait(self, seconds):
        st.write(f"Waiting for {seconds} seconds...")
        time.sleep(seconds)

    def close(self):
        st.write("Closing RPA environment...")

# Simulate sending an email
class Email:
    def send_email(self, recipient, subject, body):
        st.write(f"Email sent to: {recipient}")
        st.write(f"Subject: {subject}")
        st.write(f"Body: {body}")
        return f"Email sent to {recipient}\nSubject: {subject}\nBody:\n{body}"

# Simulated CRM Class
class CRM:
    def __init__(self):
        self.database = pd.DataFrame(columns=["Name", "Email", "Query"])

    def add_lead(self, lead_info):
        self.database = pd.concat([self.database, pd.DataFrame([lead_info])], ignore_index=True)

    def get_leads(self):
        return self.database

# Fake API to simulate CRM data, sentiment analysis, and competitive analysis
def fake_api_get_data():
    crm_data = [
        {"Name": "John Doe", "Email": "john.doe@example.com", "Query": "Interested in product X"},
        {"Name": "Jane Smith", "Email": "jane.smith@example.com", "Query": "Looking for more information on product Y"},
        {"Name": "Alice Johnson", "Email": "alice.johnson@example.com", "Query": "Product Z has been great, but I need help with A feature."},
    ]
    crm_df = pd.DataFrame(crm_data)

    # Fake competitor data
    competitor_data = [
        {
            "competitor": "TechCorp",
            "product_name": "TechPhone 3000",
            "product_description": "Next-gen smartphone with enhanced AI capabilities.",
            "price": 999.99,
            "customer_reviews": [
                {"reviewer": "Alice", "rating": 5, "review": "Excellent product, cutting-edge features!"},
                {"reviewer": "Bob", "rating": 3, "review": "Good phone, but the battery life is mediocre."},
                {"reviewer": "Charlie", "rating": 2, "review": "Not worth the price. Too many bugs."}
            ]
        },
        {
            "competitor": "GigaTech",
            "product_name": "GigaPhone Ultra",
            "product_description": "A budget-friendly phone with solid performance for the price.",
            "price": 499.99,
            "customer_reviews": [
                {"reviewer": "Dave", "rating": 4, "review": "Great value for money, but a bit slow."},
                {"reviewer": "Eve", "rating": 4, "review": "Affordable with decent features. A good choice for students."},
                {"reviewer": "Frank", "rating": 1, "review": "Terrible phone. Screen cracked after a week."}
            ]
        }
    ]

    sentiment_analysis_results = [
        {
            "competitor": "TechCorp",
            "product_name": "TechPhone 3000",
            "average_sentiment": "Mixed",
            "positive_reviews": 1,
            "neutral_reviews": 1,
            "negative_reviews": 1
        },
        {
            "competitor": "GigaTech",
            "product_name": "GigaPhone Ultra",
            "average_sentiment": "Positive",
            "positive_reviews": 2,
            "neutral_reviews": 1,
            "negative_reviews": 1
        }
    ]

    return crm_df, competitor_data, sentiment_analysis_results

# Function to generate competitive analysis report
def generate_competitive_analysis_report(competitor_data, sentiment_analysis_results):
    competitive_analysis_report = []
    for competitor in competitor_data:
        sentiment = next(item for item in sentiment_analysis_results if item['competitor'] == competitor['competitor'] and item['product_name'] == competitor['product_name'])
        analysis = {
            "competitor": competitor["competitor"],
            "product_name": competitor["product_name"],
            "price": competitor["price"],
            "sentiment": sentiment["average_sentiment"],
            "positive_reviews": sentiment["positive_reviews"],
            "negative_reviews": sentiment["negative_reviews"],
            "customer_feedback": [review["review"] for review in competitor["customer_reviews"]],
            "feedback_summary": [review["review"] for review in competitor["customer_reviews"][:2]]  # Limit feedback to the first 2 reviews
        }
        competitive_analysis_report.append(analysis)
    return competitive_analysis_report

# Simulate automating lead submission and sending a follow-up email
def rpa_automate_lead_submission(name, email, query):
    rpa = RPA()  # __init__ is called here, no need to manually call init()
    email_service = Email()

    st.write("Simulating lead submission.. instead of using a real CRM URL.")

    # Hardcoded data input
    rpa.type('input[name="name"]', name)
    rpa.type('input[name="email"]', email)
    rpa.type('textarea[name="query"]', query)

    # Simulating form submission by clicking submit
    rpa.click('button[type="submit"]')

    rpa.wait(2)  # Simulate waiting for the form submission to complete

    st.success("Lead submitted successfully!")

    # Send follow-up email
    subject = "Thanks for contacting us!"
    body = f"Dear {name},\n\nWe have received your inquiry and will get back to you shortly.\n\nBest regards,\nABC Company"

    email_service.send_email(email, subject, body)

    rpa.close()

# Streamlit UI - Displaying the CRM data and email templates
def main():
    # Initialize CRM and Email modules
    crm = CRM()
    email = Email()

    # Fetch data from the fake API
    crm_data, competitor_data, sentiment_analysis_results = fake_api_get_data()

    # Initialize session state to store leads
    if 'leads' not in st.session_state:
        st.session_state['leads'] = crm_data  # Pre-populated CRM data

    # Display the title and introduction
    st.title("RPA and NLP")
    # Explanation of the project
    st.write("""
    This project demonstrates **Robotic Process Automation (RPA)** for automating lead submissions and follow-up emails. It integrates with a simulated **CRM** to capture leads, perform **sentiment analysis** on competitor reviews using **Natural Language Processing (NLP) and generate automated email templates**.
    By automating these tasks, businesses can save time, make more informed decisions, and improve their engagement with leads and customers.
""")

    # Main Form for Simulating Lead Submission
    st.subheader("Submit a New Lead")
    with st.form("lead_form"):
        name = st.text_input("Name", value="John Doe")
        email_address = st.text_input("Email", value="john.doe@example.com")
        query = st.text_input("Query", value="Interested in product X")
        submitted = st.form_submit_button("Submit Lead")

    if submitted:
        # Automate lead submission
        rpa_automate_lead_submission(name, email_address, query)

        # Send follow-up email after lead submission
        st.write("Lead submission and follow-up email automation complete.")

    # Button to generate competitive analysis report
    if st.button("Generate Competitive Analysis Report"):
        # Generate the competitive analysis report
        competitive_analysis_report = generate_competitive_analysis_report(competitor_data, sentiment_analysis_results)

        # Check if CRM data is empty
        if crm_data.empty:
            st.error("No CRM data available.")
        else:
            # Combine all product comparisons into a single email
            st.subheader("Generated Email Template")
            report = competitive_analysis_report[0]  # Select the first competitor report to generate an email
            subject = "Discover the Best Product for You !"
            body = (
                f"Hi {crm_data.iloc[0]['Name']},\n\n"  # Use `.iloc[0]` to access the first row
                f"We noticed you're exploring top options like TechPhone 3000 and GigaPhone Ultra. Before you decide, we’d like to offer you something better:\n\n"
                f"Why Choose Our Product?\n"
                f"TechPhone 3000 (TechCorp): Cutting-edge features, but battery life is mediocre.\n"
                f"Our Product: Superior battery life that lasts all day, even with heavy use!\n\n"
                f"GigaPhone Ultra (GigaTech): Affordable but slightly slower performance.\n"
                f"Our Product: Unmatched speed and power for seamless multitasking.\n\n"
                f"Best regards,\nABC Company"
            )
            st.write(f"**Subject:** {subject}")
            st.write(f"**Body:** {body}")

if __name__ == "__main__":
    main()
