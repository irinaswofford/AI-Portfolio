import streamlit as st

# Title and Introduction
st.title("AI Sales Agent for ABC")
st.write("""
    The AI Sales Agent is designed to streamline the customer engagement and sales processes by automating key interactions.
    It automates tasks such as providing **product recommendations**, **answering customer inquiries**, **scheduling consultations**, and eventually facilitating **purchases through Stripe (coming soon)**.
    
    **Important Note**: Due to the high costs associated with using APIs, we are unable to upload the live agent in this interface for demonstration purposes. For example:
    
    - **Google Calendar API**: The cost depends on the number of events and the frequency of scheduling.
    - **Stripe API**: Stripe charges transaction fees, typically **2.9% + 30 cents** per successful transaction, which varies based on payment amounts and frequency.
    - **Grog Model and AI Models**: Using advanced AI models like **Grog** for product recommendations can result in varying API costs based on the number of requests.
    
    Therefore, this demo is a showcase of the concept rather than a live implementation.
""")

# Add Video Link
st.markdown("[Click here to watch the demo video on how this AI Sales Agent works.](https://www.loom.com/share/f2daf9be784d44b591019740ce7551c3?sid=c26206eb-13ab-4305-91a3-1296346570a2)")

# Add Image (replace with your own image path)
st.image("images/sale_agent_demo1.png", caption="AI-Scheduling meeting")
st.image("images/sale_agent_demo2.png", caption="AI-Scheduling meeting")

# Customer Engagement
st.markdown("### Customer Engagement")
st.write("""
    The AI agent engages with customers in a friendly and professional manner by:
    - Answering questions and providing detailed information about ABC's products and services.
    
    This tool uses **RAG (Retrieval-Augmented Generation)** search to retrieve general information about ABC's business, services, 
    and products. All data is pulled from our internal documentation, ensuring that the AI agent always provides accurate and 
    up-to-date information to customers. The data is sourced from the file `files/Docs.txt`.
""")

# Product Recommendations
st.markdown("### Product Recommendations")
st.write("""
    The AI agent offers personalized product suggestions by:
    - Understanding the customer's needs and preferences.
    - Retrieving relevant product data based on the customer's profile and prior interactions.
    - Providing real-time recommendations using advanced AI models.
""")

# Consultation Scheduling
st.markdown("###   Consultation Scheduling")
st.write("""
    For customers with complex inquiries or those seeking personalized advice, the AI agent:
    - Schedules consultations with ABC's tech experts.
""")