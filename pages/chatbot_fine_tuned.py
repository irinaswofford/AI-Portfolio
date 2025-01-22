import streamlit as st

# Display the title and description
st.markdown("""
# Customer Chatbot Fine-Tuning with ChatGPT Turbo

The fine-tuned chatbot is designed to handle customer support interactions for Acertify, answering questions about their products and services. The chatbot utilizes **ChatGPT Turbo**, a more efficient and cost-effective version of GPT-3.5, fine-tuned on domain-specific data to improve its performance in this particular customer service context.

## Key Aspects of the Implementation:

### 1. **Fine-Tuning with ChatGPT Turbo:**
- The base GPT-3.5 model is fine-tuned with a dataset related to Acertify's services and products. This allows the bot to better understand and respond to specific customer queries with accuracy.
- The model ID  links to the fine-tuned model deployed on OpenAI’s API.

### 2. **Streamlit Integration:**
- The chatbot runs on a **Streamlit** app interface, providing users with an interactive, easy-to-use chat window where they can ask questions, and the bot provides responses in real-time.
- Streamlit is used to create a simple web interface for the bot, where the conversation history is maintained throughout the session.

### 3. **Conversation History:**
- The bot retains the conversation history, which helps in providing coherent responses based on previous interactions. 
- It also includes a **system message** that establishes the chatbot’s persona and context for handling inquiries.

### 4. **Cost Calculation:**
- A **cost calculation** feature has been integrated, which estimates the usage cost based on the number of tokens consumed during each interaction.
- The cost is calculated using a cost-per-token rate for GPT-3.5 ($0.002 per 1000 tokens), and the user is provided with a breakdown of the token usage and cost for each exchange.

### 5. **User Experience:**
- The user is presented with a straightforward UI where they can type their query, and the bot responds, providing a seamless customer service experience.
""")

# Display a screenshot of the Streamlit app interface
st.image('images/custumer_support_chatbot.png', caption='Chatbot Interface in Action')
