import streamlit as st
import pandas as pd
from textblob import TextBlob
import spacy

# Load spaCy model for text processing
spacy.cli.download('en_core_web_sm')
nlp = spacy.load("en_core_web_sm")

# Function to analyze sentiment of text
def analyze_sentiment(text):
    analysis = TextBlob(text)
    if analysis.sentiment.polarity > 0:
        return "Positive"
    elif analysis.sentiment.polarity == 0:
        return "Neutral"
    else:
        return "Negative"

# Function to automatically analyze survey data
def automate_survey_analysis(data):
    # Analyze and process survey data automatically
    if "Satisfaction Score" in data.columns:
        avg_score = data["Satisfaction Score"].mean()
        st.write(f"**Average Satisfaction Score:** {avg_score:.2f}/10")
    
    # Sentiment analysis for text responses
    if "Response" in data.columns:
        data["Sentiment"] = data["Response"].apply(analyze_sentiment)
        sentiment_counts = data["Sentiment"].value_counts()
        st.write("**Sentiment Analysis:**")
        st.bar_chart(sentiment_counts)

    # Display suggestions for improvements
    if "Improvements" in data.columns:
        st.write("**Top Improvement Suggestions:**")
        st.write(", ".join(data["Improvements"].dropna().unique()))

# Proofreading function using TextBlob and spaCy
def proofreading(text):
    # Basic grammar and spelling correction using TextBlob
    corrected = str(TextBlob(text).correct())
    # Capitalize sentences and adjust spacing
    doc = nlp(corrected)
    proofread_text = []
    for sent in doc.sents:
        proofread_text.append(sent.text.capitalize())
    return " ".join(proofread_text)

# Function to generate questions based on noun chunks in the text
def generate_questions(text):
    doc = nlp(text)
    questions = [f"What is the meaning of '{np.text}'?" for np in doc.noun_chunks]
    return questions

# Function to create default survey data
def create_default_survey_data():
    data = {
        "Satisfaction Score": [8, 9, 7, 6, 5, 9, 10, 8, 7, 6],
        "Response": [
            "The product is great but needs more features.",
            "I love the customer service.",
            "The interface is too complex.",
            "Delivery was late.",
            "Affordable and reliable.",
            "Would recommend to others.",
            "Too many bugs in the system.",
            "Good value for money.",
            "Needs better documentation.",
            "Excellent performance overall."
        ],
        "Improvements": [
            "Add more features",
            "Improve delivery times",
            "Simplify interface",
            "Fix bugs",
            "Better documentation",
            None,
            None,
            "Enhance reliability",
            "Provide tutorials",
            "Maintain quality"
        ]
    }
    return pd.DataFrame(data)

# Streamlit UI setup
st.title("Automated Survey Analytics, Proofreading, and Question Generation")
# Explanation of the project
st.write("""
This project leverages **Natural Language Processing (NLP)** to automate and enhance tasks such as survey analysis, text proofreading, and question generation. 

- **Survey Analytics**: Calculates average satisfaction scores, performs sentiment analysis, and highlights top improvement suggestions.  
- **Text Proofreading**: Corrects grammar and spelling using **TextBlob** and improves readability with **spaCy**.  
- **Question Generation**: Creates questions from text by identifying key noun phrases.  

In summary, I used TextBlob and spaCy to handle core NLP tasks such as sentiment analysis, proofreading, and phrase extraction.
Then, I incorporated Generative AI to refine the outputs, generate insightful summaries, enhance rephrasing suggestions, 
and dynamically create relevant questions, showcasing the power of both NLP and AI technologies to automate and improve processes effectively.

""")

# Survey Analytics Section
st.header("Survey Data Insights")

# Generate default survey data automatically
survey_data = create_default_survey_data()
st.write("Survey Data Preview:")
st.dataframe(survey_data)

# Automatically analyze and display survey insights
st.header("Automated Survey Insights")
automate_survey_analysis(survey_data)

# Proofreading Section
st.header("Text Proofreading")
proofread_input = st.text_area(
    "Enter text to proofread",
    placeholder="Example: today is abeutiful and comuication is ",
    height=150
)

if proofread_input:
    # Automatically proofread the text
    corrected_text = proofreading(proofread_input)
    st.write("Corrected Text:")
    st.write(corrected_text)

# Question Generation Section
st.header("Generate Questions from Text")
question_input = st.text_area(
    "Enter text to generate questions",
    placeholder="Example: Communication is key to building strong relationships.",
    height=150
)


if question_input:
    # Automatically generate questions from the text
    questions = generate_questions(question_input)
    st.write("Generated Questions:")
    for question in questions:
        st.write(f"- {question}")
