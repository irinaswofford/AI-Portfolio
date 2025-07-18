import pandas as pd
import streamlit as st

def main():
    # Header for the page
    st.title("AI Product & Project Leadership Highlights")

    # Description with bullet points (using markdown)
    st.markdown("""
    - Defined the product vision and roadmap for multiple AI solutions aligned to user needs and business goals (e.g., automation, personalization, fraud detection).
    - Led cross-functional teams (Data Scientists, MLOps, Engineers, UX) through the full lifecycle — from ideation to MVP delivery and iteration.
    - Scoped and prioritized product features based on technical feasibility, stakeholder input, and ROI using tools like JIRA, ClickUp, and Microsoft Project!.
    - Developed AI risk governance frameworks, including Risk Management Matrices (privacy, model bias, regulatory compliance), and assigned ownership for mitigation.
    - Ensured delivery on time and budget, creating Agile timelines with milestone tracking, and proactively managing scope creep.
    - Communicated progress to execs and business stakeholders, tailoring updates to technical and non-technical audiences.
    - Launched internal AI tools to enhance user engagement and demonstrate applied GenAI capabilities.
    """)

    # Show images with captions
    st.image("images/Gannt_chart.png", caption="AI Project Roadmap")
    st.image("images/project_roadmap1.png", caption="AI Project Roadmap")
    st.image("images/project_roadmap2.png", caption="AI Project Roadmap")

    # Risk Management Matrix explanation
    st.write("""
    ### Risk Management Matrix:
    To visualize and tackle potential risks, I create a Risk Management Matrix, detailing:
    - Risks such as Data Privacy, Model Bias, and Regulatory Compliance
    - Likelihood, impact, and specific mitigation strategies for each
    - Assignment of ownership to the responsible teams (e.g., Data Governance, Compliance)
    """)

    # Risk data in a DataFrame
    risk_data = {
        "Risk": ["Data Privacy", "Model Bias", "False Positives/Negatives", "Regulatory Compliance", "Customer Trust"],
        "Likelihood": ["Medium", "Medium", "High", "Low", "Medium"],
        "Impact": ["High", "High", "Medium", "High", "High"],
        "Mitigation Plan": [
            "Anonymize data, audit compliance regularly",
            "Conduct bias audits, ensure fairness metrics",
            "Validate thresholds, conduct pilot tests",
            "Collaborate with compliance teams, document",
            "Ensure transparency, provide opt-out mechanisms"
        ],
        "Owner": ["Data Governance", "Data Science Team", "AI Development", "Compliance Team", "Marketing/Legal"]
    }
    df = pd.DataFrame(risk_data)

    # Display the risk matrix as a table
    st.subheader("Risk Management Matrix")
    st.dataframe(df)

if __name__ == "__main__":
    main()
