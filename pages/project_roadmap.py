import pandas as pd
import streamlit as st

def main():
    # Header and Image for AI Project Roadmaps
    st.title("AI Project Mangement experience")

    # Project Milestone Planning Description
    st.write("""
    - **AI Solution Design:** Led teams in designing scalable AI solutions tailored to business objectives.
    - **Cross-Functional Team Leadership:** Managed and coordinated Data Scientists, Engineers, and stakeholders to ensure alignment and project success.
    - **Timeline Management:** Developed detailed project timelines with clear milestones, tracking progress and ensuring timely delivery.
    - **Risk Management:** Created risk matrices and mitigation plans, identifying potential challenges in AI projects (e.g., data quality, model performance).
    - **Stakeholder Communication:** Facilitated regular updates, ensuring stakeholders were informed and engaged throughout the project lifecycle.
    - **Methodologies:** Applied  a combination of Agile principles  .....fix this 
    - **Resource Allocation:** Effectively managed team resources, balancing workloads to meet deadlines and optimize productivity.
    """)
    st.image("images/Gannt_chart.png", caption="AI Project Roadmap")
    st.image("images/project_roadmap1.png", caption="AI Project Roadmap")
    st.image("images/project_roadmap2.png", caption="AI Project Roadmap")
    st.write("""

    ### Risk Management Matrix:
    To visualize and tackle potential risks, I create a Risk Management Matrix, detailing:
    - Risks such as Data Privacy, Model Bias, and Regulatory Compliance
    - Likelihood, impact, and specific mitigation strategies for each
    - Assignment of ownership to the responsible teams (e.g., Data Governance, Compliance)
    """)

    # Displaying Risk Matrix in Sidebar
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

    st.subheader("Risk Management Matrix")
    st.dataframe(df)

# Run the Streamlit app
if __name__ == "__main__":
    main()
