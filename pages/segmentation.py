import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the saved model (e.g., KMeans)
kmeans_model = joblib.load('kmeans_model.pkl')

# Function to get user input via sliders
def get_user_input():
    age = st.slider('Select Age', 18, 100, 30)  # Slider for Age
    family_size = st.slider('Select Family Size', 1, 10, 3)  # Slider for Family Size
    return np.array([[age, family_size]])

# Function to plot clustering results
def plot_clustering(df, labels, centroids, user_input):
    plt.figure(figsize=(15, 7))

    # Scatter plot for all data points, colored by cluster labels
    plt.scatter(df['Age'], df['Family_Size'], c=labels, s=200, cmap='viridis', label='Data Points')

    # Scatter plot for centroids with larger red markers
    plt.scatter(centroids[:, 0], centroids[:, 1], s=300, c='red', alpha=0.5, label='Centroids')

    # Highlight user's input with a distinct marker
    plt.scatter(user_input[0][0], user_input[0][1], c='black', s=300, marker='*', label='User Input')

    plt.xlabel('Age')
    plt.ylabel('Family Size')
    plt.title('K-Means Clustering with User Input Highlighted')
    plt.legend()
    st.pyplot(plt)

# Streamlit app main function
def main():
    st.title("Customer Segmentation")
    # Explanation of the project
    st.write("""
        This project demonstrates **K-Means Clustering** for grouping customers based on **age** and **family size**.
        The **centroids** (central points) of these groups help businesses by:  
        - **Creating tailored marketing campaigns** for specific needs.  
        - **Spotting trends in customer preferences** to improve decisions.

        If you select age: 40 and family size: 4, the app predicts **3 clusters**, which means:
        - The app divides the entire population into **3 distinct clusters** based on **age** and **family size**.
        - The number **3** indicates that there are three groups (or clusters) in total, based on the data provided.
""")


    # Get user input
    user_input = get_user_input()

    # Example dataset to visualize clustering
    df = pd.DataFrame({
        'Age': [22, 25, 30, 35, 40, 45, 50, 55, 60],
        'Family_Size': [1, 2, 2, 3, 3, 4, 4, 5, 5]
    })

    # Append user input to the dataset for visualization
    user_df = pd.DataFrame(user_input, columns=['Age', 'Family_Size'])
    df = pd.concat([df, user_df], ignore_index=True)

    # Predict cluster labels for the dataset
    labels = kmeans_model.predict(df[['Age', 'Family_Size']])
    centroids = kmeans_model.cluster_centers_

    # Predict the cluster for the user input
    user_cluster = kmeans_model.predict(user_input)
    st.write(f"Predicted Cluster for User Input: {user_cluster[0]}")

    # Plot the clustering results
    st.write("Visualize Customer Segments:")
    plot_clustering(df, labels, centroids, user_input)

if __name__ == "__main__":
    main()