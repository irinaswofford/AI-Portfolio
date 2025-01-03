import streamlit as st
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# Clear the cache on app load
st.cache_data.clear()  # Clear Streamlit's cache

# Historical Sales Data (Years and corresponding sales)
years = np.array([2018, 2019, 2020, 2021, 2022]).reshape(-1, 1)
sales = np.array([150, 175, 200, 250, 300])  # Example sales data for each year

# Initialize and fit the linear regression model
model = LinearRegression()
model.fit(years, sales)

# Predict sales for 2023
predicted_sales_2023 = model.predict([[2023]])

# Calculate the growth rate from 2022 to 2023
growth_rate = (predicted_sales_2023[0] - sales[-1]) / sales[-1] * 100

# Make predictions on the training data
sales_predictions = model.predict(years)

# Calculate MSE, RMSE, and R²
mse = mean_squared_error(sales, sales_predictions)
rmse = np.sqrt(mse)
r2 = r2_score(sales, sales_predictions)

# Dynamic Report Generation

# 1. Sales Growth Trend Analysis
sales_growth = np.diff(sales) / sales[:-1] * 100  # Growth between each year
average_growth = np.mean(sales_growth)

# Dynamic Insights Generation
if average_growth > 10:
    growth_insight = "The company is experiencing strong sales growth, likely driven by market demand or successful marketing campaigns."
elif average_growth > 5:
    growth_insight = "Sales are steadily growing, which could be attributed to product improvements or gradual market expansion."
else:
    growth_insight = "Sales have grown moderately, suggesting the company may benefit from further investments in marketing and product innovation."

# Recommendations based on the growth rate
if growth_rate > 20:
    recommendations = [
        "Consider expanding your product lines to take advantage of rapid market interest.",
        "Boost marketing campaigns, especially in regions with strong growth, to capitalize on this trend."
    ]
elif growth_rate > 10:
    recommendations = [
        "Increase investment in customer retention programs to maintain the growth trajectory.",
        "Focus on regional expansion or targeting new customer segments."
    ]
else:
    recommendations = [
        "Explore new product offerings or enhance existing ones to drive more sales.",
        "Revamp marketing strategies to better capture potential customer interest."
    ]

# Generate Report
report = f"""

### 1. Sales Growth Trend Observed Over the Past 5 Years

The sales data from 2018 to 2022 reflects a consistent upward trend. The average growth rate per year has been {average_growth:.2f}%, indicating {growth_insight}

### 2. Key Factors Contributing to the Projected Sales Increase

The projected sales increase in 2023, with an expected figure of ${predicted_sales_2023[0]:.2f}, is likely influenced by the following factors:

- **Market Expansion**: The company is reaching new geographical regions and demographics.
- **Product Innovation**: Introduction of new products/features is driving demand.
- **Economic Conditions**: Positive consumer confidence and spending behavior may help drive sales in the sector.

### 3. Recommendations to Further Boost Sales Performance in 2023 and Beyond

To further boost sales, we recommend the following actions:
- {', '.join(recommendations)}

### 4. Future Sales Target Levels, Growth Rates, and Expectations

To achieve its long-term growth objectives, the company should target a compounded annual growth rate (CAGR) of 10-12% over the next five years. 

These targets are achievable with continued investment in market expansion, innovation, and customer engagement.

---

### Conclusion

The sales forecast for 2023 at ${predicted_sales_2023[0]:.2f} reflects the ongoing success and the strong growth trajectory of the company. With continued focus on innovation and marketing, the company is poised for continued growth in the coming years.
"""

# Streamlit app to display the content
st.title("Sales Forecasting")
# Explanation of the model
st.write("""
In this project, I am using a **linear regression model** to predict **sales for 2023** based on historical sales data from the years 2018 to 2022. The model's performance is evaluated using key metrics, and a dynamic report is generated to provide insights and recommendations based on the sales data. 

At the end of this prediction, the model evaluation metrics (MSE, RMSE, and R²) are displayed to give a comprehensive understanding of the model's performance. This means that:

- The **MSE of 87.50** shows that the model's predictions are relatively close to the actual sales values.
- The **RMSE of 9.35** confirms that the model's average error in predicting sales is $9.35, which is very reasonable for forecasting purposes.
- The **R² of 0.97** demonstrates that 97% of the variance in the sales data is explained by the model, indicating a strong and reliable fit.
""")
st.write(f"**Predicted Sales for 2023**: ${predicted_sales_2023[0]:.2f}")

st.subheader("Generated Sales Report")
# Display the dynamic report
st.write(report)

# Plotting the historical sales data and regression line
plt.scatter(years, sales, color='blue', label='Actual Sales')
plt.plot(years, model.predict(years), color='red', label='Regression Line')
plt.scatter(2023, predicted_sales_2023, color='green', label='Predicted Sales for 2023', zorder=5)

# Labels and title
plt.xlabel("Year")
plt.ylabel("Sales ($)")
plt.title("Sales Forecasting using Linear Regression ")
plt.legend()

# Show the plot
st.pyplot(plt)

# Display MSE, RMSE, and R²
st.subheader("Model Evaluation Metrics")
st.write(f"Mean Squared Error (MSE): {mse:.2f}")
st.write(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
st.write(f"R-squared (R²): {r2:.2f}")
