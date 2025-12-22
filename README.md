# Spotify Customer Churn Prediction

This project focuses on predicting customer churn for Spotify users using machine learning. By analyzing user behavior, demographics, and usage patterns, the model identifies users who are likely to cancel their subscription, enabling proactive retention strategies.

## 📌 Project Overview

Customer churn is a critical metric for subscription-based services like Spotify. This project analyzes a dataset of user activity to understand the factors contributing to churn and builds a predictive model to classify users as "Churned" or "Not Churned."

**Key Features:**
* **Exploratory Data Analysis (EDA):** Visualizing correlations and distributions of user data (e.g., skip rates, listening time).
* **Data Preprocessing:** Handling categorical variables (Gender, Subscription Type) and scaling numerical features.
* **Predictive Modeling:** Implementing **Logistic Regression** to classify users.
* **Evaluation:** Assessing model performance using accuracy and confusion matrices.

## 📂 Repository Structure

- `Churn_Analysis.ipynb`: The main Jupyter Notebook containing the code for data cleaning, EDA, preprocessing, and model training.
- `spotify_churn_dataset.csv`: (Assumed) The dataset used for training and testing the model.
- `correlation_heatmap.png`: Visualization showing the correlation between different features (e.g., Skip Rate vs. Churn).
- `pairplot_numeric.png`: Pairplot visualizing relationships between numeric variables.
- `README.md`: Project documentation.

## 🛠️ Technologies Used

- **Python**
- **Pandas & NumPy:** For data manipulation and analysis.
- **Matplotlib & Seaborn:** For data visualization (Heatmaps, Pairplots).
- **Scikit-Learn:** For building the Logistic Regression model, splitting data, and evaluation.

## 📊 Dataset Details

The dataset contains user-level information including:
- **Demographics:** Age, Gender, Country.
- **Subscription Info:** Subscription Type (Free, Premium, Family, Student), Device Type.
- **Usage Metrics:**
  - `listening_time`: Total time spent listening.
  - `songs_played_per_day`: Average daily song count.
  - `skip_rate`: Percentage of songs skipped (strong indicator of dissatisfaction).
  - `ads_listened_per_week`: Number of ads consumed.
  - `offline_listening`: Whether the user utilizes offline mode.
- **Target Variable:** `is_churned` (1 = Churned, 0 = Retained).

## 🧠 Model & Analysis

1.  **Data Processing:**
    - Loaded the dataset and checked for null values.
    - Performed feature scaling using `StandardScaler` to normalize numerical inputs.
    - Split data into training and testing sets.

2.  **Model:**
    - **Logistic Regression:** Used as the baseline model to predict the binary outcome of churn.

3.  **Key Insights (Example):**
    - High `skip_rate` is often positively correlated with churn.
    - Users with higher `listening_time` are generally less likely to churn.

## 🚀 How to Run

1.  Clone the repository:
    ```bash
    git clone <your-repo-url>
    ```
2.  Install required libraries:
    ```bash
    pip install pandas numpy matplotlib seaborn scikit-learn
    ```
3.  Open and run the notebook:
    ```bash
    jupyter notebook Churn_Analysis.ipynb
    ```

## 📈 Future Improvements

- Test advanced models like **Random Forest** or **XGBoost** for better accuracy.
- Perform hyperparameter tuning.
- Deploy the model as a simple web app (using Streamlit or Flask).
