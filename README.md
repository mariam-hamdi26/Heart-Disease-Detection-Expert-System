# 🫀 CardioSense: Heart Disease Detection System

**CardioSense** is a hybrid clinical decision support system designed to predict heart disease risk. It combines the interpretability of a **Rule-Based Expert System** with the predictive power of a **Machine Learning (Decision Tree)** model.

---

## 🚀 Key Features
- **Hybrid Intelligence:** Uses both a Decision Tree Classifier and 12+ clinical expert rules.
- **Interactive UI:** Built with **Streamlit** for a premium, user-friendly experience.
- **Data-Driven Insights:** Provides real-time visualizations and patient-specific risk analysis.
- **Explainable AI (XAI):** Not just a "yes/no" answer, but a detailed clinical explanation of the risk factors.

---

## 📂 Project Structure
Following the academic requirements, the project is organized as follows:

```text
Heart_Disease_Detection/
├── data/               # Raw and cleaned datasets (cleaned_data.csv)
├── notebooks/          # Exploratory Data Analysis (EDA) & Model Training
├── rule_based_system/  # Logic for the expert system (rules & engine)
├── ml_model/           # Trained Decision Tree model (.pkl)
├── ui/                 # Streamlit application (app.py)
├── requirements.txt    # List of dependencies
└── README.md           # Project documentation
🛠️ Installation & Setup
Clone the repository:

Bash
git clone [https://github.com/your-username/Heart_Disease_Detection.git](https://github.com/your-username/Heart_Disease_Detection.git)
cd Heart_Disease_Detection
Install dependencies:

Bash
pip install -r requirements.txt
Run the application:

Bash
streamlit run ui/app.py
🧠 Methodology
1. Machine Learning Engine
Algorithm: Decision Tree Classifier (Scikit-Learn).

Processing: Data normalization using MinMaxScaler and encoding of categorical variables.

Focus: High recall to ensure potential heart disease cases are not missed.

2. Rule-Based Expert System
Built to simulate clinical reasoning.

Evaluates 12 clinical rules based on patient symptoms (e.g., Chest Pain Type, ST-segment depression, age, and cholesterol levels).

Provides immediate feedback even if the ML model is uncertain.

📊 Visualizations
The system includes:

Distribution plots for key clinical features.

Dynamic Gauges for risk percentage.

Comparative analysis against the training dataset.
