
# ❤️ Heart Disease Detection System

A hybrid system that combines **Machine Learning** with a **Rule-Based Expert System** to assess the risk of heart disease accurately and explainably.

---

## 🛠️ Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/Heart_Disease_Detection.git
cd Heart_Disease_Detection
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Application

```bash
streamlit run ui/app.py
```

---

## 🧠 Methodology

### 1. Machine Learning Engine

* **Algorithm:** Decision Tree Classifier (Scikit-Learn)
* **Preprocessing:**

  * Data normalization using `MinMaxScaler`
  * Encoding categorical variables
* **Objective:**

  * Maximize **recall** to ensure high-risk heart disease cases are not missed

---

### 2. Rule-Based Expert System

* Designed to simulate **clinical decision-making**
* Evaluates **12 medical rules** based on patient data, including:

  * Chest Pain Type
  * ST-Segment Depression
  * Age
  * Cholesterol Levels
* **Advantage:**

  * Provides instant feedback even when the ML model is uncertain

---

## 📊 Visualizations

The system includes:

* 📈 Distribution plots for key clinical features
* 🎯 Dynamic gauges for risk percentage
* 📊 Comparative analysis against the training dataset

---

## 🚀 Key Features

* Hybrid AI approach (ML + Expert System)
* Explainable predictions
* Real-time interactive UI باستخدام Streamlit
* Focus on medical reliability (High Recall)
