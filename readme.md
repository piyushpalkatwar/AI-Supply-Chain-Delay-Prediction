# 🚚 AI Supply Chain Delay Prediction

An end-to-end **Machine Learning project** focused on predicting shipment delays in supply chain operations before dispatch, enabling businesses to proactively reduce logistics risk, improve operational efficiency, and minimize financial losses.

This project applies **Predictive Analytics + Machine Learning + Business Intelligence** techniques to identify high-risk shipments and estimate financial impact caused by supply chain disruptions.

---

## 📌 Project Overview

Supply chain disruptions create significant operational and financial challenges for businesses. This project builds an **AI-powered delay prediction system** that analyzes supplier performance, transportation routes, weather risk, demand spikes, and operational bottlenecks to predict whether a shipment will be delayed.

The model helps logistics teams make proactive decisions such as:

* Shipment rerouting
* Supplier risk monitoring
* Delay prevention planning
* Inventory optimization
* Cost reduction strategy

---

## 🎯 Business Problem

Late shipments impact:

* Customer satisfaction
* Inventory planning
* Warehouse management
* Supplier relationships
* Revenue and operational costs

The objective of this project is to **predict shipment delays before dispatch** using machine learning models and identify the key factors responsible for delays.

---

## 🛠 Tech Stack

**Programming Language**

* Python

**Libraries Used**

* Pandas
* NumPy
* Matplotlib
* Scikit-learn
* Seaborn

**Machine Learning Models**

* Random Forest Classifier
* Gradient Boosting Classifier
* Logistic Regression

---

## 📂 Dataset Used

The project uses multiple CSV datasets simulating a real-world supply chain environment.

### 1. Shipments Data

Contains shipment-related details:

* Shipment ID
* Order Date
* Supplier ID
* Route ID
* Order Value
* Expected Delivery Days
* Shipment Weight
* Delay Status

### 2. Suppliers Data

Supplier performance information:

* Supplier Reliability Score
* Supplier Tier
* Country
* Product Category

### 3. Routes Data

Transportation route details:

* Distance in KM
* Average Delivery Days

---

## ⚙️ Project Workflow

### 1. Data Loading

* Import shipment, supplier, and route datasets
* Merge multiple datasets using SQL-style joins

### 2. Exploratory Data Analysis (EDA)

Analyze delay trends based on:

* Transport Mode
* Supplier Reliability
* Weather Risk
* Port Congestion
* Demand Spike
* Supplier Issues

### 3. Feature Engineering

Created advanced business features:

* Transport Encoding
* Country Encoding
* Supplier Tier Encoding
* Seasonal Demand Analysis
* Value Per KG Calculation
* Delivery Buffer Time
* Peak Season Detection

### 4. Model Training

Trained multiple classification models:

* Random Forest
* Gradient Boosting
* Logistic Regression

### 5. Model Evaluation

Performance measured using:

* AUC Score
* Cross Validation
* Confusion Matrix
* ROC Curve
* Classification Report

### 6. Risk Scoring

Generated shipment risk categories:

* Low Risk
* Medium Risk
* High Risk

### 7. Financial Impact Analysis

Estimated cost savings by identifying shipment delays early and preventing losses.

---

## 📊 Key Insights Generated

The project identifies:

* Most delay-prone transport modes
* High-risk suppliers
* Root causes of shipment delays
* Probability of future shipment delays
* Financial impact of logistics disruptions

---

## 📈 Dashboard & Visualizations

The project generates an analytical dashboard containing:

* Delay Rate by Transport Mode
* Risk Factor Comparison
* ROC Curve Comparison
* Feature Importance Analysis
* Confusion Matrix
* Delay Probability Distribution

Dashboard output:

`p3_supply_chain_dashboard.png`

---

## 📁 Output Files Generated

The model exports the following outputs:

### Shipment Risk Scores

`shipment_risk_scores_output.csv`

Contains:

* Shipment ID
* Delay Probability
* Risk Level

### Supplier Risk Analysis

`supplier_risk_output.csv`

Contains:

* Supplier Delay Rate
* Supplier Reliability Score
* Risk Percentage

### Monthly Delay Trend

`monthly_delay_trend_output.csv`

Contains:

* Monthly Shipments
* Monthly Delay Rate

---

## 💡 Business Value

This AI system helps companies:

✅ Predict shipment delays early
✅ Reduce logistics costs
✅ Improve supply chain efficiency
✅ Optimize supplier management
✅ Improve operational decision-making

---

## Machine Learning Pipeline

Data Collection
↓
Data Cleaning
↓
Feature Engineering
↓
Exploratory Data Analysis
↓
Model Training
↓
Model Evaluation
↓
Risk Prediction
↓
Financial Impact Analysis
↓
Dashboard Generation

---

## Future Improvements

Possible future enhancements:

* Real-time API integration
* Deep Learning models for prediction
* Deployment using Flask or FastAPI
* Cloud deployment on AWS
* Automated alert system for high-risk shipments

---

## Author

**Piyush Palkatwar**

Aspiring **AI/ML Engineer | Data Scientist | Generative AI Enthusiast**

Focused on building real-world machine learning solutions that solve business problems through data-driven decision making.

---

## Project Goal

Transform raw supply chain data into actionable business intelligence and build an AI system capable of preventing operational disruptions before they happen.
