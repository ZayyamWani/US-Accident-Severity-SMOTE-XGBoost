# US Accident Severity: SMOTE & XGBoost Optimization

## 📖 Project Overview
This project focuses on predicting road accident severity using the **US Accidents (2016-2023)** dataset [1, 2]. Containing approximately **7.7 million accident records** spanning 49 states, the goal is to accurately classify the impact of an accident on traffic flow (Severity levels 1-4) using advanced predictive modeling [1-3].

## 🚩 The Problem (The Gap)
Existing methodologies in traffic accident analysis often achieve high overall accuracy but suffer from **extremely low recall for minority severity classes** (Levels 1 and 4) [1].
* **The Gap:** Most models are biased toward the majority class (Level 2), meaning the most critical accidents (Level 4) are often missed by the system [1].
* **Objective:** This project fills this gap by prioritizing the **Macro-Averaged F1-score** and recall over simple accuracy [1].

## 🛠️ My Improved Methodology
To solve the identified gaps, this repository introduces a refined machine learning pipeline [1]:

### 1. Lifecycle Feature Engineering
We moved beyond basic spatiotemporal data by extracting **temporal risk indices** [1]:
* **Rush Hour Detection:** Identified high-risk periods (7-9 AM and 4-6 PM) to capture peak traffic volatility [1].
* **Temporal Granularity:** Extracted hour, weekday, and month features to understand seasonal and daily accident patterns [1].

### 2. Advanced Preprocessing & Cleaning
Given the dataset size (3.06 GB), we implemented memory-efficient handling [1, 3]:
* **Threshold-Based Dropping:** Columns with more than 40% missing values (such as `End_Lat` and `End_Lng`) were removed to preserve data quality [1, 4].
* **Imputation:** Numeric values were handled via median imputation, and categorical values via mode imputation to ensure zero NaN values for model stability [1].

### 3. Class Imbalance Resolution (SMOTE)
To address the primary gap (low recall), we applied **Synthetic Minority Over-sampling Technique (SMOTE)** [1]. This ensures the training set has an equal distribution of all severity levels, forcing the model to learn the characteristics of rare, high-severity events [1].

### 4. Optimized XGBoost Training
We utilized an **XGBoost Classifier** trained on the balanced dataset [1]. This ensemble approach allows the model to capture complex, non-linear relationships between weather conditions, points of interest (POI), and accident severity [1].

## 📊 Dataset Information
* **Source:** [US Accidents (2016 - 2023) on Kaggle](https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents) [1, 5].
* **Size:** 7.7 Million Records [1, 2].
* **Timeframe:** February 2016 - March 2023 [1, 2, 6].
* **Coverage:** 49 States in the Contiguous United States [1, 2, 6].

## 🚀 How to Run
1. Clone this repository [1].
2. Download the `US_Accidents_March23.csv` from Kaggle [1, 3].
3. Install dependencies: 
   ```bash
   pip install pandas numpy scikit-learn xgboost imbalanced-learn
   ``` [1].
4. Run the Jupyter/Kaggle notebook to view the improved classification report [1].

## 📚 References
As per course requirements, the following recent research (2022+) informed this methodology [1]:
* **[1]** S. Zhou, "Traffic Accident Severity Prediction Using Machine Learning," *IEEE Access*, 2024. [https://ieeexplore.ieee.org/](https://ieeexplore.ieee.org/) [1].
* **[5]** D. P. Yamarthi, "US Road Accident Prediction using ML Algorithms," *University at Buffalo*, 2025. [https://www.buffalo.edu/](https://www.buffalo.edu/) [1].
* **[2]** N. Goubraim, et al., "Boosting Traffic Crash Prediction," *Safety*, 2025. [https://doi.org/10.3390/safety11040121](https://doi.org/10.3390/safety11040121) [1].
* **[7]** J. Tang, et al., "Research on Traffic Accident Severity Level," *Systems*, 2025. [https://doi.org/10.3390/systems13010031](https://doi.org/10.3390/systems13010031) [1].
* **[6]** J. Alotaibi, "Enhancing Traffic Accident Severity Prediction," *Vehicles*, 2025. [https://doi.org/10.3390/vehicles7020038](https://doi.org/10.3390/vehicles7020038) [1].

## ⚖️ Acknowledgments
If you use this project, please cite the original dataset authors [1]:
> Moosavi, Sobhan, Mohammad Hossein Samavatian, Srinivasan Parthasarathy, and Rajiv Ramnath. "A Countrywide Traffic Accident Dataset," 2019 [1, 7].
