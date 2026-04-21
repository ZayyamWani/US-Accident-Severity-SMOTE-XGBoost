US Accident Severity: SMOTE & XGBoost Optimization
  
📋 Project Overview
This project focuses on predicting road accident severity using the US Accidents (2016-2023) dataset, which contains approximately 7.7 million accident records spanning 49 states
. The goal is to accurately classify the impact of an accident on traffic flow (Severity levels 1-4) using advanced predictive modeling
.
🚩 The Problem (The Gap)
Existing methodologies in traffic accident analysis often achieve high overall accuracy but suffer from extremely low recall for minority severity classes (Levels 1 and 4)
.
The Gap: Most models are biased toward the majority class (Level 2), meaning the most critical accidents (Level 4) are often missed by the system.
Objective: This project fills this gap by prioritizing the Macro-Averaged F1-score and recall over simple accuracy
.
🛠️ My Improved Methodology
To solve the identified gaps, this repository introduces a refined machine learning pipeline:
1. Lifecycle Feature Engineering
We moved beyond basic spatiotemporal data by extracting temporal risk indices:
Rush Hour Detection: Identified high-risk periods (7-9 AM and 4-6 PM) to capture peak traffic volatility.
Temporal Granularity: Extracted hour, weekday, and month features to understand seasonal and daily accident patterns.
2. Advanced Preprocessing & Cleaning
Given the dataset size (3.06 GB), we implemented memory-efficient handling
:
Threshold-Based Dropping: Columns with more than 40% missing values (like End_Lat and End_Lng) were removed to preserve data quality
.
Imputation: Numeric values were handled via median imputation, and categorical values via mode imputation to ensure zero NaN values for model stability.
3. Class Imbalance Resolution (SMOTE)
To address the primary gap (low recall), we applied Synthetic Minority Over-sampling Technique (SMOTE)
. This ensures the training set has an equal distribution of all severity levels, forcing the model to learn the characteristics of rare, high-severity events.
4. Optimized XGBoost Training
We utilized an XGBoost Classifier trained on the balanced dataset. This ensemble approach allows the model to capture complex, non-linear relationships between weather conditions, points of interest (POI), and accident severity.
📊 Dataset Information
Source: US Accidents (2016 - 2023) on Kaggle
Size: 7.7 Million Records
Timeframe: February 2016 - March 2023
Coverage: 49 States in the Contiguous United States
🚀 How to Run
Clone this repository.
Download the US_Accidents_March23.csv from Kaggle.
Install dependencies: pip install pandas numpy scikit-learn xgboost imbalanced-learn.
Run the Jupyter/Kaggle notebook to view the improved classification report.
📚 References
As per course requirements, the following recent research (2022+) informed this methodology
:
 S. Zhou, "Traffic Accident Severity Prediction Using Machine Learning," IEEE Access, 2024. https://ieeexplore.ieee.org/
 D. P. Yamarthi, "US Road Accident Prediction using ML Algorithms," 2025. https://www.buffalo.edu/
 N. Goubraim, et al., "Boosting Traffic Crash Prediction," Safety, 2025. https://doi.org/10.3390/safety11040121
 J. Tang, et al., "Research on Traffic Accident Severity Level," Systems, 2025. https://doi.org/10.3390/systems13010031
 J. Alotaibi, "Enhancing Traffic Accident Severity Prediction," Vehicles, 2025. https://doi.org/10.3390/vehicles7020038
⚖️ Acknowledgments
If you use this project, please cite the original dataset authors
:
Moosavi, Sobhan, et al. "A Countrywide Traffic Accident Dataset," 2019.
