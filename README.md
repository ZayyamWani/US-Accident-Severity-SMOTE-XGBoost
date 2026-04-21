Project Overview: This project focuses on predicting road accident severity using the US Accidents (2016-2023) dataset containing 7.7 million records
.
The Problem (The Gap): Previous methodologies often suffered from high accuracy but extremely low recall for minority severity classes (Levels 1 and 4), meaning the most critical accidents were often missed by the models
.
My Improved Methodology: To fill this gap, this repository introduces:
Lifecycle Feature Engineering: Extraction of Rush Hour and temporal risk indices
.
Class Imbalance Resolution: Implementation of SMOTE to balance the training set
.
Optimized XGBoost Training: Leveraging ensemble learning to maximize F1-scores across all severity levels.
