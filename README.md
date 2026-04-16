# Ames Housing: Analytical Valuation Engine

This project implements a high-performance regression framework to quantify property values in the Ames, Iowa, housing market. By synthesizing automated feature engineering with multivariate linear modeling, the engine translates architectural and spatial attributes into precise financial insights, maintaining a strong focus on statistical reliability and model interpretability.

## Technical Workflow

### 1. Feature Intelligence & Discovery
The initial phase focused on identifying the primary drivers of market value from a high-dimensional dataset of 80+ variables. Rather than utilizing the entire feature space, a strict statistical filter was applied to ensure model parsimony and focus on variables with the highest predictive power.

* **Correlation Filtering:** Automated selection of features maintaining a Pearson Correlation $|r| \ge 0.6$ with the target variable (**SalePrice**).
* **Multicollinearity Mitigation:** Executed a redundancy audit to eliminate overlapping variance and ensure coefficient stability. This prevents the model from "double-counting" similar features.
    * *Strategic Decision:* **GarageArea** was removed in favor of **GarageCars**.
    * *Strategic Decision:* **1stFlrSF** was removed in favor of **TotalBsmtSF**.
* **Result:** Engineered a refined, high-performance feature set including **OverallQual**, **TotalBsmtSF**, **GrLivArea**, and **GarageCars**.

### 2. Predictive Modeling & Implementation
Following the discovery stage, a **Multivariate Linear Regression** framework was implemented to quantify the relationship between curated physical attributes and market price.

* **Validation Strategy:** Partitioned data into an 80/20 Train-Test split to rigorously validate the model's ability to generalize to unseen data.
* **Interpretability:** By utilizing a streamlined feature set, the model remains transparent, providing clear evidence of how specific property improvements (like increasing living area or basement size) contribute to total dollar value.

### 3. Performance Analysis & Evaluation
The final stage involved a multi-metric stress test to gain a comprehensive understanding of prediction accuracy and market volatility.

| Metric | Testing Value | Statistical Narrative |
| :--- | :--- | :--- |
| **R² Score** | **0.7650** | The model captures ~76.5% of the variance in residential market pricing. |
| **Mean Absolute Error (MAE)** | **$23,571** | The average absolute deviation from the true market price. |
| **RMSE** | **$31,310** | The elevated RMSE indicates sensitivity to high-value outliers. |
| **Median Absolute Error** | **$20,001** | For 50% of cases, the model's accuracy is higher than the MAE suggests. |

## Key Technical Insights

* **Model Generalization:** The marginal delta between Training **R² (0.77)** and Testing **R² (0.76)** confirms a highly robust model that effectively avoids overfitting, ensuring reliable performance on new data.
* **Error Distribution Analysis:** The divergence between the **MAE** and the **Median Absolute Error** highlights a critical market insight: the model is exceptionally accurate for "typical" residential properties but faces higher variance when predicting unique, luxury-tier estates. This suggests that high-end homes may be influenced by subjective factors not captured in the current feature set.