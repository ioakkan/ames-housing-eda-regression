# Ames Housing: Price Prediction Model
This project explores and models housing prices in Ames, Iowa using a full data science workflow. It includes exploratory data analysis (EDA), feature selection, and a multiple linear regression model to predict house prices. The focus is on interpretability, feature understanding, and baseline predictive performance.

---

## Project Workflow

### 1. Feature Selection
From a dataset with 80+ variables, key features were selected to reduce complexity and improve model stability.

- **Correlation filtering:** Selected features with strong correlation (|r| ≥ 0.6) with **SalePrice**
- **Multicollinearity handling:** Removed overlapping features to avoid redundancy  
  - Dropped **GarageArea** in favor of **GarageCars**  
  - Dropped **1stFlrSF** in favor of **TotalBsmtSF**

**Final feature set:**
- OverallQual  
- TotalBsmtSF  
- GrLivArea  
- GarageCars  

---

## Data Visualization

All visualizations are stored in the `visualizations/` folder.  
Each feature plot includes a histogram, scatter plot, and boxplot combined in one figure.

---

### Correlation Heatmap

![Correlation Heatmap](visualizations/features_corr_heatmap.png)

---

### Overall Quality (OverallQual)

![OverallQual Plots](visualizations/OverallQual_plots.png)

---

### Living Area (GrLivArea)

![GrLivArea Plots](visualizations/GrLivArea_plots.png)

---

### Basement Area (TotalBsmtSF)

![TotalBsmtSF Plots](visualizations/TotalBsmtSF_plots.png)

---

### Garage Capacity (GarageCars)

![GarageCars Plots](visualizations/GarageCars_plots.png)

---

## Model Development

A **Multiple Linear Regression** model was trained to predict house prices.

- Train/Test split: 80% / 20%  
- Focus: interpretability and baseline performance  

---

## Model Evaluation

| Metric | Value |
|------|------|
| R² Score | 0.765 |
| Mean Absolute Error (MAE) | $23,571 |
| Root Mean Squared Error (RMSE) | $31,310 |
| Median Absolute Error | $20,001 |

---

## Key Findings

- Model explains ~76% of price variance  
- Small gap between training and testing performance → good generalization  
- Higher RMSE indicates sensitivity to expensive outliers  
- Better performance on typical homes than luxury properties  

---

