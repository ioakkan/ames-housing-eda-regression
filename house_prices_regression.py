import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score,mean_absolute_error,mean_absolute_percentage_error, median_absolute_error
from sklearn.model_selection import train_test_split
import numpy as np
from ames_housing_EDA import high_correlated_features # importing train features after EDA 
# Loading Dataset
house_dataset =  pd.read_csv('dataset/house_prices.csv')


# Training Features (after EDA)

training_features = high_correlated_features

print(f"The training features are: \n {training_features} \n")

# Dataframe of selected feautures for linear regression
X_features = house_dataset[training_features]

# Target Feature
Y = house_dataset['SalePrice']

# Splitting the Dataset to training and testing  data 
X_train, X_test, y_train, y_test = train_test_split(X_features, Y,test_size=0.2,random_state=41)

print("---------Dataset samples length/shapes -------")
print(f"X_train length is {len(X_train)} y_train length is {len(y_train)} shapes are  X: {X_train.shape} and Y {y_train.shape}")
print(f"X_test length is {len(X_test)} y_test length is {len(y_test)}  shapes are  X: {X_test.shape} and Y {y_test.shape} \n")

print("--------- House prices statistics --------")
print(f"minimum House  Saleprice is : {Y.min()}")
print(f"maximum House  Saleprice is : {Y.max()}")
print(f"mean House Saleprice is : {Y.mean()}")
print(f"median House Saleprice is : {Y.median()} \n")

 # Create Linear Regression model
model = LinearRegression()

# Train model
model.fit(X_train,y_train)

# Train predictions 
train_predictions = model.predict(X_train)

# Training evaluation 
MSE = mean_squared_error(y_train,train_predictions)
R_square_train = r2_score(y_train,train_predictions)
MAE = mean_absolute_error(y_train,train_predictions)
MAPE = mean_absolute_percentage_error(y_train,train_predictions)
med_absolue_error = median_absolute_error(y_train,train_predictions)
rmse_train = np.sqrt(MSE)

print (f"--------------------- Training metrics  ----------------")
print(f"R^2 score is : {R_square_train:.4f} meaning {R_square_train:.1f} of price changes are captured by these features.") #  Measures how much variance the model explains
print(f"model Error(MSE) : {MSE:.4f}") # Mean Squared Error (MSE)
print(f"On average, the model is off by : {MAE:.3f} $ ") # The average miss (error) in dollars ,This is the "Expected Error" for a typical house. (Bad predicitions-> outliers effect MAE)
print(f" ('worst') average error prediction in dollars (RMSE) : {rmse_train:.4f} &") #  Average Estimation that  punishes outliers  derived from  MSE 
print(f"mean_absolute_percentage_error(MAPE) is : {MAPE}") # Average Error expressed as a percentage of the house price.
print(f"median_absolute_error is : {med_absolue_error} \n") # Median House price error


# Evaluation

# Model's Predictions
y_pred = model.predict(X_test)


# Training evaluation 
MSE = mean_squared_error(y_pred,y_test)
R_square_test = r2_score(y_pred,y_test)
MAE = mean_absolute_error(y_pred,y_test)
MAPE = mean_absolute_percentage_error(y_pred,y_test)
med_absolue_error = median_absolute_error(y_pred,y_test)
rmse_test = np.sqrt(MSE)

print (f"--------------------- Evaluation metrics  ----------------")
print(f"R^2 score is : {R_square_test}") 
print(f"model Error(MSE) : {MSE}")
print(f"average error prediction in dollars (MAE) : {MAE} ") 
print(f"worst average error prediction in dollars(RMSE) {rmse_test}") 
print(f"mean_absolute_percentage_error(MAPE) is : {MAPE}") 
print(f"median_absolute_error is : {med_absolue_error}")

# Checking Overfitting Ratio = TestRMSE/TrainRMSE
print(f"\n TestRMSE/TrainRMSE Model Generalization Check: {rmse_test/rmse_train}")  
print(f"Train R^2:| Test R^2: {R_square_train/R_square_test}")