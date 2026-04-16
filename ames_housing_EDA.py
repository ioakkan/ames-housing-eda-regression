import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Loading the ames_housing dataset
house_dataset =  pd.read_csv('dataset/house_prices.csv')


# Creating new DF with all the numerical columns of initial dataframe
numeric_columns = house_dataset.select_dtypes('number').copy() 

# Calculate the correlation between numerical columns
house_correlation_matrix = numeric_columns.corr()

# Minimum threshold value for correlated features
threshold = 0.6 

# Finding correlated features with SalePrice that satisfy the threshold
high_correlated_features = house_correlation_matrix[abs(house_correlation_matrix['SalePrice']) >= threshold].index.tolist() 

 # Remove the target column from the features
high_correlated_features.remove('SalePrice') 

# After checking we remove features because of redudancy and high correlation with other selected features  to prevent multicolinearity(no high correlation between features)
high_correlated_features.remove('GarageArea') # redudant feature capturing existing correlation of GarageCars.(GarageCars have lower correlation with other features thats why we remove 'GarageArea')

high_correlated_features.remove('1stFlrSF') # redudant feature capturing existing correlation of 'TotalBsmtSF' .('TotalBsmtSF' have lower correlation with other features thats why we remove '1stFlrSF')

# New Dataframe with high correlated features
high_cor_features_df = house_dataset[high_correlated_features]

# Creating histograms,boxplots,scatteplots  to gain insight on training features
def feature_plots(df,feature:str,target_feature = house_dataset['SalePrice']) :
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    #  Histogram 
    sns.histplot(df[feature], kde=True, ax=axes[0], color='skyblue')
    axes[0].set_title(f'Distribution of {feature}')

    # Boxplot 
    sns.boxplot(x=df[feature], ax=axes[1], color='lightgreen')
    axes[1].set_title(f'Boxplot of {feature}')

    # 3. Scatter + Reg Line 
    sns.regplot(data=df, x=feature, y=target_feature, ax=axes[2], 
                scatter_kws={'alpha':0.3}, line_kws={'color':'red'})
    axes[2].set_title(f'{feature} vs SalePrice')
    plt.savefig(f'visualizations/{feature}_plots.png')
    plt.tight_layout()


if __name__ == "__main__":
    # Quick Review of dataset
    house_dataset.info()
    print(f'number of houses : {len(house_dataset)}')
    # Checking columns for missing values
    print(f'missing values \n: { house_dataset.isna().sum()[house_dataset.isna().sum() > 0]}')
    
    from pathlib import Path

    # Define the folder name to save figures of EDA
    folder_path = Path("visualizations")
    # Create the folder if it doesn't exist
    folder_path.mkdir(parents=True, exist_ok=True)

    # Correlation heatmap  of all features
    plt.figure(figsize=(18,10))
    sns.heatmap(
        house_correlation_matrix,
        annot=True,
        cmap = "coolwarm",
        linewidths=0.3,
        fmt=".2f")
    plt.title('Correlation heatmap  of all features')
    plt.savefig('visualizations/corr_heatmap_all.png')
    plt.show()

    # Correlation heatmap of high correlated features with target variable 'SalePrice'
    high_cor_features_matrix = high_cor_features_df.corr()
    plt.figure(figsize=(18,10))   
    sns.heatmap(
        high_cor_features_matrix,
        annot=True,
        cmap = "coolwarm",
        linewidths=0.3,
        fmt=".2f")
    plt.title('Correlation heatmap  of training features')
    plt.savefig('visualizations/features_corr_heatmap.png')
    plt.show()
    # Visual Analysis of top features
    for col in high_correlated_features:
        feature_plots(house_dataset, col)
    plt.show()
  
