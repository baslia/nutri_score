#%%
import pandas as pd
#%%
sample = 100_000
opf_data = pd.read_csv('/Users/baslad01/data_dump/openfoodfacts/product_data/en.openfoodfacts.org.products.csv', sep='\t', encoding='utf-8', on_bad_lines='skip', nrows=sample)
opf_data.head()
#%%
# find the columns that have 'sugar' in them
sugar_columns = [col for col in opf_data.columns if 'sugar' in col]
sugar_columns
#%%
# Filter the data where added sugar is not missing
opf_data_sugg_add = opf_data[opf_data['added-sugars_100g'].notna()]
opf_data_sugg_add
#%%
# Plot the nutriscore in function of added sugar
import matplotlib.pyplot as plt
import seaborn as sns

sorted_nutriscore_grades = sorted(opf_data_sugg_add['nutriscore_grade'].unique())


sns.set(style="whitegrid")
plt.figure(figsize=(12, 6))
sns.boxplot(x='nutriscore_grade', y='added-sugars_100g', data=opf_data_sugg_add, order=sorted_nutriscore_grades)
plt.title('Nutriscore in function of added sugar')
plt.show()
#%%
# Plot the energy by 100g in fct of sugar and added suggar
plt.figure(figsize=(12, 6))
sns.scatterplot(x='added-sugars_100g', y='energy_100g', hue='nutriscore_grade', data=opf_data_sugg_add)
plt.title('Energy in function of sugar and added sugar')
plt.show()
#%%
# Build a regression model to predict the added sugar value given all the other nutrients in the product marked by _100g
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR

# Filter the data where added sugar is not missing
opf_data_sugg_add = opf_data[opf_data['added-sugars_100g'].notna()]

# Filter the columns that have _100g in them
opf_num_features = opf_data_sugg_add.filter(regex='_100g')

# Remove the added sugar column
opf_num_features.drop(columns='added-sugars_100g', inplace=True)

# replace Nan with zeros
opf_num_features.fillna(0, inplace=True)

# Filter the added sugar column
added_sugar = opf_data_sugg_add['added-sugars_100g']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(opf_num_features, added_sugar, test_size=0.2, random_state=11)

# Initialize the models
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=11),
    'Support Vector Regressor': SVR(),
    'K-Neighbors Regressor': KNeighborsRegressor()
}

# Train the models and predict the values
predictions = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    predictions[name] = model.predict(X_test)

#%%
# Plot the predicted values against the true values
plt.figure(figsize=(12, 6))
for name, y_pred in predictions.items():
    plt.scatter(y_test, y_pred, label=name, alpha=0.5)

# Add the true value line
min_val = min(y_test.min(), min([y_pred.min() for y_pred in predictions.values()]))
max_val = max(y_test.max(), max([y_pred.max() for y_pred in predictions.values()]))
plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', label='True Value Line')

plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.title('True vs Predicted Values for Multiple Models')
plt.legend()
plt.show()
#%%
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Initialize a dictionary to store the evaluation metrics
metrics = {
    'Model': [],
    'MAE': [],
    'MSE': [],
    'R²': []
}

# Calculate the evaluation metrics for each model
for name, y_pred in predictions.items():
    metrics['Model'].append(name)
    metrics['MAE'].append(mean_absolute_error(y_test, y_pred))
    metrics['MSE'].append(mean_squared_error(y_test, y_pred))
    metrics['R²'].append(r2_score(y_test, y_pred))

# Convert the metrics dictionary to a DataFrame for better visualization
metrics_df = pd.DataFrame(metrics)

# Display the metrics
print(metrics_df)

# Determine the best model based on the lowest MSE
best_model = metrics_df.loc[metrics_df['MSE'].idxmin()]

print(f"Best model based on MSE: {best_model['Model']}")
#%%
# Plot the feature importances for the Random Forest model
import numpy as np
# Set a threshold for feature importance
threshold = 0.02

# Get the feature importances from the Random Forest model
importances = models['Random Forest'].feature_importances_

# Filter the features based on the threshold
high_importance_indices = np.where(importances > threshold)[0]
high_importance_features = X_train.columns[high_importance_indices]
high_importance_values = importances[high_importance_indices]

# Sort the features by importance
sorted_indices = np.argsort(high_importance_values)[::-1]
high_importance_features = high_importance_features[sorted_indices]
high_importance_values = high_importance_values[sorted_indices]

# Plot the high importance features
plt.figure(figsize=(12, 6))
plt.bar(high_importance_features, high_importance_values)
plt.xlabel('Features')
plt.ylabel('Importance')
plt.title('High Importance Features for Random Forest Model')
plt.xticks(rotation=90)
plt.show()
#%%
# Predict the added sugar value for rest of the data, using the best model
# Filter the data where added sugar is missing
opf_data_sugg_add_missing = opf_data[opf_data['added-sugars_100g'].isna()]

# Filter the columns that have _100g in them
opf_num_features_missing = opf_data_sugg_add_missing.filter(regex='_100g')

# Remove the added sugar column
opf_num_features_missing.drop(columns='added-sugars_100g', inplace=True)

# replace Nan with zeros
opf_num_features_missing.fillna(0, inplace=True)

# Predict the added sugar values using the best model
added_sugar_missing = models['Random Forest'].predict(opf_num_features_missing)

# Add the predicted values to the original DataFrame
opf_data_sugg_add_missing['added-sugars_100g'] = added_sugar_missing

# Display the updated DataFrame
opf_data_sugg_add_missing
#%%
# Get the top 10 products with the highest predicted added sugar values
top_10_products = opf_data_sugg_add_missing.nlargest(10, 'added-sugars_100g')
top_10_products[['product_name', 'added-sugars_100g']]
#%%
# Get the top 10 products with the lowest predicted added sugar values
bottom_10_products = opf_data_sugg_add_missing.nsmallest(10, 'added-sugars_100g')
bottom_10_products[['product_name', 'added-sugars_100g']]
#%%
