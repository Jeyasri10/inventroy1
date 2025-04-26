import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error as mae
import streamlit as st

# Step 1: Load the DataFrame (Ensure the path to your CSV is correct)
try:
    df = pd.read_csv('retail_store_inventory.csv')  # Replace with your actual dataset path
    print("DataFrame loaded successfully.")
except FileNotFoundError:
    print("Error: The file was not found. Please check the file path.")
    exit()

# Step 2: Feature Engineering
df['Discounted Price'] = df['Price'] * (1 - df['Discount'] / 100)
df['Price Difference'] = df['Price'] - df['Competitor Pricing']
df['Stock to Order Ratio'] = df['Inventory Level'] / (df['Units Ordered'] + 1)
df['Forecast Accuracy'] = abs(df['Demand Forecast'] - df['Units Sold']) / (df['Units Sold'] + 1)

# Step 3: Target Creation - Classification
def classify_units(units):
    if units <= 50:
        return 0  # Low
    elif units <= 150:
        return 1  # Medium
    else:
        return 2  # High

df['Demand Class'] = df['Units Sold'].apply(classify_units)

# Step 4: Feature Selection
feature_cols = [
    'Price', 'Discount', 'Demand Forecast', 'Competitor Pricing',
    'Discounted Price', 'Price Difference', 'Stock to Order Ratio',
    'Forecast Accuracy', 'Holiday/Promotion', 'Year', 'Month', 'Day'
] + [col for col in df.columns if 'Category_' in col or 'Region_' in col or
      'Weather Condition_' in col or 'Seasonality_' in col]

X = df[feature_cols]
y = df['Demand Class']

# Step 5: Train-Test Split
X_train, X_val, Y_train, Y_val = train_test_split(X, y, test_size=0.05, random_state=22)

# Step 6: Scale Numerical Features
numerical_cols = ['Price', 'Discount', 'Demand Forecast', 'Competitor Pricing', 
                  'Discounted Price', 'Price Difference', 'Stock to Order Ratio', 'Forecast Accuracy']

scaler = StandardScaler()

# Fit and transform on training data, transform on validation
X_train_scaled = scaler.fit_transform(X_train[numerical_cols])
X_val_scaled = scaler.transform(X_val[numerical_cols])

# Convert scaled arrays back to DataFrames (preserve column names + indices)
X_train_scaled = pd.DataFrame(X_train_scaled, columns=numerical_cols, index=X_train.index)
X_val_scaled = pd.DataFrame(X_val_scaled, columns=numerical_cols, index=X_val.index)

# Concatenate numerical (scaled) and categorical (unchanged)
X_train_final = pd.concat([X_train_scaled, X_train.drop(columns=numerical_cols)], axis=1)
X_val_final = pd.concat([X_val_scaled, X_val.drop(columns=numerical_cols)], axis=1)

# Models to train
models = [
    ("Linear Regression", LinearRegression()),
    ("Lasso Regression", Lasso(alpha=0.1)),
    ("Ridge Regression", Ridge(alpha=1.0))
]

# Initialize lists for storing MAE for train and validation sets
train_errors = []
val_errors = []

# Train and evaluate each model
for name, model in models:
    print(f'Training {name}...')

    # Fit the model
    model.fit(X_train_final, Y_train)

    # Predictions on training data
    train_preds = model.predict(X_train_final)
    train_error = mae(Y_train, train_preds)
    train_errors.append(train_error)
    print(f'Training Error (MAE): {train_error:.4f}')

    # Predictions on validation data
    val_preds = model.predict(X_val_final)
    val_error = mae(Y_val, val_preds)
    val_errors.append(val_error)
    print(f'Validation Error (MAE): {val_error:.4f}')
    print()

# Visualization: Bar Plot of MAE
fig, ax = plt.subplots(figsize=(10, 6))

# Create a bar plot
bar_width = 0.35
index = range(len(models))

bar1 = ax.bar(index, train_errors, bar_width, label='Training MAE', color='skyblue')
bar2 = ax.bar([i + bar_width for i in index], val_errors, bar_width, label='Validation MAE', color='salmon')

# Adding labels and title
ax.set_xlabel('Model')
ax.set_ylabel('Mean Absolute Error (MAE)')
ax.set_title('Comparison of MAE for Different Models')
ax.set_xticks([i + bar_width / 2 for i in index])
ax.set_xticklabels([model[0] for model in models])
ax.legend()

# Show plot using Streamlit
st.pyplot(fig)  # This will render the plot in Streamlit
