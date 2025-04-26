import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error as mae

# Step 1: Load the DataFrame (Ensure the path to your CSV is correct)
try:
    df = pd.read_csv('retail_store_inventory.csv')  # Replace with your actual dataset path
    st.write("DataFrame loaded successfully.")
except FileNotFoundError:
    st.write("Error: The file was not found. Please check the file path.")
    exit()

# Step 2: Feature Engineering (Create additional features for better analysis)
df['Discounted Price'] = df['Price'] * (1 - df['Discount'] / 100)
df['Price Difference'] = df['Price'] - df['Competitor Pricing']
df['Stock to Order Ratio'] = df['Inventory Level'] / (df['Units Ordered'] + 1)
df['Forecast Accuracy'] = abs(df['Demand Forecast'] - df['Units Sold']) / (df['Units Sold'] + 1)

# Step 3: Target Creation
def classify_units(units):
    if units <= 50:
        return 0  # Low
    elif units <= 150:
        return 1  # Medium
    else:
        return 2  # High

df['Demand Class'] = df['Units Sold'].apply(classify_units)

# Check and display columns
st.write("Columns in the DataFrame:")
st.write(df.columns)

# Define feature columns based on the actual columns in your CSV
feature_cols = ['Price', 'Discount', 'Demand Forecast', 'Competitor Pricing',
                'Discounted Price', 'Price Difference', 'Stock to Order Ratio',
                'Forecast Accuracy', 'Holiday/Promotion', 'Year', 'Month', 'Day']

# Check for missing columns
missing_cols = [col for col in feature_cols if col not in df.columns]
if missing_cols:
    st.write(f"Missing columns: {missing_cols}")

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

# Models to compare
models = [
    ("Linear Regression", LinearRegression()),
    ("Lasso Regression", Lasso(alpha=0.1)),
    ("Ridge Regression", Ridge(alpha=1.0))
]

# Initialize lists to store MAE values
train_errors = []
val_errors = []

# Train and evaluate each model
for name, model in models:
    st.write(f'Training {name}...')

    # Fit the model
    model.fit(X_train_final, Y_train)

    # Predictions on training data
    train_preds = model.predict(X_train_final)
    train_error = mae(Y_train, train_preds)
    train_errors.append(train_error)

    # Predictions on validation data
    val_preds = model.predict(X_val_final)
    val_error = mae(Y_val, val_preds)
    val_errors.append(val_error)

# Visualization: Bar Plot for MAE of all models
fig, ax = plt.subplots(figsize=(10, 6))

bar_width = 0.25
index = range(len(models))

bar1 = ax.bar(index, train_errors, bar_width, label='Training MAE', color='skyblue')
bar2 = ax.bar([i + bar_width for i in index], val_errors, bar_width, label='Validation MAE', color='salmon')

# Adding labels and title
ax.set_xlabel('Model')
ax.set_ylabel('Mean Absolute Error (MAE)')
ax.set_title('MAE Comparison: Linear Regression, Lasso, and Ridge')
ax.set_xticks([i + bar_width for i in index])
ax.set_xticklabels([name for name, _ in models])
ax.legend()

# Show plot using Streamlit
st.pyplot(fig)  # This will render the plot in Streamlit
