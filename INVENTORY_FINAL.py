import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error as mae

# Step 1: Load the DataFrame
try:
    df = pd.read_csv('retail_store_inventory.csv')
    print(df.columns.tolist())
    st.success("✅ DataFrame loaded successfully.")
except FileNotFoundError:
    st.error("❌ Error: The file was not found. Please check the file path.")
    st.stop()

# 🛠 Safe Date Handling
if 'Date' in df.columns:
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')  # If wrong format, put NaT
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    print("Date features created successfully.")
else:
    print("⚠️ Warning: 'Date' column not found in DataFrame. Skipping date features.")


# Step 3: Feature Engineering
df['Discounted Price'] = df['Price'] * (1 - df['Discount'] / 100)
df['Price Difference'] = df['Price'] - df['Competitor Pricing']
df['Stock to Order Ratio'] = df['Inventory Level'] / (df['Units Ordered'] + 1)
df['Forecast Accuracy'] = abs(df['Demand Forecast'] - df['Units Sold']) / (df['Units Sold'] + 1)

# Step 4: Target for Regression (Units Sold)
target = df['Units Sold']

# Step 5: Feature Selection
feature_cols = [
    'Price', 'Discount', 'Demand Forecast', 'Competitor Pricing',
    'Discounted Price', 'Price Difference', 'Stock to Order Ratio',
    'Forecast Accuracy', 'Holiday/Promotion'
]

# Add Year, Month, Day if they exist
for time_feature in ['Year', 'Month', 'Day']:
    if time_feature in df.columns:
        feature_cols.append(time_feature)

# Then proceed
X = df[feature_cols]
y = df['Demand Class']




# Step 6: Train-Test Split
X_train, X_val, y_train, y_val = train_test_split(X, target, test_size=0.1, random_state=22)

# Step 7: Scaling Numerical Features
numerical_cols = ['Price', 'Discount', 'Demand Forecast', 'Competitor Pricing',
                  'Discounted Price', 'Price Difference', 'Stock to Order Ratio',
                  'Forecast Accuracy']

scaler = StandardScaler()

X_train_num_scaled = scaler.fit_transform(X_train[numerical_cols])
X_val_num_scaled = scaler.transform(X_val[numerical_cols])

# Replace scaled numerical columns
X_train_scaled = X_train.copy()
X_val_scaled = X_val.copy()

X_train_scaled[numerical_cols] = X_train_num_scaled
X_val_scaled[numerical_cols] = X_val_num_scaled

# Step 8: Models - Linear, Lasso, Ridge
models = [
    ("Linear Regression", LinearRegression()),
    ("Lasso Regression", Lasso(alpha=0.1)),
    ("Ridge Regression", Ridge(alpha=1.0))
]

# Collecting errors
train_errors = []
val_errors = []

for name, model in models:
    model.fit(X_train_scaled, y_train)
    y_train_pred = model.predict(X_train_scaled)
    y_val_pred = model.predict(X_val_scaled)

    train_mae = mae(y_train, y_train_pred)
    val_mae = mae(y_val, y_val_pred)

    train_errors.append(train_mae)
    val_errors.append(val_mae)

    st.write(f"**{name}** - Train MAE: {train_mae:.2f}, Validation MAE: {val_mae:.2f}")

# Step 9: Visualization
fig, ax = plt.subplots(figsize=(10,6))

bar_width = 0.35
index = range(len(models))

bar1 = ax.bar(index, train_errors, bar_width, label='Training MAE', color='skyblue')
bar2 = ax.bar([i + bar_width for i in index], val_errors, bar_width, label='Validation MAE', color='salmon')

ax.set_xlabel('Model')
ax.set_ylabel('Mean Absolute Error')
ax.set_title('Model Comparison - Training vs Validation MAE')
ax.set_xticks([i + bar_width/2 for i in index])
ax.set_xticklabels([name for name, _ in models])
ax.legend()

st.pyplot(fig)
