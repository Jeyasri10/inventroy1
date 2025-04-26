import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error as mae

# Step 1: Load the DataFrame (Ensure the path to your CSV is correct)
try:
    df = pd.read_csv('retail_store_inventory.csv')  # Replace with your actual dataset path
    print("DataFrame loaded successfully.")
except FileNotFoundError:
    print("Error: The file was not found. Please check the file path.")
    exit()

# Step 2: Define the list of columns you want to check
feature_cols = ['col1', 'col2', 'col3']  # Replace with your actual column names

# Step 3: Check if the columns are missing in the DataFrame
missing_cols = [col for col in feature_cols if col not in df.columns]

# Step 4: Handle missing columns
if missing_cols:
    print(f"Missing columns: {missing_cols}")
else:
    print("All specified columns are present in the DataFrame.")

# Step 5: Optionally, perform operations on your DataFrame
# Example operation: Display the first few rows of the DataFrame
print(df.head())

# 📆 Step 2: Convert Date column and extract time-based features
df['Date'] = pd.to_datetime(df['Date'])
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day

# 🧮 Step 4: Feature Engineering
df['Discounted Price'] = df['Price'] * (1 - df['Discount'] / 100)
df['Price Difference'] = df['Price'] - df['Competitor Pricing']
df['Stock to Order Ratio'] = df['Inventory Level'] / (df['Units Ordered'] + 1)
df['Forecast Accuracy'] = abs(df['Demand Forecast'] - df['Units Sold']) / (df['Units Sold'] + 1)

# 🧠 Step 5: Target Creation - Classification
def classify_units(units):
    if units <= 50:
        return 0  # Low
    elif units <= 150:
        return 1  # Medium
    else:
        return 2  # High

df['Demand Class'] = df['Units Sold'].apply(classify_units)

# ✅ Step 6: Feature Selection
feature_cols = [
    'Price', 'Discount', 'Demand Forecast', 'Competitor Pricing',
    'Discounted Price', 'Price Difference', 'Stock to Order Ratio',
    'Forecast Accuracy', 'Holiday/Promotion', 'Year', 'Month', 'Day'
] + [col for col in df.columns if 'Category_' in col or 'Region_' in col or
      'Weather Condition_' in col or 'Seasonality_' in col]

X = df[feature_cols]
y = df['Demand Class']
X, y

# ------------------------------- 
# 🧹 Step 1: Define Feature Groups
# -------------------------------
numerical_cols = ['Price', 'Discount', 'Demand Forecast', 'Competitor Pricing', 
                  'Discounted Price', 'Stock to Order Ratio', 'Inventory Level',
                  'Units Ordered']

categorical_cols = [col for col in df.columns if 'Category_' in col or 
                    'Region_' in col or 'Weather Condition_' in col or 
                    'Seasonality_' in col]

# ------------------------------------
# 🧼 Step 2: Drop Target & Non-Features
# ------------------------------------
drop_cols = [col for col in ['Units Sold', 'Year', 'Date'] if col in df.columns]
features = df.drop(columns=drop_cols)
target = df['Units Sold'].values  # Your prediction target

# ------------------------------------
# ✂️ Step 3: Train-Test Split
# ------------------------------------
X_train, X_val, Y_train, Y_val = train_test_split(features, target, test_size=0.05, random_state=22)

# ------------------------------------
# 📏 Step 4: Scale Numerical Features
# ------------------------------------
scaler = StandardScaler()
X_train_num = X_train[numerical_cols]
X_val_num = X_val[numerical_cols]

X_train_num_scaled = scaler.fit_transform(X_train_num)
X_val_num_scaled = scaler.transform(X_val_num)

X_train_scaled = pd.DataFrame(X_train_num_scaled, columns=numerical_cols, index=X_train.index)
X_val_scaled = pd.DataFrame(X_val_num_scaled, columns=numerical_cols, index=X_val.index)

# ------------------------------------
# 🧩 Step 5: Combine with Categorical
# ------------------------------------
X_train_cat = X_train[categorical_cols]
X_val_cat = X_val[categorical_cols]

X_train_final = pd.concat([X_train_scaled, X_train_cat], axis=1)
X_val_final = pd.concat([X_val_scaled, X_val_cat], axis=1)

# ✅ Done! Final datasets ready for modeling
print("🔍 Preview of Final Processed Train Set:")
print(X_train_final.head())

# ------------------------------
# Linear Regression Model
# ------------------------------
model = LinearRegression()

# Train the model
print(f'Training Linear Regression...')

# Fit the model
model.fit(X_train_scaled, Y_train)

# Predictions on training data
train_preds = model.predict(X_train_scaled)
train_error = mae(Y_train, train_preds)
print(f'Training Error (MAE): {train_error:.4f}')

# Predictions on validation data
val_preds = model.predict(X_val_scaled)
val_error = mae(Y_val, val_preds)
print(f'Validation Error (MAE): {val_error:.4f}')

# Visualization: Bar Plot of MAE for Linear Regression
fig, ax = plt.subplots(figsize=(8, 6))

# Create a bar plot for Linear Regression's training and validation errors
bar_width = 0.35
index = [0]

bar1 = ax.bar(index, train_error, bar_width, label='Training MAE', color='skyblue')
bar2 = ax.bar([i + bar_width for i in index], val_error, bar_width, label='Validation MAE', color='salmon')

# Adding labels and title
ax.set_xlabel('Model')
ax.set_ylabel('Mean Absolute Error (MAE)')
ax.set_title('Linear Regression - MAE')
ax.set_xticks([i + bar_width / 2 for i in index])
ax.set_xticklabels(['Linear Regression'])
ax.legend()

# Show the plot
plt.tight_layout()
plt.show()
