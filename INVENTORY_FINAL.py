import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error as mae

# Step 1: Load Data
try:
    df = pd.read_csv('retail_store_inventory.csv')  # Your CSV file
    print("✅ DataFrame loaded successfully.")
except FileNotFoundError:
    print("❌ Error: File not found.")
    exit()

print("🧾 Columns in the DataFrame:", df.columns.tolist())

# Step 2: Handle Date and Time Features
if 'Date' in df.columns:
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    print("✅ Date features created: Year, Month, Day.")
else:
    print("⚠️ Warning: 'Date' column missing. Skipping Date feature creation.")

# Step 3: Feature Engineering
if all(col in df.columns for col in ['Price', 'Discount', 'Competitor Pricing']):
    df['Discounted Price'] = df['Price'] * (1 - df['Discount'] / 100)
    df['Price Difference'] = df['Price'] - df['Competitor Pricing']
else:
    print("⚠️ Warning: Some columns missing for price features!")

if all(col in df.columns for col in ['Inventory Level', 'Units Ordered']):
    df['Stock to Order Ratio'] = df['Inventory Level'] / (df['Units Ordered'] + 1)

if all(col in df.columns for col in ['Demand Forecast', 'Units Sold']):
    df['Forecast Accuracy'] = abs(df['Demand Forecast'] - df['Units Sold']) / (df['Units Sold'] + 1)

# Step 4: Demand Class creation
def classify_units(units):
    if units <= 50:
        return 0
    elif units <= 150:
        return 1
    else:
        return 2

if 'Units Sold' in df.columns:
    df['Demand Class'] = df['Units Sold'].apply(classify_units)
else:
    print("⚠️ Warning: 'Units Sold' missing. Cannot create Demand Class.")

# Step 5: Feature Selection
feature_cols = [
    'Price', 'Discount', 'Demand Forecast', 'Competitor Pricing',
    'Discounted Price', 'Price Difference', 'Stock to Order Ratio',
    'Forecast Accuracy', 'Holiday/Promotion'
]

# Only add Year, Month, Day if they exist
for time_feature in ['Year', 'Month', 'Day']:
    if time_feature in df.columns:
        feature_cols.append(time_feature)

# Also include one-hot encoded categorical columns dynamically
feature_cols += [col for col in df.columns if
                 'Category_' in col or 'Region_' in col or
                 'Weather Condition_' in col or 'Seasonality_' in col]

# Step 6: Safely select Features
missing_features = [col for col in feature_cols if col not in df.columns]
if missing_features:
    print(f"⚠️ Warning: Missing columns {missing_features} will be ignored.")
    feature_cols = [col for col in feature_cols if col in df.columns]

X = df[feature_cols]
y = df['Demand Class']

# Step 7: Train-Test Split
X_train, X_val, Y_train, Y_val = train_test_split(X, y, test_size=0.05, random_state=22)

# Step 8: Scale Numerical Features
numerical_cols = ['Price', 'Discount', 'Demand Forecast', 'Competitor Pricing',
                  'Discounted Price', 'Stock to Order Ratio', 'Inventory Level',
                  'Units Ordered']

numerical_cols = [col for col in numerical_cols if col in X_train.columns]

scaler = StandardScaler()
X_train_num = X_train[numerical_cols]
X_val_num = X_val[numerical_cols]

X_train_num_scaled = scaler.fit_transform(X_train_num)
X_val_num_scaled = scaler.transform(X_val_num)

X_train_scaled = pd.DataFrame(X_train_num_scaled, columns=numerical_cols, index=X_train.index)
X_val_scaled = pd.DataFrame(X_val_num_scaled, columns=numerical_cols, index=X_val.index)

# Combine scaled numerical + categorical features
categorical_cols = [col for col in X_train.columns if col not in numerical_cols]

X_train_cat = X_train[categorical_cols]
X_val_cat = X_val[categorical_cols]

X_train_final = pd.concat([X_train_scaled, X_train_cat], axis=1)
X_val_final = pd.concat([X_val_scaled, X_val_cat], axis=1)

print("✅ Final datasets ready for modeling!")

# Step 9: Models Training
models = [
    ("Linear Regression", LinearRegression()),
    ("Lasso Regression", Lasso(alpha=0.1)),
    ("Ridge Regression", Ridge(alpha=1.0))
]

for name, model in models:
    print(f"🔵 Training {name}...")

    model.fit(X_train_scaled, Y_train)

    train_preds = model.predict(X_train_scaled)
    train_error = mae(Y_train, train_preds)
    print(f"Training Error (MAE): {train_error:.4f}")

    val_preds = model.predict(X_val_scaled)
    val_error = mae(Y_val, val_preds)
    print(f"Validation Error (MAE): {val_error:.4f}")
    print()
