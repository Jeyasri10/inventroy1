import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns



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
    # You can add further handling here, like adding the missing columns with default values
    # Example: df[missing_cols] = 0  # Adding missing columns with default value 0
else:
    print("All specified columns are present in the DataFrame.")

# Step 5: Optionally, perform operations on your DataFrame
# Example operation: Display the first few rows of the DataFrame
print(df.head())

# Step 6: Save the modified DataFrame (if needed)
# If you made changes and want to save the DataFrame to a new CSV file




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
X,y

# Plot histograms of all numerical features
numerical_cols = ['Price', 'Units Sold', 'Units Ordered', 'Demand Forecast', 'Discount',
                  'Inventory Level', 'Competitor Pricing', 'Discounted Price']

df[numerical_cols].hist(figsize=(12, 8), bins=30, color='skyblue', edgecolor='black')
plt.suptitle('Histograms of Numerical Features', fontsize=16)
plt.tight_layout()
plt.show()


# Correlation heatmap for numerical columns
import seaborn as sns

correlation_matrix = df[numerical_cols].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Heatmap of Numerical Features', fontsize=16)
plt.tight_layout()
plt.show()


# Box plots to show the relationship between numerical features and the target 'Demand Class'
plt.figure(figsize=(12, 8))
sns.boxplot(x='Demand Class', y='Price', data=df)
plt.title('Price Distribution by Demand Class')
plt.show()

# Same for other features
sns.boxplot(x='Demand Class', y='Discounted Price', data=df)
plt.title('Discounted Price Distribution by Demand Class')
plt.show()


# Group by Demand Class and calculate mean for each feature
df_grouped = df.groupby('Demand Class')[numerical_cols].mean()

# Visualize the distribution of numerical features by Demand Class
df_grouped.plot(kind='bar', figsize=(12, 8), colormap='viridis')
plt.title('Mean of Numerical Features by Demand Class')
plt.ylabel('Mean Value')
plt.xlabel('Demand Class')
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()



# Convert 'Date' column to datetime format (if it's not already)
df['Date'] = pd.to_datetime(df['Date'])

# Plotting trends over time for some numerical features
df.groupby(df['Date'].dt.month)['Units Sold'].mean().plot(kind='line', color='skyblue')
plt.title('Average Units Sold per Month')
plt.ylabel('Average Units Sold')
plt.tight_layout()
plt.show()


from scipy import stats

# Calculate z-scores for numerical columns to detect outliers
z_scores = stats.zscore(df[numerical_cols])
abs_z_scores = abs(z_scores)
outliers = (abs_z_scores > 3)  # Consider values with z-score > 3 as outliers

print(f'Number of outliers in the dataset: {outliers.sum()}')

# Plot boxplots for each numerical feature
for col in numerical_cols:
    plt.figure(figsize=(8, 4))
    sns.boxplot(x=df[col])
    plt.title(f'Boxplot for {col}')
    plt.tight_layout()
    plt.show()


df = df[(df['Units Sold'] < 140) & (df['Price'] > 10)]
df

X = df[feature_cols]
X

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# -------------------------------
# 🧹 Step 1: Define Feature Groups
# -------------------------------

# Define numerical columns based on your dataset
numerical_cols = ['Price', 'Discount', 'Demand Forecast', 'Competitor Pricing',
                  'Discounted Price', 'Stock to Order Ratio', 'Inventory Level',
                  'Units Ordered']  # Add 'Is High Season' only if it exists

# Dynamically fetch one-hot encoded categorical columns
categorical_cols = [col for col in df.columns if
                    'Category_' in col or
                    'Region_' in col or
                    'Weather Condition_' in col or
                    'Seasonality_' in col]

# ------------------------------------
# 🧼 Step 2: Drop Target & Non-Features
# ------------------------------------

# Drop target and time-related columns, if they exist
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

# Initialize the scaler
scaler = StandardScaler()

# Select only the numerical columns from train/val sets
X_train_num = X_train[numerical_cols]
X_val_num = X_val[numerical_cols]

# Fit and transform on training data, transform on validation
X_train_num_scaled = scaler.fit_transform(X_train_num)
X_val_num_scaled = scaler.transform(X_val_num)

# Convert scaled arrays back to DataFrames (preserve column names + indices)
X_train_scaled = pd.DataFrame(X_train_num_scaled, columns=numerical_cols, index=X_train.index)
X_val_scaled = pd.DataFrame(X_val_num_scaled, columns=numerical_cols, index=X_val.index)

# ------------------------------------
# 🧩 Step 5: Combine with Categorical
# ------------------------------------

# Select categorical columns (already one-hot encoded)
X_train_cat = X_train[categorical_cols]
X_val_cat = X_val[categorical_cols]

# Concatenate numerical (scaled) and categorical (unchanged)
X_train_final = pd.concat([X_train_scaled, X_train_cat], axis=1)
X_val_final = pd.concat([X_val_scaled, X_val_cat], axis=1)

# ✅ Done! Final datasets ready for modeling
print("🔍 Preview of Final Processed Train Set:")
print(X_train_final.head())

print("\n🔍 Preview of Final Processed Validation Set:")
print(X_val_final.head())


from sklearn.metrics import mean_absolute_error as mae

models = [
    ("Linear Regression", LinearRegression()),
    ("Decision Tree", DecisionTreeRegressor(random_state=42)),
    ("Random Forest", RandomForestRegressor(random_state=42)),
    ("Lasso Regression", Lasso(alpha=0.1)),
    ("Ridge Regression", Ridge(alpha=1.0))
]

# Train and evaluate each model
for name, model in models:
    print(f'Training {name}...')

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
    print()
