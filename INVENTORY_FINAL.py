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
 @@ -13,35 +15,13 @@
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
 # Step 2: Feature Engineering
 df['Discounted Price'] = df['Price'] * (1 - df['Discount'] / 100)
 df['Price Difference'] = df['Price'] - df['Competitor Pricing']
 df['Stock to Order Ratio'] = df['Inventory Level'] / (df['Units Ordered'] + 1)
 df['Forecast Accuracy'] = abs(df['Demand Forecast'] - df['Units Sold']) / (df['Units Sold'] + 1)
 
 # 🧠 Step 5: Target Creation - Classification
 # Step 3: Target Creation - Classification
 def classify_units(units):
     if units <= 50:
         return 0  # Low
 @@ -52,7 +32,7 @@ def classify_units(units):
 
 df['Demand Class'] = df['Units Sold'].apply(classify_units)
 
 # ✅ Step 6: Feature Selection
 # Step 4: Feature Selection
 feature_cols = [
     'Price', 'Discount', 'Demand Forecast', 'Competitor Pricing',
     'Discounted Price', 'Price Difference', 'Stock to Order Ratio',
 @@ -62,74 +42,36 @@ def classify_units(units):
 
 X = df[feature_cols]
 y = df['Demand Class']
 X, y
 
 # -------------------------------
 # 🧹 Step 1: Define Feature Groups
 # -------------------------------
 # Define numerical columns based on your dataset
 numerical_cols = ['Price', 'Discount', 'Demand Forecast', 'Competitor Pricing',
                   'Discounted Price', 'Stock to Order Ratio', 'Inventory Level',
                   'Units Ordered']
 
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
 scaler = StandardScaler()
 X_train_num = X_train[numerical_cols]
 X_val_num = X_val[numerical_cols]
 
 X_train_num_scaled = scaler.fit_transform(X_train_num)
 X_val_num_scaled = scaler.transform(X_val_num)
 # Step 5: Train-Test Split
 X_train, X_val, Y_train, Y_val = train_test_split(X, y, test_size=0.05, random_state=22)
 
 X_train_scaled = pd.DataFrame(X_train_num_scaled, columns=numerical_cols, index=X_train.index)
 X_val_scaled = pd.DataFrame(X_val_num_scaled, columns=numerical_cols, index=X_val.index)
 # Step 6: Scale Numerical Features
 numerical_cols = ['Price', 'Discount', 'Demand Forecast', 'Competitor Pricing', 
                   'Discounted Price', 'Price Difference', 'Stock to Order Ratio', 'Forecast Accuracy']
 
 # ------------------------------------
 # 🧩 Step 5: Combine with Categorical
 # ------------------------------------
 X_train_cat = X_train[categorical_cols]
 X_val_cat = X_val[categorical_cols]
 scaler = StandardScaler()
 
 X_train_final = pd.concat([X_train_scaled, X_train_cat], axis=1)
 X_val_final = pd.concat([X_val_scaled, X_val_cat], axis=1)
 # Fit and transform on training data, transform on validation
 X_train_scaled = scaler.fit_transform(X_train[numerical_cols])
 X_val_scaled = scaler.transform(X_val[numerical_cols])
 
 # ✅ Done! Final datasets ready for modeling
 print("🔍 Preview of Final Processed Train Set:")
 print(X_train_final.head())
 # Convert scaled arrays back to DataFrames (preserve column names + indices)
 X_train_scaled = pd.DataFrame(X_train_scaled, columns=numerical_cols, index=X_train.index)
 X_val_scaled = pd.DataFrame(X_val_scaled, columns=numerical_cols, index=X_val.index)
 
 print("\n🔍 Preview of Final Processed Validation Set:")
 print(X_val_final.head())
 # Concatenate numerical (scaled) and categorical (unchanged)
 X_train_final = pd.concat([X_train_scaled, X_train.drop(columns=numerical_cols)], axis=1)
 X_val_final = pd.concat([X_val_scaled, X_val.drop(columns=numerical_cols)], axis=1)
 
 # ------------------------------
 # Models: Linear, Lasso, Ridge
 # ------------------------------
 # Models to train
 models = [
     ("Linear Regression", LinearRegression()),
     ("Lasso Regression", Lasso(alpha=0.1)),
     ("Ridge Regression", Ridge(alpha=1.0))
 ]
 
 # Initialize lists for storing MAE for train and validation sets
 train_errors = []
 val_errors = []
 
 @@ -138,27 +80,22 @@ def classify_units(units):
     print(f'Training {name}...')
 
     # Fit the model
     model.fit(X_train_scaled, Y_train)
     model.fit(X_train_final, Y_train)
 
     # Predictions on training data
     train_preds = model.predict(X_train_scaled)
     train_preds = model.predict(X_train_final)
     train_error = mae(Y_train, train_preds)
     train_errors.append(train_error)
     print(f'Training Error (MAE): {train_error:.4f}')
 
     # Predictions on validation data
     val_preds = model.predict(X_val_scaled)
     val_preds = model.predict(X_val_final)
     val_error = mae(Y_val, val_preds)
     val_errors.append(val_error)
 
     print(f'Training Error (MAE): {train_error:.4f}')
     print(f'Validation Error (MAE): {val_error:.4f}')
     print()
 
 # ----------------------------------------
 # Visualization: Bar Plot of MAE
 # ----------------------------------------
 
 # Plotting the MAE of each model for both training and validation sets
 fig, ax = plt.subplots(figsize=(10, 6))
 
 # Create a bar plot
 @@ -176,6 +113,5 @@ def classify_units(units):
 ax.set_xticklabels([model[0] for model in models])
 ax.legend()
 
 # Show plot
 plt.tight_layout()
 plt.show()
 # Show plot using Streamlit
 st.pyplot(fig)  # This will render the plot in Streamlit
