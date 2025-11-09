#!/usr/bin/env python3
"""
Pack SGA - Simple Model Retraining Script
Quickly retrain model with new data
"""

import pandas as pd
import numpy as np
import pickle
import sys
import argparse
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

def load_data(data_path):
    """Load training data"""
    try:
        if data_path.endswith('.csv'):
            df = pd.read_csv(data_path)
        elif data_path.endswith('.xlsx'):
            df = pd.read_excel(data_path)
        else:
            print("❌ Error: Data file must be .csv or .xlsx")
            sys.exit(1)
        
        print(f"✓ Loaded {len(df)} records from {data_path}")
        return df
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        sys.exit(1)

def prepare_features(df):
    """Prepare features from raw data"""
    print("✓ Preparing features...")
    
    # Extract date features if FullDate exists
    if 'FullDate' in df.columns:
        df['FullDate'] = pd.to_datetime(df['FullDate'])
        df['Year'] = df['FullDate'].dt.year
        df['Month'] = df['FullDate'].dt.month
        df['Day'] = df['FullDate'].dt.day
        df['DayOfWeek'] = df['FullDate'].dt.dayofweek
        df['Quarter'] = df['FullDate'].dt.quarter
    
    # Separate features and target
    target_col = '%Pack'
    exclude_cols = ['FullDate', '%Pack', 'Pack (Ton)']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    X = df[feature_cols].copy()
    y = df[target_col].copy()
    
    print(f"  Features: {len(feature_cols)}")
    print(f"  Target: {target_col}")
    
    return X, y, feature_cols

def encode_and_scale(X, label_encoders=None, scaler=None, fit=True):
    """Encode categorical and scale features"""
    print("✓ Encoding and scaling features...")
    
    # Identify categorical columns
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    
    # Encode categorical variables
    if fit:
        label_encoders = {}
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            label_encoders[col] = le
            print(f"  Encoded {col}: {len(le.classes_)} classes")
    else:
        for col in categorical_cols:
            X[col] = label_encoders[col].transform(X[col].astype(str))
    
    # Store feature names
    feature_names = X.columns.tolist()
    
    # Scale features
    if fit:
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = scaler.transform(X)
    
    X_scaled = pd.DataFrame(X_scaled, columns=feature_names)
    
    return X_scaled, label_encoders, scaler, categorical_cols, feature_names

def train_model(X_train, y_train, X_test, y_test, model_params=None):
    """Train Random Forest model"""
    print("✓ Training Random Forest model...")
    
    if model_params is None:
        model_params = {
            'n_estimators': 100,
            'max_depth': 20,
            'min_samples_split': 5,
            'random_state': 42,
            'n_jobs': -1
        }
    
    model = RandomForestRegressor(**model_params)
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
    mae_test = mean_absolute_error(y_test, y_pred_test)
    
    print(f"  Train R²: {r2_train:.4f}")
    print(f"  Test R²:  {r2_test:.4f}")
    print(f"  RMSE:     {rmse_test:.4f}")
    print(f"  MAE:      {mae_test:.4f}")
    
    return model, {
        'r2_train': r2_train,
        'r2_test': r2_test,
        'rmse': rmse_test,
        'mae': mae_test
    }

def save_model(model, scaler, label_encoders, feature_names, categorical_cols, 
               metrics, output_path='pack_model.pkl'):
    """Save trained model package"""
    print(f"✓ Saving model to {output_path}...")
    
    model_package = {
        'model': model,
        'scaler': scaler,
        'label_encoders': label_encoders,
        'feature_names': feature_names,
        'categorical_cols': categorical_cols,
        'best_model_name': 'Random Forest',
        'metrics': metrics,
        'trained_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    with open(output_path, 'wb') as f:
        pickle.dump(model_package, f)
    
    print("✓ Model saved successfully")

def main():
    parser = argparse.ArgumentParser(
        description='Pack SGA - Simple model retraining',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python retrain_simple.py --data Pack_SGA_2425.xlsx --output new_model.pkl
        """
    )
    
    parser.add_argument('--data', required=True,
                       help='Path to training data (CSV or Excel)')
    parser.add_argument('--output', default='pack_model_new.pkl',
                       help='Output model file (default: pack_model_new.pkl)')
    parser.add_argument('--test-size', type=float, default=0.2,
                       help='Test set proportion (default: 0.2)')
    parser.add_argument('--n-estimators', type=int, default=100,
                       help='Number of trees (default: 100)')
    parser.add_argument('--max-depth', type=int, default=20,
                       help='Max tree depth (default: 20)')
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("Pack SGA - Model Retraining System")
    print("="*70 + "\n")
    
    # Load data
    df = load_data(args.data)
    
    if '%Pack' not in df.columns:
        print("❌ Error: Data must contain '%Pack' column")
        sys.exit(1)
    
    # Check missing values
    missing = df.isnull().sum().sum()
    if missing > 0:
        print(f"⚠️  Warning: {missing} missing values detected - removing...")
        df = df.dropna()
        print(f"  Cleaned to {len(df)} records")
    
    # Prepare features
    X, y, feature_cols = prepare_features(df)
    
    # Encode and scale
    X_scaled, label_encoders, scaler, categorical_cols, feature_names = \
        encode_and_scale(X, fit=True)
    
    # Train-test split
    print(f"✓ Splitting data ({int((1-args.test_size)*100)}/{int(args.test_size*100)})...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=args.test_size, random_state=42
    )
    print(f"  Train: {len(X_train)} samples")
    print(f"  Test:  {len(X_test)} samples")
    
    # Train model
    model_params = {
        'n_estimators': args.n_estimators,
        'max_depth': args.max_depth,
        'min_samples_split': 5,
        'random_state': 42,
        'n_jobs': -1
    }
    model, metrics = train_model(X_train, y_train, X_test, y_test, model_params)
    
    # Save model
    save_model(model, scaler, label_encoders, feature_names, categorical_cols,
               metrics, args.output)
    
    # Summary
    print("\n" + "="*70)
    print("RETRAINING COMPLETE")
    print("="*70)
    print(f"Model saved:        {args.output}")
    print(f"Training samples:   {len(X_train)}")
    print(f"Test samples:       {len(X_test)}")
    print(f"Test R² Score:      {metrics['r2_test']:.4f}")
    print(f"Test RMSE:          {metrics['rmse']:.4f}")
    print(f"Test MAE:           {metrics['mae']:.4f}")
    
    if metrics['r2_test'] >= 0.75:
        print("\n✅ Model performance: Good!")
    elif metrics['r2_test'] >= 0.65:
        print("\n⚠️  Model performance: Fair - consider more data or features")
    else:
        print("\n❌ Model performance: Poor - review data quality")
    
    print("="*70 + "\n")

if __name__ == '__main__':
    main()
