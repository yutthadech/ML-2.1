#!/usr/bin/env python3
"""
Pack SGA - Model Testing Script
Test model accuracy and generate performance report
"""

import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import cross_val_score
import sys
import argparse

def load_model(model_path='pack_model.pkl'):
    """Load trained model package"""
    try:
        with open(model_path, 'rb') as f:
            model_package = pickle.load(f)
        print("✓ Model loaded successfully")
        return model_package
    except FileNotFoundError:
        print(f"❌ Error: Model file '{model_path}' not found")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        sys.exit(1)

def load_data(data_path):
    """Load test data"""
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

def prepare_data(df, model_package):
    """Prepare data for testing"""
    scaler = model_package['scaler']
    label_encoders = model_package['label_encoders']
    feature_names = model_package['feature_names']
    categorical_cols = model_package['categorical_cols']
    
    # Extract date features if needed
    if 'FullDate' in df.columns:
        df['FullDate'] = pd.to_datetime(df['FullDate'])
        df['Year'] = df['FullDate'].dt.year
        df['Month'] = df['FullDate'].dt.month
        df['Day'] = df['FullDate'].dt.day
        df['DayOfWeek'] = df['FullDate'].dt.dayofweek
        df['Quarter'] = df['FullDate'].dt.quarter
    
    # Separate features and target
    X = df[feature_names].copy()
    y = df['%Pack'].values
    
    # Encode categorical variables
    for col in categorical_cols:
        X[col] = label_encoders[col].transform(X[col].astype(str))
    
    # Scale features
    X_scaled = scaler.transform(X)
    
    return X_scaled, y

def evaluate_model(model, X, y, cv=5):
    """Evaluate model performance"""
    # Predictions
    y_pred = model.predict(X)
    
    # Metrics
    r2 = r2_score(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    mae = mean_absolute_error(y, y_pred)
    
    # Cross-validation
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring='r2', n_jobs=-1)
    
    return {
        'r2': r2,
        'rmse': rmse,
        'mae': mae,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'predictions': y_pred
    }

def print_report(results, model_name, n_samples):
    """Print evaluation report"""
    print("\n" + "="*70)
    print("MODEL PERFORMANCE REPORT")
    print("="*70)
    print(f"\nModel:              {model_name}")
    print(f"Test Samples:       {n_samples}")
    print("\n" + "-"*70)
    print("Performance Metrics:")
    print("-"*70)
    print(f"R² Score:           {results['r2']:.4f}")
    print(f"RMSE:               {results['rmse']:.4f}")
    print(f"MAE:                {results['mae']:.4f}")
    print(f"CV R² (mean±std):   {results['cv_mean']:.4f} ± {results['cv_std']:.4f}")
    
    # Predictions summary
    y_pred = results['predictions']
    print("\n" + "-"*70)
    print("Predictions Summary:")
    print("-"*70)
    print(f"Mean:               {y_pred.mean():.4f} ({y_pred.mean()*100:.2f}%)")
    print(f"Median:             {np.median(y_pred):.4f} ({np.median(y_pred)*100:.2f}%)")
    print(f"Std Dev:            {y_pred.std():.4f}")
    print(f"Min:                {y_pred.min():.4f} ({y_pred.min()*100:.2f}%)")
    print(f"Max:                {y_pred.max():.4f} ({y_pred.max()*100:.2f}%)")
    
    # Performance interpretation
    print("\n" + "-"*70)
    print("Interpretation:")
    print("-"*70)
    if results['r2'] >= 0.90:
        print("✅ Excellent model performance (R² ≥ 0.90)")
    elif results['r2'] >= 0.80:
        print("✅ Very good model performance (R² ≥ 0.80)")
    elif results['r2'] >= 0.70:
        print("✅ Good model performance (R² ≥ 0.70)")
    elif results['r2'] >= 0.60:
        print("⚠️  Fair model performance (R² ≥ 0.60)")
    else:
        print("❌ Poor model performance (R² < 0.60)")
    
    if results['rmse'] <= 0.05:
        print("✅ Excellent prediction accuracy (RMSE ≤ 0.05)")
    elif results['rmse'] <= 0.10:
        print("✅ Good prediction accuracy (RMSE ≤ 0.10)")
    else:
        print("⚠️  Moderate prediction accuracy (RMSE > 0.10)")
    
    print("="*70 + "\n")

def save_predictions(df, y_pred, output_file):
    """Save predictions to file"""
    df_output = df.copy()
    df_output['Predicted_%Pack'] = y_pred
    df_output['Predicted_%Pack_Pct'] = y_pred * 100
    df_output['Actual_%Pack'] = df['%Pack']
    df_output['Error'] = df['%Pack'] - y_pred
    df_output['Abs_Error'] = np.abs(df_output['Error'])
    df_output['Error_Pct'] = df_output['Error'] * 100
    
    if output_file.endswith('.csv'):
        df_output.to_csv(output_file, index=False)
    elif output_file.endswith('.xlsx'):
        df_output.to_excel(output_file, index=False)
    else:
        output_file += '.csv'
        df_output.to_csv(output_file, index=False)
    
    print(f"✓ Predictions saved to: {output_file}")

def main():
    parser = argparse.ArgumentParser(
        description='Pack SGA - Test model performance',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python test_model.py --data test_data.xlsx --output predictions.csv
        """
    )
    
    parser.add_argument('--model', default='pack_model.pkl',
                       help='Path to model file (default: pack_model.pkl)')
    parser.add_argument('--data', required=True,
                       help='Path to test data (CSV or Excel with %Pack column)')
    parser.add_argument('--output', help='Output file to save predictions (optional)')
    parser.add_argument('--cv', type=int, default=5,
                       help='Number of cross-validation folds (default: 5)')
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("Pack SGA - Model Testing System")
    print("="*70 + "\n")
    
    # Load model
    model_package = load_model(args.model)
    model = model_package['model']
    model_name = model_package.get('best_model_name', 'Unknown Model')
    
    # Load and prepare data
    df = load_data(args.data)
    
    if '%Pack' not in df.columns:
        print("❌ Error: Data must contain '%Pack' column for testing")
        sys.exit(1)
    
    print("✓ Preparing data...")
    X, y = prepare_data(df, model_package)
    
    # Evaluate model
    print("✓ Evaluating model...")
    results = evaluate_model(model, X, y, cv=args.cv)
    
    # Print report
    print_report(results, model_name, len(y))
    
    # Save predictions if requested
    if args.output:
        save_predictions(df, results['predictions'], args.output)
        print()

if __name__ == '__main__':
    main()
