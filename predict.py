#!/usr/bin/env python3
"""
Pack SGA - Standalone Prediction Script
Use this script to make predictions without Streamlit
"""

import pickle
import pandas as pd
import numpy as np
import sys
import argparse
from datetime import datetime

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

def predict_single(model_package, **kwargs):
    """
    Make single prediction
    
    Parameters:
    -----------
    shift : str - 'A', 'B', or 'C'
    furnace : int - 2 or 3
    line : int - 1-10
    customer : str - Customer name
    product_type : str - Product type
    description : str - Product description
    total_fg : int - Total FG bottles
    year : int - Year
    month : int - Month (1-12)
    day : int - Day (1-31)
    day_of_week : int - Day of week (0-6)
    quarter : int - Quarter (1-4)
    """
    
    model = model_package['model']
    scaler = model_package['scaler']
    label_encoders = model_package['label_encoders']
    feature_names = model_package['feature_names']
    categorical_cols = model_package['categorical_cols']
    
    # Create input dataframe
    input_data = pd.DataFrame([kwargs])
    
    # Encode categorical variables
    for col in categorical_cols:
        if col in input_data.columns:
            try:
                input_data[col] = label_encoders[col].transform(input_data[col].astype(str))
            except ValueError as e:
                print(f"❌ Error: Invalid value for {col}")
                print(f"   Valid values: {list(label_encoders[col].classes_)}")
                sys.exit(1)
    
    # Ensure correct column order
    input_data = input_data[feature_names]
    
    # Scale features
    input_scaled = scaler.transform(input_data)
    
    # Predict
    prediction = model.predict(input_scaled)[0]
    
    return prediction

def predict_batch(model_package, input_file):
    """
    Make batch predictions from CSV/Excel file
    
    Parameters:
    -----------
    input_file : str - Path to input file (.csv or .xlsx)
    """
    
    model = model_package['model']
    scaler = model_package['scaler']
    label_encoders = model_package['label_encoders']
    feature_names = model_package['feature_names']
    categorical_cols = model_package['categorical_cols']
    
    # Load input file
    try:
        if input_file.endswith('.csv'):
            df = pd.read_csv(input_file)
        elif input_file.endswith('.xlsx'):
            df = pd.read_excel(input_file)
        else:
            print("❌ Error: Input file must be .csv or .xlsx")
            sys.exit(1)
        
        print(f"✓ Loaded {len(df)} records from {input_file}")
    except Exception as e:
        print(f"❌ Error loading file: {e}")
        sys.exit(1)
    
    # Process date features if FullDate exists
    if 'FullDate' in df.columns:
        df['FullDate'] = pd.to_datetime(df['FullDate'])
        df['Year'] = df['FullDate'].dt.year
        df['Month'] = df['FullDate'].dt.month
        df['Day'] = df['FullDate'].dt.day
        df['DayOfWeek'] = df['FullDate'].dt.dayofweek
        df['Quarter'] = df['FullDate'].dt.quarter
    
    # Select features
    X = df[feature_names].copy()
    
    # Encode categorical variables
    for col in categorical_cols:
        try:
            X[col] = label_encoders[col].transform(X[col].astype(str))
        except Exception as e:
            print(f"❌ Error encoding {col}: {e}")
            sys.exit(1)
    
    # Scale features
    X_scaled = scaler.transform(X)
    
    # Predict
    predictions = model.predict(X_scaled)
    
    # Add predictions to dataframe
    df['Predicted_%Pack'] = predictions
    df['Predicted_%Pack_Pct'] = predictions * 100
    
    return df

def main():
    parser = argparse.ArgumentParser(
        description='Pack SGA - Make %Pack predictions',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single prediction
  python predict.py --single \\
    --shift A --furnace 2 --line 1 \\
    --customer "Customer A" --type "Type1" \\
    --description "Desc1" --total_fg 200000

  # Batch prediction
  python predict.py --batch input.csv --output predictions.csv
        """
    )
    
    parser.add_argument('--model', default='pack_model.pkl',
                       help='Path to model file (default: pack_model.pkl)')
    
    # Single prediction mode
    parser.add_argument('--single', action='store_true',
                       help='Single prediction mode')
    parser.add_argument('--shift', help='Shift (A, B, or C)')
    parser.add_argument('--furnace', type=int, help='Furnace number')
    parser.add_argument('--line', type=int, help='Line number')
    parser.add_argument('--customer', help='Customer name')
    parser.add_argument('--type', dest='product_type', help='Product type')
    parser.add_argument('--description', help='Product description')
    parser.add_argument('--total_fg', type=int, help='Total FG (bottles)')
    parser.add_argument('--year', type=int, default=datetime.now().year,
                       help='Year (default: current year)')
    parser.add_argument('--month', type=int, default=datetime.now().month,
                       help='Month (default: current month)')
    parser.add_argument('--day', type=int, default=datetime.now().day,
                       help='Day (default: current day)')
    parser.add_argument('--day_of_week', type=int, default=datetime.now().weekday(),
                       help='Day of week (0-6, default: current)')
    parser.add_argument('--quarter', type=int, default=(datetime.now().month-1)//3 + 1,
                       help='Quarter (1-4, default: current)')
    
    # Batch prediction mode
    parser.add_argument('--batch', help='Batch prediction mode - input CSV/Excel file')
    parser.add_argument('--output', help='Output file for batch predictions (default: predictions.csv)')
    
    args = parser.parse_args()
    
    # Load model
    print("\n" + "="*60)
    print("Pack SGA - Prediction System")
    print("="*60)
    model_package = load_model(args.model)
    
    if args.single:
        # Single prediction mode
        required_fields = ['shift', 'furnace', 'line', 'customer', 
                          'product_type', 'description', 'total_fg']
        
        missing = [f for f in required_fields if getattr(args, f) is None]
        if missing:
            print(f"\n❌ Error: Missing required fields: {', '.join(missing)}")
            parser.print_help()
            sys.exit(1)
        
        # Prepare input
        input_params = {
            'Shift': args.shift,
            'Furnace': args.furnace,
            'Line': args.line,
            'Customer': args.customer,
            'Type': args.product_type,
            'Description': args.description,
            'Total FG (Bottle)': args.total_fg,
            'Year': args.year,
            'Month': args.month,
            'Day': args.day,
            'DayOfWeek': args.day_of_week,
            'Quarter': args.quarter
        }
        
        print("\n📋 Input Parameters:")
        for key, value in input_params.items():
            print(f"  {key:20s}: {value}")
        
        # Predict
        prediction = predict_single(model_package, **input_params)
        
        print("\n" + "="*60)
        print("🎯 PREDICTION RESULT")
        print("="*60)
        print(f"Predicted %Pack:  {prediction:.4f} ({prediction*100:.2f}%)")
        print("="*60)
        
        if prediction >= 0.92:
            print("✅ Status: Excellent efficiency!")
        elif prediction >= 0.88:
            print("ℹ️  Status: Good efficiency")
        else:
            print("⚠️  Status: Below average - investigate factors")
        print()
        
    elif args.batch:
        # Batch prediction mode
        output_file = args.output or 'predictions.csv'
        
        print(f"\n📊 Processing batch predictions...")
        df_results = predict_batch(model_package, args.batch)
        
        # Save results
        if output_file.endswith('.csv'):
            df_results.to_csv(output_file, index=False)
        elif output_file.endswith('.xlsx'):
            df_results.to_excel(output_file, index=False)
        else:
            output_file += '.csv'
            df_results.to_csv(output_file, index=False)
        
        print(f"✓ Saved predictions to: {output_file}")
        
        # Show summary
        predictions = df_results['Predicted_%Pack'].values
        print("\n" + "="*60)
        print("📊 BATCH PREDICTION SUMMARY")
        print("="*60)
        print(f"Total records:  {len(predictions)}")
        print(f"Mean %Pack:     {predictions.mean():.4f} ({predictions.mean()*100:.2f}%)")
        print(f"Median %Pack:   {np.median(predictions):.4f} ({np.median(predictions)*100:.2f}%)")
        print(f"Min %Pack:      {predictions.min():.4f} ({predictions.min()*100:.2f}%)")
        print(f"Max %Pack:      {predictions.max():.4f} ({predictions.max()*100:.2f}%)")
        print(f"Std Dev:        {predictions.std():.4f}")
        print("="*60)
        print()
        
    else:
        print("\n❌ Error: Must specify either --single or --batch mode")
        parser.print_help()
        sys.exit(1)

if __name__ == '__main__':
    main()
