import pandas as pd
import numpy as np
from pathlib import Path
import os

# Set up data directory path
DATA_DIR = Path(__file__).parent

def load_and_inspect_data():
    """
    Load all CSV files and perform initial data inspection
    """
    print("🔧 Step 3: Loading and Inspecting Data")
    print("=" * 50)
    
    try:
        # Load all CSV files
        print("📁 Loading CSV files...")
        drug_reviews = pd.read_csv(DATA_DIR / 'drug_reviews.csv')
        patient_demo = pd.read_csv(DATA_DIR / 'patient_demographics.csv')
        faers_signals = pd.read_csv(DATA_DIR / 'faers_signals.csv')
        drug_indications = pd.read_csv(DATA_DIR / 'drug_indications.csv')
        
        print("✅ All files loaded successfully!")
        print()
        
        # Data inspection
        datasets = {
            'Drug Reviews': drug_reviews,
            'Patient Demographics': patient_demo,
            'FAERS Signals': faers_signals,
            'Drug Indications': drug_indications
        }
        
        for name, df in datasets.items():
            print(f"📊 {name}:")
            print(f"   Shape: {df.shape}")
            print(f"   Columns: {list(df.columns)}")
            print(f"   Sample data:")
            print(df.head(3).to_string(index=False))
            print(f"   Missing values: {df.isnull().sum().sum()}")
            print("-" * 50)
        
        return datasets
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None

def verify_csv_format():
    """
    Verify CSV format and data integrity
    """
    print("🧼 Step 4: Verifying CSV Format and Data Integrity")
    print("=" * 50)
    
    csv_files = [
        'drug_reviews.csv',
        'patient_demographics.csv', 
        'faers_signals.csv',
        'drug_indications.csv'
    ]
    
    for file in csv_files:
        file_path = DATA_DIR / file
        if file_path.exists():
            try:
                df = pd.read_csv(file_path)
                print(f"✅ {file}: Valid CSV format")
                print(f"   - Rows: {len(df)}")
                print(f"   - Columns: {len(df.columns)}")
                print(f"   - Has headers: {'Yes' if not df.columns.str.contains('Unnamed').any() else 'No'}")
                print(f"   - Empty rows: {df.isnull().all(axis=1).sum()}")
            except Exception as e:
                print(f"❌ {file}: Error - {e}")
        else:
            print(f"❌ {file}: File not found")
        print()

def main():
    """
    Main function to run data loading and verification
    """
    print("🚀 Backend Data Processing Pipeline")
    print("=" * 50)
    
    # Step 1: Verify folder structure
    print("📁 Current folder structure:")
    print(f"   Backend directory: {Path(__file__).parent}")
    print(f"   Data directory: {DATA_DIR}")
    print(f"   Data directory exists: {DATA_DIR.exists()}")
    print()
    
    # Step 2: Verify CSV format
    verify_csv_format()
    
    # Step 3: Load and inspect data
    datasets = load_and_inspect_data()
    
    if datasets:
        print("🎉 Data loading completed successfully!")
        print("📋 Next steps:")
        print("   - Data is ready for analysis")
        print("   - Consider data cleaning if needed")
        print("   - Ready for TabPFN or other ML models")
    else:
        print("❌ Data loading failed. Please check file paths and formats.")

if __name__ == "__main__":
    main()