import os
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# Create output directories
output_dir = r"c:\Users\User\Desktop\FYP_WORK\cleaned_datasets"
scalers_dir = os.path.join(output_dir, "scalers")
os.makedirs(output_dir, exist_ok=True)
os.makedirs(scalers_dir, exist_ok=True)

print("Output folders created successfully.")

# Define all dataset files to process
datasets_to_process = [
    # IoT Datasets
    {
        "category": "IoT",
        "input_path": r"c:\Users\User\Desktop\FYP_WORK\Train_Test_datasets\Train_Test_IoT_dataset\Train_Test_IoT_Fridge.csv",
        "output_name": "cleaned_IoT_Fridge.csv"
    },
    {
        "category": "IoT",
        "input_path": r"c:\Users\User\Desktop\FYP_WORK\Train_Test_datasets\Train_Test_IoT_dataset\Train_Test_IoT_GPS_Tracker.csv",
        "output_name": "cleaned_IoT_GPS_Tracker.csv"
    },
    {
        "category": "IoT",
        "input_path": r"c:\Users\User\Desktop\FYP_WORK\Train_Test_datasets\Train_Test_IoT_dataset\Train_Test_IoT_Garage_Door.csv",
        "output_name": "cleaned_IoT_Garage_Door.csv"
    },
    {
        "category": "IoT",
        "input_path": r"c:\Users\User\Desktop\FYP_WORK\Train_Test_datasets\Train_Test_IoT_dataset\Train_Test_IoT_Modbus.csv",
        "output_name": "cleaned_IoT_Modbus.csv"
    },
    {
        "category": "IoT",
        "input_path": r"c:\Users\User\Desktop\FYP_WORK\Train_Test_datasets\Train_Test_IoT_dataset\Train_Test_IoT_Motion_Light.csv",
        "output_name": "cleaned_IoT_Motion_Light.csv"
    },
    {
        "category": "IoT",
        "input_path": r"c:\Users\User\Desktop\FYP_WORK\Train_Test_datasets\Train_Test_IoT_dataset\Train_Test_IoT_Thermostat.csv",
        "output_name": "cleaned_IoT_Thermostat.csv"
    },
    {
        "category": "IoT",
        "input_path": r"c:\Users\User\Desktop\FYP_WORK\Train_Test_datasets\Train_Test_IoT_dataset\Train_Test_IoT_Weather.csv",
        "output_name": "cleaned_IoT_Weather.csv"
    },
    # Linux OS Datasets
    {
        "category": "Linux",
        "input_path": r"c:\Users\User\Desktop\FYP_WORK\Train_Test_datasets\Train_Test_Linux_dataset\Train_Test_Linux_process.csv",
        "output_name": "cleaned_Linux_process.csv"
    },
    {
        "category": "Linux",
        "input_path": r"c:\Users\User\Desktop\FYP_WORK\Train_Test_datasets\Train_Test_Linux_dataset\Train_test_linux_disk.csv",
        "output_name": "cleaned_Linux_disk.csv"
    },
    {
        "category": "Linux",
        "input_path": r"c:\Users\User\Desktop\FYP_WORK\Train_Test_datasets\Train_Test_Linux_dataset\Train_test_linux_memory.csv",
        "output_name": "cleaned_Linux_memory.csv"
    },
    # Windows OS Datasets
    {
        "category": "Windows",
        "input_path": r"c:\Users\User\Desktop\FYP_WORK\Train_Test_datasets\Train_Test_Windows_dataset\Train_Test_Windows_10.csv",
        "output_name": "cleaned_Windows_10.csv"
    },
    {
        "category": "Windows",
        "input_path": r"c:\Users\User\Desktop\FYP_WORK\Train_Test_datasets\Train_Test_Windows_dataset\Train_Test_Windows_7.csv",
        "output_name": "cleaned_Windows_7.csv"
    }
]

# Standard metadata and transient identifiers to exclude from training
metadata_cols = ['ts', 'date', 'time', 'PID', 'index']

for idx, ds in enumerate(datasets_to_process, 1):
    input_path = ds["input_path"]
    output_name = ds["output_name"]
    category = ds["category"]
    
    print(f"\n==================================================")
    print(f"[{idx}/{len(datasets_to_process)}] Processing: {os.path.basename(input_path)} ({category})")
    print(f"==================================================")
    
    # 1. Load file
    if not os.path.exists(input_path):
        print(f"Warning: File not found at {input_path}. Skipping.")
        continue
        
    df = pd.read_csv(input_path, low_memory=False)
    print(f"Original shape: {df.shape}")
    
    # Replace Zeek dashes with NaNs
    df.replace('-', np.nan, inplace=True)
    
    # Standardize 'attack' to 'label' (special case for Linux Disk)
    if 'attack' in df.columns:
        df.rename(columns={'attack': 'label'}, inplace=True)
        print("Renamed 'attack' column to 'label' for standard compatibility.")
        
    # Check if target columns are present
    if 'label' not in df.columns or 'type' not in df.columns:
        print(f"Warning: Missing 'label' or 'type' in {input_path}. Skipping.")
        continue
        
    # 2. Drop empty columns (> 60% missing)
    missing_pct = df.isnull().mean() * 100
    empty_cols = missing_pct[missing_pct > 60.0].index
    if len(empty_cols) > 0:
        df.drop(columns=empty_cols, inplace=True)
        print(f"Dropped {len(empty_cols)} empty columns: {list(empty_cols)}")
        
    # Drop duplicates & remaining NaNs
    df.drop_duplicates(inplace=True)
    df.dropna(inplace=True)
    print(f"Shape after duplicates and NaNs dropped: {df.shape}")
    
    # 3. Separate routing metadata (IP/Port details) if they exist
    found_metadata = [col for col in metadata_cols if col in df.columns]
    if len(found_metadata) > 0:
        # Save isolated metadata for Wazuh EDR
        meta_out_name = f"metadata_{output_name}"
        meta_out_path = os.path.join(output_dir, meta_out_name)
        df[found_metadata].to_csv(meta_out_path, index=False)
        print(f"Saved EDR metadata ({found_metadata}) to: {meta_out_path}")
        
        # Drop metadata columns from feature table
        df.drop(columns=found_metadata, inplace=True)
        
    # 4. Encode text columns
    text_cols = df.select_dtypes(include=['object']).columns
    print(f"Encoding text columns: {list(text_cols)}")
    
    for col in text_cols:
        if col != 'type': # Leave target label 'type'
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            
    # Encode target multiclass label 'type'
    type_le = LabelEncoder()
    df['type'] = type_le.fit_transform(df['type'].astype(str))
    print(f"Encoded classes for 'type': {list(type_le.classes_)}")
    
    # Separate features and targets
    y_bin = df['label'].reset_index(drop=True)
    y_mul = df['type'].reset_index(drop=True)
    X_feats = df.drop(columns=['label', 'type']).reset_index(drop=True)
    
    # 5. Fit & Transform scaling (leakage-free)
    split_idx = int(0.8 * len(X_feats))
    scaler = MinMaxScaler()
    
    # Fit only on train split, transform the entire features table
    scaler.fit(X_feats.iloc[:split_idx])
    X_scaled = pd.DataFrame(scaler.transform(X_feats), columns=X_feats.columns)
    
    # Save the scaler
    scaler_save_name = f"scaler_{output_name.replace('.csv', '.joblib')}"
    scaler_save_path = os.path.join(scalers_dir, scaler_save_name)
    joblib.dump(scaler, scaler_save_path)
    print(f"Saved fitted scaler to: {scaler_save_path}")
    
    # Re-attach targets
    cleaned_df = pd.concat([X_scaled, y_bin, y_mul], axis=1)
    
    # 6. Save clean dataset
    final_out_path = os.path.join(output_dir, output_name)
    cleaned_df.to_csv(final_out_path, index=False)
    print(f"Saved clean dataset to: {final_out_path}")
    print(f"Clean shape: {cleaned_df.shape}")

print("\n==================================================")
print("             ALL DATASETS CLEANED                 ")
print("==================================================")
