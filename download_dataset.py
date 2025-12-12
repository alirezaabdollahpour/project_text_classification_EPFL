import os
import aicrowd
from aicrowd import dataset
from pathlib import Path

# --- CONFIGURATION ---
API_KEY = "0fbf985caaf61b3b7ec8b83b52211f14"
CHALLENGE_NAME = "epfl-ml-text-classification"
OUTPUT_DIR = "data"

def setup_aicrowd_config(api_key):
    """
    Manually creates the AIcrowd config file.
    This bypasses the need to run 'aicrowd login' in the shell.
    """
    # Define the standard config path for Linux
    config_dir = Path(os.path.expanduser("~/.config/aicrowd-cli"))
    config_file = config_dir / "config.toml"
    
    # Create directory if it doesn't exist
    config_dir.mkdir(parents=True, exist_ok=True)
    
    # Write the API key to the config file
    print(f"Creating config file at: {config_file}")
    with open(config_file, "w") as f:
        f.write(f'aicrowd_api_key = "{api_key}"\n')

def download_data():
    print(f"--- Downloading Dataset for {CHALLENGE_NAME} ---")
    
    # Ensure output directory exists
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # Trigger the download
    # We explicitly pass jobs=1 and dataset_files=[] to satisfy strict library versions
    dataset.download_dataset(
        challenge=CHALLENGE_NAME,
        output_dir=OUTPUT_DIR,
        jobs=1,              # force single threaded to be safe
        dataset_files=[]     # empty list = download all files
    )
    print(f"\nSuccess! Files saved to: {os.path.abspath(OUTPUT_DIR)}")

if __name__ == "__main__":
    try:
        # 1. Force-create the login credentials
        setup_aicrowd_config(API_KEY)
        
        # 2. Download the data
        download_data()
        
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        # If the library version is very old/new, it might need different args.
        print("Tip: If you see a TypeError, check the library version arguments.")