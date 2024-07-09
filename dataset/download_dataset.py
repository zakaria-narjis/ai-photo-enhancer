import os
import gdown
import zipfile
import shutil
from pathlib import Path

def download_and_prepare_dataset():
    # URL of the Google Drive file
    url = "https://drive.google.com/uc?id=11HEUmchFXyepI4v3dhjnDnmhW_DgwfRR"
    
    # Create dataset directory if it doesn't exist
    current_dir = Path(__file__).parent.absolute()
    dataset_dir = os.path.join(current_dir,"FiveK")
    os.makedirs(dataset_dir, exist_ok=True)

    # Download the zip file
    zip_path = os.path.join(dataset_dir, "FiveK.zip")
    gdown.download(url, zip_path, quiet=False)
    
    # Unzip the file
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(dataset_dir)
    
    # Delete the zip file
    os.remove(zip_path)
    
    print("Dataset downloaded, unzipped, and cleaned up successfully.")

    # If the dataset is inside a subdirectory, move it to the main dataset directory
    subdirs = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]
    if len(subdirs) == 1:
        subdir_path = os.path.join(dataset_dir, subdirs[0])
        for item in os.listdir(subdir_path):
            shutil.move(os.path.join(subdir_path, item), dataset_dir)
        os.rmdir(subdir_path)
        print(f"Moved contents from {subdirs[0]} to {dataset_dir}")

if __name__ == "__main__":
    download_and_prepare_dataset()