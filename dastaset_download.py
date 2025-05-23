import os
import urllib.request
import zipfile

def download_and_extract(url, output_dir, zip_name):
    os.makedirs(output_dir, exist_ok=True)
    zip_path = os.path.join(output_dir, zip_name)

    if not os.path.exists(zip_path):
        print(f"Downloading {zip_name}...")
        urllib.request.urlretrieve(url, zip_path)
        print(f"Downloaded {zip_name}.")
    else:
        print(f"{zip_name} already exists. Skipping download.")

    print(f"Extracting {zip_name}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(output_dir)
    print(f"Extracted to {output_dir}.")

# Configuration
base_dir = "./data/DIV2K"
train_url = "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip"
valid_url = "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip"

# Download and extract training and validation sets
download_and_extract(train_url, base_dir, "DIV2K_train_HR.zip")
download_and_extract(valid_url, base_dir, "DIV2K_valid_HR.zip")
