import os
import requests

# Function to download and extract dataset
def download_and_extract(url, output_dir="datasets"):
    os.makedirs(output_dir, exist_ok=True)
    file_name = os.path.join(output_dir, url.split("/")[-1])

    # Download the file
    print(f"Downloading {url}...")
    response = requests.get(url)
    with open(file_name, "wb") as f:
        f.write(response.content)
    print(f"Downloaded to {file_name}")

    # Extract the file
    extracted_file = file_name.replace(".gz", "")
    with gzip.open(file_name, "rt", encoding="utf-8") as gz_file:
        with open(extracted_file, "w", encoding="utf-8") as out_file:
            out_file.write(gz_file.read())
    print(f"Extracted to {extracted_file}")
    return extracted_file
