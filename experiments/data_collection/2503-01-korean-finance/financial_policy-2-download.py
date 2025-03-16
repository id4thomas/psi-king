import json
import multiprocessing
import os
import random
import requests
import time
from typing import List, Literal

from bs4 import BeautifulSoup
import httpx
import pandas as pd
from pydantic import BaseModel, Field
from tqdm import tqdm

from src.config import settings

DEST_DIR = os.path.join(
    settings.data_dir,
    "retrieval_dataset/2503-01-korean-finance/kr-fsc_policy"
)
os.makedirs(DEST_DIR, exist_ok=True)

def download_pdf(item: dict):
    """
    Downloads a PDF using item['item_id'] (upperNo) and item['no'] (fileNo).
    Saves the file locally using item['name'] and item['extension'].
    """
    item_id = item["item_id"]
    file_no = item["no"]
    file_name = item["name"]
    extension = item["extension"]

    # Construct the URL for download
    url = f"https://www.fsc.go.kr/comm/getFile?srvcId=BBSTY1&upperNo={item_id}&fileTy=ATTACH&fileNo={file_no}"

    try:
        response = requests.get(url, timeout=30)
        if response.status_code == 200:
            # Create a clean filename: you can customize how the filename is generated
            safe_name = f"{item_id}_{file_no}.{extension}"
            file_path = os.path.join(DEST_DIR, safe_name)
            with open(file_path, "wb") as f:
                f.write(response.content)
            print(f"Downloaded: {file_path}")
        else:
            print(f"Failed to download (item_id={item_id}, no={file_no}), status={response.status_code}")
    except Exception as e:
        print(f"Error downloading (item_id={item_id}, no={file_no}): {e}")
        
    x = random.randint(0, 10)
    time.sleep(0.1*x)

def main():
    # Load Metadata
    with open(os.path.join(settings.data_dir, "retrieval_dataset/2503-01-korean-finance/kr-fsc_pdf_file_metadata.json"), "r") as f:
        metadata = json.load(f)

    df = pd.DataFrame.from_dict(metadata)
    print(df.shape, df.columns)

    # Convert DataFrame rows to dictionaries for easy multiprocessing
    items = df.to_dict(orient="records")

    # Use multiprocessing to download in parallel
    with multiprocessing.Pool(processes=4) as pool:
        pool.map(download_pdf, items)

if __name__ == "__main__":
    main()