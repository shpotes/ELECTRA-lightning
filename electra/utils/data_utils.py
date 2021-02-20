import io
import requests
from typing import Optional
import zipfile

def download_wikitext(output_path: Optional[str] = None):
    """
    Download the wikitext-103 dataset

    Stephen Merity, Caiming Xiong, James Bradbury, and Richard Socher. 2016.
    Pointer Sentinel Mixture Models
    """
    
    _url = "https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-v1.zip"

    response = requests.get(_url, stream=True)
    with zipfile.ZipFile(io.BytesIO(response.content)) as zip_buf:
        zip_buf.extractall(output_path)
