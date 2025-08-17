import requests
import os
from urllib.parse import urlparse
from typing import Union, Mapping, Any
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from config import (
    logger, 
    IPFS_PREFIX, 
    IPFS_BASE_URL, 
    PINATA_JWT,
)


class IPFSError(Exception):
    """Error during blockchain operations."""
    pass


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=2, max=32),
    retry=retry_if_exception_type((IPFSError, requests.RequestException)),
    before_sleep=lambda retry_state: logger.info(f"Retrying IPFS upload (attempt {retry_state.attempt_number}/3)...")
)
def pin(data: Union[str, Mapping[str, Any]]) -> str:
    """
    Upload to Pinata and return an ipfs:// URL.

    Accepts:
      - URL string (http/https): downloads and uploads as a file
      - Local file path string: uploads the file
      - JSON blob (dict-like Mapping): uses pinJSONToIPFS
    """

    if not PINATA_JWT:
        raise IPFSError("PINATA_JWT not configured")
    
    file_endpoint = f"{IPFS_BASE_URL}/pinning/pinFileToIPFS"
    auth_headers = {"Authorization": f"Bearer {PINATA_JWT}"}

    try:
        # JSON blob
        if isinstance(data, Mapping):
            url = f"{IPFS_BASE_URL}/pinning/pinJSONToIPFS"
            payload = {"pinataContent": dict(data)}
            logger.info("Uploading JSON to IPFS...")
            r = requests.post(url, headers=auth_headers, json=payload, timeout=60)
            
        # URL
        elif data.startswith(("http://", "https://")):
            logger.info(f"Downloading file from {data}...")
            dl = requests.get(data, timeout=60)
            if dl.status_code != 200:
                raise IPFSError(f"Failed to download file {data}: {dl.status_code}")
            filename = os.path.basename(urlparse(data).path) or "file"
            files = {"file": (filename, dl.content)}
            logger.info("Uploading downloaded file to IPFS...")
            r = requests.post(file_endpoint, files=files, headers=auth_headers, timeout=60)

        # Local file path
        elif os.path.isfile(data):
            filename = os.path.basename(data)
            logger.info(f"Uploading local file to IPFS: {filename}...")
            with open(data, "rb") as f:
                files = {"file": (filename, f)}
                r = requests.post(file_endpoint, files=files, headers=auth_headers, timeout=60)

        else:
            raise ValueError("Data must be json, URL, or local file path")

        # Check response
        if r.status_code != 200:
            try:
                detail = r.json()
            except Exception:
                detail = r.text
            raise IPFSError(f"IPFS JSON upload failed: {r.status_code} {detail}")

        ipfs_hash = r.json().get("IpfsHash")        
        if not ipfs_hash:
            raise IPFSError(f"Malformed response from IPFS: {r.text}")
        
        url = f"{IPFS_PREFIX}{ipfs_hash}"
        logger.info(f"Uploaded to IPFS: {url}")        
        return ipfs_hash
        
    except requests.RequestException as e:
        raise IPFSError(f"Network error during IPFS upload: {e}")

    except Exception as e:
        raise IPFSError(f"IPFS upload error: {e}")