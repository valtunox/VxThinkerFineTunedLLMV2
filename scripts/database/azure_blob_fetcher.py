#!/usr/bin/env python3
"""
Azure Blob Storage Dataset Fetcher
Downloads datasets from Azure Blob Storage for VaLLM training
"""
import argparse
import os
import sys
from pathlib import Path
import logging
from typing import Optional
import time

try:
    from azure.identity import DefaultAzureCredential, ManagedIdentityCredential
    from azure.storage.blob import BlobServiceClient, BlobClient
    from azure.core.exceptions import AzureError
except ImportError:
    print("Error: azure-storage-blob and azure-identity packages required")
    print("Install with: pip install azure-storage-blob azure-identity")
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_size(size_str: str) -> int:
    """Parse size string (e.g., '50GB', '100MB') to bytes"""
    size_str = size_str.upper().strip()
    multipliers = {
        'KB': 1024,
        'MB': 1024 ** 2,
        'GB': 1024 ** 3,
        'TB': 1024 ** 4
    }
    
    for unit, multiplier in multipliers.items():
        if size_str.endswith(unit):
            return int(float(size_str[:-len(unit)]) * multiplier)
    
    return int(size_str)


def get_blob_service_client(account_name: str, use_managed_identity: bool = True):
    """Get Azure Blob Service Client"""
    account_url = f"https://{account_name}.blob.core.windows.net"
    
    if use_managed_identity:
        # Use Managed Identity (for AKS)
        credential = ManagedIdentityCredential()
    else:
        # Use Default Credential (for local development)
        credential = DefaultAzureCredential()
    
    return BlobServiceClient(account_url=account_url, credential=credential)


def download_blob(
    blob_client: BlobClient,
    destination: Path,
    max_size: Optional[int] = None
) -> bool:
    """Download a single blob"""
    try:
        # Get blob properties
        properties = blob_client.get_blob_properties()
        size = properties.size
        
        if max_size and size > max_size:
            logger.warning(f"Skipping {blob_client.blob_name}: size {size} exceeds max {max_size}")
            return False
        
        # Create destination directory
        destination.parent.mkdir(parents=True, exist_ok=True)
        
        # Download blob
        logger.info(f"Downloading {blob_client.blob_name} -> {destination}")
        with open(destination, "wb") as download_file:
            download_file.write(blob_client.download_blob().readall())
        
        # Verify download
        if destination.exists() and destination.stat().st_size == size:
            logger.info(f"✓ Successfully downloaded {blob_client.blob_name} ({size} bytes)")
            return True
        else:
            logger.error(f"✗ Download verification failed for {blob_client.blob_name}")
            return False
            
    except AzureError as e:
        logger.error(f"Azure error downloading {blob_client.blob_name}: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error downloading {blob_client.blob_name}: {e}")
        return False


def list_blobs(blob_service_client: BlobServiceClient, container_name: str, prefix: str) -> list:
    """List all blobs in container with prefix"""
    blobs = []
    
    try:
        container_client = blob_service_client.get_container_client(container_name)
        for blob in container_client.list_blobs(name_starts_with=prefix):
            blobs.append(blob)
        return blobs
    except AzureError as e:
        logger.error(f"Error listing blobs: {e}")
        return []


def main():
    parser = argparse.ArgumentParser(description='Fetch datasets from Azure Blob Storage')
    parser.add_argument('--account', required=True, help='Azure Storage Account name')
    parser.add_argument('--container', required=True, help='Container name')
    parser.add_argument('--prefix', default='datasets/', help='Blob prefix (default: datasets/)')
    parser.add_argument('--destination', required=True, help='Local destination directory')
    parser.add_argument('--max-size', default='50GB', help='Maximum file size (default: 50GB)')
    parser.add_argument('--use-managed-identity', action='store_true', default=True,
                       help='Use Managed Identity (default: True)')
    parser.add_argument('--dry-run', action='store_true', help='List files without downloading')
    
    args = parser.parse_args()
    
    # Parse max size
    max_size_bytes = parse_size(args.max_size)
    
    # Create destination directory
    destination = Path(args.destination)
    destination.mkdir(parents=True, exist_ok=True)
    
    # Initialize Blob Service Client
    try:
        blob_service_client = get_blob_service_client(
            args.account,
            use_managed_identity=args.use_managed_identity
        )
        logger.info(f"Connected to Azure Storage Account: {args.account}")
    except Exception as e:
        logger.error(f"Error connecting to Azure Storage: {e}")
        sys.exit(1)
    
    # List blobs
    logger.info(f"Listing blobs with prefix: {args.prefix}")
    blobs = list_blobs(blob_service_client, args.container, args.prefix)
    
    if not blobs:
        logger.warning(f"No blobs found with prefix: {args.prefix}")
        return
    
    logger.info(f"Found {len(blobs)} blobs")
    
    if args.dry_run:
        logger.info("DRY RUN - Files that would be downloaded:")
        for blob in blobs:
            size = blob.size
            logger.info(f"  {blob.name} ({size} bytes)")
        return
    
    # Download blobs
    success_count = 0
    failed_count = 0
    total_size = 0
    
    for blob in blobs:
        blob_name = blob.name
        size = blob.size
        
        # Skip if too large
        if size > max_size_bytes:
            logger.warning(f"Skipping {blob_name}: size exceeds limit")
            failed_count += 1
            continue
        
        # Determine local file path
        relative_path = blob_name[len(args.prefix):] if blob_name.startswith(args.prefix) else blob_name
        local_path = destination / relative_path
        
        # Skip if already exists and size matches
        if local_path.exists() and local_path.stat().st_size == size:
            logger.info(f"✓ Already exists: {blob_name}")
            success_count += 1
            total_size += size
            continue
        
        # Get blob client and download
        blob_client = blob_service_client.get_blob_client(
            container=args.container,
            blob=blob_name
        )
        
        if download_blob(blob_client, local_path, max_size_bytes):
            success_count += 1
            total_size += size
        else:
            failed_count += 1
        
        # Small delay to avoid rate limiting
        time.sleep(0.1)
    
    # Summary
    logger.info("=" * 60)
    logger.info(f"Download Summary:")
    logger.info(f"  Success: {success_count}")
    logger.info(f"  Failed: {failed_count}")
    logger.info(f"  Total Size: {total_size / (1024**3):.2f} GB")
    logger.info("=" * 60)
    
    if failed_count > 0:
        sys.exit(1)


if __name__ == '__main__':
    main()
