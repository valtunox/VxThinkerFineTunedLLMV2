#!/usr/bin/env python3
"""
S3 Dataset Fetcher
Downloads datasets from S3 buckets for VaLLM training
"""
import argparse
import os
import sys
import boto3
from botocore.exceptions import ClientError, NoCredentialsError
from pathlib import Path
import hashlib
import logging
from typing import Optional
import time

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


def calculate_checksum(file_path: Path, algorithm: str = 'sha256') -> str:
    """Calculate file checksum"""
    hash_obj = hashlib.new(algorithm)
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b''):
            hash_obj.update(chunk)
    return hash_obj.hexdigest()


def download_s3_object(
    s3_client,
    bucket: str,
    key: str,
    destination: Path,
    max_size: Optional[int] = None
) -> bool:
    """Download a single S3 object"""
    try:
        # Get object metadata
        head_response = s3_client.head_object(Bucket=bucket, Key=key)
        size = head_response.get('ContentLength', 0)
        
        if max_size and size > max_size:
            logger.warning(f"Skipping {key}: size {size} exceeds max {max_size}")
            return False
        
        # Create destination directory
        destination.parent.mkdir(parents=True, exist_ok=True)
        
        # Download file
        logger.info(f"Downloading s3://{bucket}/{key} -> {destination}")
        s3_client.download_file(bucket, key, str(destination))
        
        # Verify download
        if destination.exists() and destination.stat().st_size == size:
            logger.info(f"✓ Successfully downloaded {key} ({size} bytes)")
            return True
        else:
            logger.error(f"✗ Download verification failed for {key}")
            return False
            
    except ClientError as e:
        logger.error(f"Error downloading {key}: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error downloading {key}: {e}")
        return False


def list_s3_objects(s3_client, bucket: str, prefix: str) -> list:
    """List all objects in S3 bucket with prefix"""
    objects = []
    paginator = s3_client.get_paginator('list_objects_v2')
    
    try:
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            if 'Contents' in page:
                objects.extend(page['Contents'])
        return objects
    except ClientError as e:
        logger.error(f"Error listing objects: {e}")
        return []


def main():
    parser = argparse.ArgumentParser(description='Fetch datasets from S3')
    parser.add_argument('--bucket', required=True, help='S3 bucket name')
    parser.add_argument('--prefix', default='datasets/', help='S3 prefix (default: datasets/)')
    parser.add_argument('--destination', required=True, help='Local destination directory')
    parser.add_argument('--max-size', default='50GB', help='Maximum file size (default: 50GB)')
    parser.add_argument('--region', default='us-east-1', help='AWS region')
    parser.add_argument('--verify-checksum', action='store_true', help='Verify file checksums')
    parser.add_argument('--dry-run', action='store_true', help='List files without downloading')
    
    args = parser.parse_args()
    
    # Parse max size
    max_size_bytes = parse_size(args.max_size)
    
    # Create destination directory
    destination = Path(args.destination)
    destination.mkdir(parents=True, exist_ok=True)
    
    # Initialize S3 client
    try:
        s3_client = boto3.client('s3', region_name=args.region)
        logger.info(f"Connected to S3 bucket: {args.bucket}")
    except NoCredentialsError:
        logger.error("AWS credentials not found. Using IAM role or environment variables.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error connecting to S3: {e}")
        sys.exit(1)
    
    # List objects
    logger.info(f"Listing objects with prefix: {args.prefix}")
    objects = list_s3_objects(s3_client, args.bucket, args.prefix)
    
    if not objects:
        logger.warning(f"No objects found with prefix: {args.prefix}")
        return
    
    logger.info(f"Found {len(objects)} objects")
    
    if args.dry_run:
        logger.info("DRY RUN - Files that would be downloaded:")
        for obj in objects:
            size = obj.get('Size', 0)
            logger.info(f"  {obj['Key']} ({size} bytes)")
        return
    
    # Download objects
    success_count = 0
    failed_count = 0
    total_size = 0
    
    for obj in objects:
        key = obj['Key']
        size = obj.get('Size', 0)
        
        # Skip if too large
        if size > max_size_bytes:
            logger.warning(f"Skipping {key}: size exceeds limit")
            failed_count += 1
            continue
        
        # Determine local file path
        relative_path = key[len(args.prefix):] if key.startswith(args.prefix) else key
        local_path = destination / relative_path
        
        # Skip if already exists and size matches
        if local_path.exists() and local_path.stat().st_size == size:
            logger.info(f"✓ Already exists: {key}")
            success_count += 1
            total_size += size
            continue
        
        # Download
        if download_s3_object(s3_client, args.bucket, key, local_path, max_size_bytes):
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
