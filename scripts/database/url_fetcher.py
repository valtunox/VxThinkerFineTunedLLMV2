#!/usr/bin/env python3
"""
URL Dataset Fetcher
Downloads datasets from URLs for VaLLM training
"""
import argparse
import os
import sys
from pathlib import Path
import logging
from typing import Optional, Tuple
import hashlib
import time
import ssl

try:
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
except ImportError:
    print("Error: requests package required")
    print("Install with: pip install requests")
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


def parse_checksum(checksum_str: str) -> Tuple[str, str]:
    """Parse checksum string (e.g., 'sha256:abc123')"""
    if ':' in checksum_str:
        algorithm, value = checksum_str.split(':', 1)
        return algorithm.lower(), value
    return 'sha256', checksum_str


def calculate_checksum(file_path: Path, algorithm: str = 'sha256') -> str:
    """Calculate file checksum"""
    hash_obj = hashlib.new(algorithm)
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b''):
            hash_obj.update(chunk)
    return hash_obj.hexdigest()


def verify_checksum(file_path: Path, expected_checksum: str, algorithm: str = 'sha256') -> bool:
    """Verify file checksum"""
    actual = calculate_checksum(file_path, algorithm)
    return actual.lower() == expected_checksum.lower()


def create_session(verify_ssl: bool = True, timeout: int = 300) -> requests.Session:
    """Create requests session with retry strategy"""
    session = requests.Session()
    
    retry_strategy = Retry(
        total=5,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["HEAD", "GET", "OPTIONS"]
    )
    
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    
    session.verify = verify_ssl
    session.timeout = timeout
    
    return session


def get_file_size(url: str, session: requests.Session) -> Optional[int]:
    """Get file size from URL using HEAD request"""
    try:
        response = session.head(url, allow_redirects=True)
        if response.status_code == 200:
            content_length = response.headers.get('Content-Length')
            if content_length:
                return int(content_length)
    except Exception as e:
        logger.warning(f"Could not get file size for {url}: {e}")
    return None


def download_file(
    url: str,
    destination: Path,
    session: requests.Session,
    max_size: Optional[int] = None,
    expected_checksum: Optional[str] = None,
    checksum_algorithm: str = 'sha256'
) -> bool:
    """Download a file from URL"""
    try:
        # Get file size first
        file_size = get_file_size(url, session)
        if file_size and max_size and file_size > max_size:
            logger.warning(f"Skipping {url}: size {file_size} exceeds max {max_size}")
            return False
        
        # Create destination directory
        destination.parent.mkdir(parents=True, exist_ok=True)
        
        # Download file
        logger.info(f"Downloading {url} -> {destination}")
        response = session.get(url, stream=True)
        response.raise_for_status()
        
        # Download with progress
        total_size = int(response.headers.get('Content-Length', 0))
        downloaded = 0
        
        with open(destination, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    
                    # Check size during download
                    if max_size and downloaded > max_size:
                        logger.error(f"File size exceeded during download: {url}")
                        destination.unlink()
                        return False
        
        # Verify checksum if provided
        if expected_checksum:
            if not verify_checksum(destination, expected_checksum, checksum_algorithm):
                logger.error(f"Checksum verification failed for {url}")
                destination.unlink()
                return False
            logger.info(f"✓ Checksum verified: {url}")
        
        logger.info(f"✓ Successfully downloaded {url} ({downloaded} bytes)")
        return True
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Error downloading {url}: {e}")
        if destination.exists():
            destination.unlink()
        return False
    except Exception as e:
        logger.error(f"Unexpected error downloading {url}: {e}")
        if destination.exists():
            destination.unlink()
        return False


def parse_urls_file(urls_file: Path) -> list:
    """Parse URLs file (format: URL|FILENAME|CHECKSUM)"""
    urls = []
    
    if not urls_file.exists():
        logger.error(f"URLs file not found: {urls_file}")
        return urls
    
    with open(urls_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            
            # Skip comments and empty lines
            if not line or line.startswith('#'):
                continue
            
            # Parse line: URL|FILENAME|CHECKSUM (checksum optional)
            parts = line.split('|')
            if len(parts) < 2:
                logger.warning(f"Invalid line {line_num}: {line}")
                continue
            
            url = parts[0].strip()
            filename = parts[1].strip()
            checksum = parts[2].strip() if len(parts) > 2 else None
            
            if checksum:
                checksum_algorithm, checksum_value = parse_checksum(checksum)
            else:
                checksum_algorithm = None
                checksum_value = None
            
            urls.append({
                'url': url,
                'filename': filename,
                'checksum': checksum_value,
                'algorithm': checksum_algorithm
            })
    
    return urls


def main():
    parser = argparse.ArgumentParser(description='Fetch datasets from URLs')
    parser.add_argument('--urls-file', required=True, help='File containing URLs (one per line)')
    parser.add_argument('--destination', required=True, help='Local destination directory')
    parser.add_argument('--max-size', default='50GB', help='Maximum file size (default: 50GB)')
    parser.add_argument('--verify-ssl', action='store_true', default=True, help='Verify SSL certificates')
    parser.add_argument('--timeout', type=int, default=300, help='Request timeout in seconds')
    parser.add_argument('--dry-run', action='store_true', help='List URLs without downloading')
    
    args = parser.parse_args()
    
    # Parse max size
    max_size_bytes = parse_size(args.max_size)
    
    # Create destination directory
    destination = Path(args.destination)
    destination.mkdir(parents=True, exist_ok=True)
    
    # Parse URLs file
    urls_file = Path(args.urls_file)
    urls = parse_urls_file(urls_file)
    
    if not urls:
        logger.error("No URLs found in file")
        sys.exit(1)
    
    logger.info(f"Found {len(urls)} URLs to download")
    
    if args.dry_run:
        logger.info("DRY RUN - URLs that would be downloaded:")
        for url_info in urls:
            logger.info(f"  {url_info['url']} -> {url_info['filename']}")
        return
    
    # Create session
    session = create_session(verify_ssl=args.verify_ssl, timeout=args.timeout)
    
    # Download files
    success_count = 0
    failed_count = 0
    total_size = 0
    
    for url_info in urls:
        url = url_info['url']
        filename = url_info['filename']
        checksum = url_info['checksum']
        algorithm = url_info['algorithm']
        
        local_path = destination / filename
        
        # Skip if already exists
        if local_path.exists():
            if checksum:
                if verify_checksum(local_path, checksum, algorithm):
                    logger.info(f"✓ Already exists and verified: {filename}")
                    success_count += 1
                    total_size += local_path.stat().st_size
                    continue
                else:
                    logger.warning(f"Existing file checksum mismatch, re-downloading: {filename}")
                    local_path.unlink()
            else:
                logger.info(f"✓ Already exists: {filename}")
                success_count += 1
                total_size += local_path.stat().st_size
                continue
        
        # Download
        if download_file(url, local_path, session, max_size_bytes, checksum, algorithm):
            success_count += 1
            total_size += local_path.stat().st_size
        else:
            failed_count += 1
        
        # Small delay to avoid rate limiting
        time.sleep(1)
    
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
