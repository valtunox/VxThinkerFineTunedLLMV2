#!/usr/bin/env python3
"""
Dataset Processor
Processes and validates downloaded datasets for VaLLM training
"""
import argparse
import os
import sys
from pathlib import Path
import logging
import pandas as pd
import hashlib
from typing import List, Set
import json

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_file_hash(file_path: Path) -> str:
    """Get file hash for deduplication"""
    hash_obj = hashlib.md5()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b''):
            hash_obj.update(chunk)
    return hash_obj.hexdigest()


def validate_csv(file_path: Path) -> bool:
    """Validate CSV file"""
    try:
        # Try to read CSV
        df = pd.read_csv(file_path, nrows=1000)  # Read first 1000 rows for validation
        
        # Check for required columns (adjust based on your schema)
        if df.empty:
            logger.warning(f"CSV file is empty: {file_path}")
            return False
        
        logger.info(f"✓ CSV validated: {file_path} ({len(df)} rows, {len(df.columns)} columns)")
        return True
        
    except Exception as e:
        logger.error(f"CSV validation failed for {file_path}: {e}")
        return False


def deduplicate_files(files: List[Path]) -> List[Path]:
    """Remove duplicate files based on content hash"""
    seen_hashes: Set[str] = set()
    unique_files: List[Path] = []
    duplicates: List[Path] = []
    
    for file_path in files:
        file_hash = get_file_hash(file_path)
        
        if file_hash in seen_hashes:
            logger.info(f"Duplicate found: {file_path}")
            duplicates.append(file_path)
        else:
            seen_hashes.add(file_hash)
            unique_files.append(file_path)
    
    if duplicates:
        logger.info(f"Found {len(duplicates)} duplicate files")
    
    return unique_files, duplicates


def process_csv_file(
    input_file: Path,
    output_file: Path,
    validate: bool = True
) -> bool:
    """Process a single CSV file"""
    try:
        # Validate if requested
        if validate and not validate_csv(input_file):
            return False
        
        # Read CSV
        logger.info(f"Processing CSV: {input_file}")
        df = pd.read_csv(input_file)
        
        # Basic cleaning
        # Remove completely empty rows
        df = df.dropna(how='all')
        
        # Remove duplicate rows
        initial_rows = len(df)
        df = df.drop_duplicates()
        removed = initial_rows - len(df)
        if removed > 0:
            logger.info(f"  Removed {removed} duplicate rows")
        
        # Save processed CSV
        output_file.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_file, index=False)
        
        logger.info(f"✓ Processed: {input_file} -> {output_file} ({len(df)} rows)")
        return True
        
    except Exception as e:
        logger.error(f"Error processing {input_file}: {e}")
        return False


def process_datasets(
    source_dir: Path,
    output_dir: Path,
    format: str = 'csv',
    validate: bool = True,
    deduplicate: bool = True
):
    """Process all datasets in source directory"""
    
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all files
    if format == 'csv':
        files = list(source_dir.rglob('*.csv'))
    else:
        files = list(source_dir.rglob('*'))
    
    if not files:
        logger.warning(f"No {format} files found in {source_dir}")
        return
    
    logger.info(f"Found {len(files)} files to process")
    
    # Deduplicate if requested
    if deduplicate:
        unique_files, duplicates = deduplicate_files(files)
        files = unique_files
        
        # Remove duplicate files
        for dup in duplicates:
            logger.info(f"Removing duplicate: {dup}")
            dup.unlink()
    
    # Process files
    success_count = 0
    failed_count = 0
    
    for file_path in files:
        relative_path = file_path.relative_to(source_dir)
        output_file = output_dir / relative_path
        
        if format == 'csv':
            if process_csv_file(file_path, output_file, validate):
                success_count += 1
            else:
                failed_count += 1
        else:
            # For non-CSV files, just copy
            output_file.parent.mkdir(parents=True, exist_ok=True)
            import shutil
            shutil.copy2(file_path, output_file)
            logger.info(f"✓ Copied: {file_path} -> {output_file}")
            success_count += 1
    
    # Summary
    logger.info("=" * 60)
    logger.info(f"Processing Summary:")
    logger.info(f"  Success: {success_count}")
    logger.info(f"  Failed: {failed_count}")
    logger.info(f"  Output directory: {output_dir}")
    logger.info("=" * 60)
    
    # Generate metadata
    metadata = {
        'source_directory': str(source_dir),
        'output_directory': str(output_dir),
        'files_processed': success_count,
        'files_failed': failed_count,
        'format': format
    }
    
    metadata_file = output_dir / 'processing_metadata.json'
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Metadata saved to: {metadata_file}")


def main():
    parser = argparse.ArgumentParser(description='Process datasets for VaLLM training')
    parser.add_argument('--source', required=True, help='Source directory with datasets')
    parser.add_argument('--output', required=True, help='Output directory for processed datasets')
    parser.add_argument('--format', default='csv', choices=['csv', 'all'], help='File format to process')
    parser.add_argument('--validate', action='store_true', default=True, help='Validate files')
    parser.add_argument('--deduplicate', action='store_true', default=True, help='Remove duplicate files')
    
    args = parser.parse_args()
    
    source = Path(args.source)
    if not source.exists():
        logger.error(f"Source directory does not exist: {source}")
        sys.exit(1)
    
    process_datasets(
        source_dir=source,
        output_dir=Path(args.output),
        format=args.format,
        validate=args.validate,
        deduplicate=args.deduplicate
    )


if __name__ == '__main__':
    main()
