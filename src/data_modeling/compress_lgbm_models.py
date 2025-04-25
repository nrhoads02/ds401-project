# compress_lgbm_models.py
# ---------------------------------------------------------------
# Utility to compress existing LightGBM models using zlib
# Locates all .lgb files in the model directory,
# compresses them using zlib, and saves as .lgb.z files
# ---------------------------------------------------------------

import os
import zlib
import glob
import logging
import argparse
from typing import Dict, List, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("compress_lgbm")

def compress_model_file(input_path: str, output_path: str = None, 
                        compression_level: int = 9) -> Tuple[str, float, float]:
    """
    Compresses a LightGBM model file using zlib.
    
    Args:
        input_path: Path to the original .lgb file
        output_path: Path for the compressed file (defaults to input_path + '.z')
        compression_level: Zlib compression level (1-9)
        
    Returns:
        Tuple of (output_path, original_size_mb, compressed_size_mb)
    """
    if output_path is None:
        output_path = f"{input_path}.z"
    
    try:
        with open(input_path, 'rb') as f:
            model_data = f.read()
        
        original_size = len(model_data)
        compressed_data = zlib.compress(model_data, level=compression_level)
        
        with open(output_path, 'wb') as f:
            f.write(compressed_data)
        
        compressed_size = os.path.getsize(output_path)
        
        original_size_mb = original_size / (1024 * 1024)
        compressed_size_mb = compressed_size / (1024 * 1024)
        
        return output_path, original_size_mb, compressed_size_mb
    
    except Exception as e:
        logger.error(f"Error compressing {input_path}: {e}")
        return None, 0, 0

def find_model_files(model_dir: str, pattern: str = "*.lgb") -> List[str]:
    """
    Find all model files matching the pattern in the directory.
    
    Args:
        model_dir: Directory to search
        pattern: File pattern to match
        
    Returns:
        List of file paths
    """
    search_path = os.path.join(model_dir, "**", pattern)
    return glob.glob(search_path, recursive=True)

def compress_all_models(model_dir: str, dry_run: bool = False, 
                        skip_existing: bool = True) -> Dict[str, Dict[str, float]]:
    """
    Compress all .lgb model files in the given directory.
    
    Args:
        model_dir: Root directory containing model files
        dry_run: If True, only report what would be done without writing files
        skip_existing: If True, skip compression if .z file already exists
        
    Returns:
        Dictionary with statistics about the compressed files
    """
    model_files = find_model_files(model_dir)
    logger.info(f"Found {len(model_files)} model files to compress")
    
    results = {}
    total_original = 0
    total_compressed = 0
    
    for input_path in model_files:
        output_path = f"{input_path}.z"
        
        # Skip if compressed file already exists and skip_existing is True
        if os.path.exists(output_path) and skip_existing:
            logger.info(f"Skipping {input_path} - compressed file already exists")
            original_size = os.path.getsize(input_path) / (1024 * 1024)
            compressed_size = os.path.getsize(output_path) / (1024 * 1024)
            
            results[input_path] = {
                "original_mb": original_size,
                "compressed_mb": compressed_size,
                "ratio": original_size / compressed_size if compressed_size > 0 else 0,
                "status": "skipped"
            }
            total_original += original_size
            total_compressed += compressed_size
            continue
        
        if dry_run:
            logger.info(f"Would compress: {input_path} -> {output_path}")
            continue
        
        logger.info(f"Compressing: {input_path}")
        output_path, original_mb, compressed_mb = compress_model_file(input_path, output_path)
        
        if output_path:
            ratio = original_mb / compressed_mb if compressed_mb > 0 else 0
            logger.info(f"  Original: {original_mb:.2f} MB, Compressed: {compressed_mb:.2f} MB, Ratio: {ratio:.2f}x")
            
            results[input_path] = {
                "original_mb": original_mb,
                "compressed_mb": compressed_mb,
                "ratio": ratio,
                "status": "success"
            }
            total_original += original_mb
            total_compressed += compressed_mb
        else:
            results[input_path] = {"status": "error"}
    
    # Log summary
    if not dry_run and total_compressed > 0:
        logger.info(f"Summary: Total original: {total_original:.2f} MB, " 
                    f"Total compressed: {total_compressed:.2f} MB, "
                    f"Overall ratio: {total_original/total_compressed:.2f}x")
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Compress LightGBM model files using zlib")
    parser.add_argument("--model_dir", type=str, default="data/models/surface_lgbm",
                       help="Directory containing model files")
    parser.add_argument("--dry-run", action="store_true", 
                       help="Only report what would be done without writing files")
    parser.add_argument("--force", action="store_true",
                       help="Overwrite existing compressed files")
    parser.add_argument("--level", type=int, default=9, choices=range(1, 10),
                       help="Zlib compression level (1-9)")
    
    args = parser.parse_args()
    
    logger.info(f"Starting model compression in {args.model_dir}")
    results = compress_all_models(
        args.model_dir, 
        dry_run=args.dry_run, 
        skip_existing=not args.force
    )
    
    # Display results
    if not args.dry_run:
        logger.info("Compression results by file:")
        for path, stats in results.items():
            if stats["status"] == "success":
                logger.info(f"  {os.path.basename(path)}: {stats['original_mb']:.2f} MB -> "
                            f"{stats['compressed_mb']:.2f} MB ({stats['ratio']:.2f}x)")
    
    logger.info("Model compression complete!")

if __name__ == "__main__":
    main()