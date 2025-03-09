import os
import logging
import argparse
import pandas as pd
from datasets import load_dataset
from tqdm.auto import tqdm
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
DATASET_CSV = "dataset.csv"

def setup_argparse():
    """Setup command line arguments."""
    parser = argparse.ArgumentParser(description="Fetch dataset from HuggingFace")
    parser.add_argument("--output", type=str, default=DATASET_CSV, help="Output CSV file path")
    parser.add_argument("--split", type=str, default="test", help="Dataset split to use (train, validation, test)")
    parser.add_argument("--cache-dir", type=str, default=None, help="Cache directory for HuggingFace datasets")
    return parser.parse_args()

def fetch_dataset(split="test", cache_dir=None):
    """
    Fetch the dataset from HuggingFace using the datasets library.
    
    Args:
        split: Dataset split to use (train, validation, test)
        cache_dir: Cache directory for HuggingFace datasets
    
    Returns:
        The dataset object
    """
    logger.info(f"Fetching dataset 'lmms-lab/AISG_Challenge' with split '{split}'...")
    
    # Load dataset from HuggingFace
    try:
        dataset = load_dataset("lmms-lab/AISG_Challenge", split=split, cache_dir=cache_dir)
        logger.info(f"Successfully loaded dataset with {len(dataset)} entries")
        return dataset
    except Exception as e:
        logger.error(f"Error loading dataset: {str(e)}")
        raise

def dataset_to_csv(dataset, output_file=DATASET_CSV):
    """
    Convert the dataset to a CSV file.
    
    Args:
        dataset: HuggingFace dataset object
        output_file: Output CSV file path
    """
    logger.info(f"Converting dataset to CSV and saving to {output_file}...")
    
    # Convert to pandas DataFrame and save as CSV
    try:
        df = pd.DataFrame(dataset)
        
        # Clean the dataframe to handle potential encoding issues
        for col in df.columns:
            if df[col].dtype == 'object':  # String columns
                # Replace problematic characters
                df[col] = df[col].apply(lambda x: x.encode('utf-8', 'replace').decode('utf-8') if isinstance(x, str) else x)
        
        df.to_csv(output_file, index=False, encoding='utf-8')
        logger.info(f"Successfully saved dataset to {output_file}")
        
        # Print column names and sample rows for verification
        logger.info(f"Dataset columns: {', '.join(df.columns)}")
        logger.info(f"Sample of first entry:")
        for col in df.columns:
            if len(df) > 0:
                sample_value = str(df.iloc[0][col])
                if len(sample_value) > 100:
                    sample_value = sample_value[:100] + "..."
                logger.info(f"  {col}: {sample_value}")
        
        return output_file
    except Exception as e:
        logger.error(f"Error saving dataset to CSV: {str(e)}")
        # Try backup approach with minimal columns
        try:
            logger.info("Attempting to save with minimal columns...")
            essential_cols = ['qid', 'video_id', 'question', 'youtube_url']
            available_cols = [col for col in essential_cols if col in dataset.column_names]
            
            df = pd.DataFrame(dataset)[available_cols]
            df.to_csv(f"{output_file}.minimal", index=False, encoding='utf-8')
            logger.info(f"Saved minimal dataset to {output_file}.minimal")
            return f"{output_file}.minimal"
        except Exception as backup_err:
            logger.error(f"Backup save failed: {str(backup_err)}")
            raise e

def main():
    """Main function to fetch dataset and save as CSV."""
    # Parse command line arguments
    args = setup_argparse()
    
    # Fetch dataset
    dataset = fetch_dataset(split=args.split, cache_dir=args.cache_dir)
    
    # Save to CSV
    dataset_to_csv(dataset, args.output)
    
    logger.info("Done!")

if __name__ == "__main__":
    main()