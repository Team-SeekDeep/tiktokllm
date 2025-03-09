import os
import csv
import json
import logging
import time
import argparse
from typing import Dict, List, Optional
from tqdm.auto import tqdm
from google import genai
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
METADATA_FILE = "video_metadata.csv"
RESULTS_FILE = "results.csv"
VIDEO_TRACKING_FILE = "downloaded_videos.json"

BATCH_SIZE = 100  # Number of videos to process in one batch
INFERENCE_RATE_LIMIT = 15  # Number of inferences per minute / set to 0 for no limit

def setup_argparse():
    """Setup command line arguments."""
    parser = argparse.ArgumentParser(description="Perform inference on videos using Google's Generative AI API")
    parser.add_argument("--api-key", type=str, help="Google Generative AI API key")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Number of videos to process in one batch")
    parser.add_argument("--start-index", type=int, default=0, help="Starting index in the list of videos to process")
    parser.add_argument("--max-videos", type=int, default=None, help="Maximum number of videos to process")
    parser.add_argument("--model", type=str, default="gemini-2.0-flash", help="Model name to use for inference")
    parser.add_argument("--retry", type=int, default=3, help="Number of retries for failed inferences")
    parser.add_argument("--output", type=str, default=RESULTS_FILE, help="Output CSV file for results")
    return parser.parse_args()

def get_api_key(args):
    """Get API key from arguments or environment variables."""
    api_key = args.api_key or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("Please provide a Google Generative AI API key via --api-key or GEMINI_API_KEY environment variable")
    return api_key

def load_metadata() -> List[Dict]:
    """
    Load video metadata from CSV.
    
    Returns:
        List[Dict]: A list of dictionaries with video metadata.
    """
    if not os.path.exists(METADATA_FILE):
        # Check if backup exists
        if os.path.exists(f"{METADATA_FILE}.backup"):
            logger.warning(f"Main metadata file not found, using backup file {METADATA_FILE}.backup")
            metadata_file = f"{METADATA_FILE}.backup"
        else:
            raise FileNotFoundError(f"Metadata file {METADATA_FILE} not found. Run download_upload.py first.")
    else:
        metadata_file = METADATA_FILE
    
    metadata = []
    try:
        with open(metadata_file, 'r', newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                # Only include videos that have been uploaded to Google API
                if "google_uri" in row and "google_name" in row:
                    metadata.append(row)
        
        logger.info(f"Loaded metadata for {len(metadata)} videos")
        return metadata
    except UnicodeDecodeError:
        # Try with a different encoding if UTF-8 fails
        logger.warning("UTF-8 decoding failed, trying with cp1252 encoding")
        with open(metadata_file, 'r', newline='', encoding='cp1252') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                if "google_uri" in row and "google_name" in row:
                    metadata.append(row)
        
        logger.info(f"Loaded metadata for {len(metadata)} videos with cp1252 encoding")
        return metadata

def load_video_tracking() -> Dict:
    """Load tracking info of uploaded videos."""
    if os.path.exists(VIDEO_TRACKING_FILE):
        with open(VIDEO_TRACKING_FILE, 'r') as f:
            return json.load(f)
    return {"downloaded": {}, "uploaded": {}}

def load_existing_results(results_file: str) -> Dict[str, str]:
    """
    Load existing results from CSV.
    
    Returns:
        Dict[str, str]: A dictionary mapping QID to predicted answer.
    """
    results = {}
    if os.path.exists(results_file):
        try:
            with open(results_file, 'r', newline='', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    if "qid" in row and "pred" in row:
                        results[row["qid"]] = row["pred"]
        except UnicodeDecodeError:
            # Try with a different encoding if UTF-8 fails
            logger.warning("UTF-8 decoding failed for results file, trying with cp1252 encoding")
            with open(results_file, 'r', newline='', encoding='cp1252') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    if "qid" in row and "pred" in row:
                        results[row["qid"]] = row["pred"]
    
    logger.info(f"Loaded {len(results)} existing results")
    return results

def save_result(qid: str, pred: str, results_file: str) -> None:
    """
    Save a single result to the CSV file.
    
    Args:
        qid (str): The QID for the video.
        pred (str): The predicted answer.
        results_file (str): Path to results CSV file.
    """
    file_exists = os.path.exists(results_file)
    
    try:
        # Clean the prediction to ensure it can be encoded
        if pred is not None and isinstance(pred, str):
            # Replace problematic characters with their closest equivalents
            pred = pred.encode('utf-8', 'replace').decode('utf-8')
        
        with open(results_file, 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=["qid", "pred"])
            
            if not file_exists:
                writer.writeheader()
            
            writer.writerow({"qid": qid, "pred": pred})
        
        logger.info(f"Saved result for QID {qid}")
    except Exception as e:
        logger.error(f"Error saving result for QID {qid}: {str(e)}")
        # Try backup approach - save to separate file with just qid
        try:
            with open(f"{results_file}.incomplete", 'a', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=["qid"])
                if not os.path.exists(f"{results_file}.incomplete"):
                    writer.writeheader()
                writer.writerow({"qid": qid})
            logger.warning(f"Saved QID {qid} to incomplete results file (without prediction)")
        except Exception as backup_err:
            logger.error(f"Even backup save failed: {str(backup_err)}")

def verify_file_exists(tracking: Dict, qid: str, client) -> bool:
    """
    Verify that a file exists in Google's File API and is active.
    
    Args:
        tracking: Dictionary tracking uploaded videos.
        qid: QID for the video.
        client: Google Generative AI client.
    
    Returns:
        bool: True if the file exists and is active, False otherwise.
    """
    if qid not in tracking["uploaded"]:
        logger.warning(f"No upload record for QID {qid}")
        return False
    
    try:
        file_name = tracking["uploaded"][qid]["name"]
        file = client.files.get(name=file_name)
        
        if file.state.name != "ACTIVE":
            logger.warning(f"File for QID {qid} exists but state is {file.state.name}")
            return False
        
        logger.info(f"Verified file for QID {qid} exists and is ACTIVE")
        return True
    except Exception as e:
        logger.warning(f"Error verifying file for QID {qid}: {str(e)}")
        return False

def perform_inference(
    video_info: Dict, 
    tracking: Dict, 
    client, 
    model_name: str,
    max_retries: int = 3
) -> Optional[str]:
    """
    Perform inference on a video using Google's Generative AI API.
    
    Args:
        video_info: Information about the video.
        tracking: Dictionary tracking uploaded videos.
        client: Google Generative AI client.
        model_name: Name of the model to use.
        max_retries: Maximum number of retries for failed inferences.
    
    Returns:
        Optional[str]: The predicted answer, or None if inference failed.
    """
    qid = video_info["qid"]
    question = video_info["question"]
    question_prompt = video_info.get("question_prompt", "")

    sleep_time = 60 / INFERENCE_RATE_LIMIT if INFERENCE_RATE_LIMIT > 0 else 0
    
    # Verify the file exists in Google's File API
    if not verify_file_exists(tracking, qid, client):
        logger.error(f"File for QID {qid} not found or not ACTIVE in Google's File API")
        return None
    
    # Get file reference
    file_name = tracking["uploaded"][qid]["name"]
    file_ref = client.files.get(name=file_name)
    
    # Construct the prompt
    prompt = question
    if question_prompt and question_prompt.strip():
        if not prompt.endswith("?") and not prompt.endswith("."):
            prompt += "."
        prompt += f" {question_prompt}"
    
    for attempt in range(max_retries + 1):
        try:
            logger.info(f"Performing inference on video with QID {qid} (attempt {attempt+1}/{max_retries+1})")
            logger.info(f"Question: {prompt}")
            
            # Call the API
            response = client.models.generate_content(
                model=model_name,
                contents=[file_ref, prompt]
            )
            
            # Extract the answer
            answer = response.text.strip()
            if answer:
                logger.info(f"Got answer: {answer}")
                time.sleep(sleep_time)
                return answer
            else:
                logger.warning(f"Empty response for QID {qid}")
                if attempt < max_retries:
                    time.sleep(sleep_time)
        except Exception as e:
            logger.error(f"Error performing inference on video with QID {qid}: {str(e)}")
            if attempt < max_retries:
                time.sleep(4 * sleep_time)
    
    logger.error(f"All {max_retries+1} attempts failed for QID {qid}")
    return None

def process_videos(
    videos: List[Dict], 
    tracking: Dict, 
    existing_results: Dict[str, str], 
    client, 
    model_name: str,
    results_file: str,
    max_retries: int = 3
) -> None:
    """
    Process videos in sequence.
    
    Args:
        videos: List of videos to process.
        tracking: Dictionary tracking uploaded videos.
        existing_results: Dictionary of existing results.
        client: Google Generative AI client.
        model_name: Name of the model to use.
        results_file: Path to results CSV file.
        max_retries: Maximum number of retries for failed inferences.
    """
    for video_info in tqdm(videos, desc="Processing videos"):
        qid = video_info["qid"]
        
        # Check if already processed
        if qid in existing_results:
            logger.info(f"QID {qid} already processed, skipping")
            continue
        
        # Perform inference
        answer = perform_inference(video_info, tracking, client, model_name, max_retries)
        
        # Save result if successful
        if answer:
            save_result(qid, answer, results_file)
        else:
            logger.warning(f"No answer obtained for QID {qid}")

def main():
    """Main function to perform inference on videos."""
    # Parse command line arguments
    args = setup_argparse()
    api_key = get_api_key(args)
    model_name = args.model
    results_file = args.output
    
    # Initialize Google Generative AI client
    genai_client = genai.Client(api_key=api_key)
    
    # Load metadata, tracking, and existing results
    video_metadata = load_metadata()
    tracking = load_video_tracking()
    existing_results = load_existing_results(results_file)
    
    # Determine which videos to process
    start_idx = args.start_index
    max_videos = args.max_videos
    
    if max_videos is not None:
        end_idx = min(start_idx + max_videos, len(video_metadata))
    else:
        end_idx = len(video_metadata)
    
    videos_to_process = video_metadata[start_idx:end_idx]
    
    # Print summary of what we'll be doing
    num_videos = len(videos_to_process)
    logger.info(f"Processing {num_videos} videos from index {start_idx} to {end_idx-1}")
    logger.info(f"Using model: {model_name}")
    logger.info(f"Maximum retries per video: {args.retry}")
    logger.info(f"Results will be saved to: {results_file}")
    
    # Confirm with user
    if num_videos > 0:
        confirm = input(f"Ready to process {num_videos} videos? (y/n): ")
        if confirm.lower() != 'y':
            logger.info("Operation cancelled by user")
            return
    else:
        logger.info("No videos to process")
        return
    
    # Process the videos
    process_videos(
        videos_to_process, 
        tracking, 
        existing_results, 
        genai_client, 
        model_name,
        results_file,
        args.retry
    )
    
    # Print summary of results
    if os.path.exists(results_file):
        with open(results_file, 'r', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            results_count = sum(1 for _ in reader)
        
        total_videos = len(video_metadata)
        completion_percentage = (results_count / total_videos) * 100 if total_videos > 0 else 0
        
        logger.info(f"Results summary:")
        logger.info(f"  Total results: {results_count}")
        logger.info(f"  Total videos: {total_videos}")
        logger.info(f"  Completion: {completion_percentage:.2f}%")
    
    logger.info("Finished processing videos")

if __name__ == "__main__":
    main()