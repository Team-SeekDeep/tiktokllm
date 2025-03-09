import os
import json
import logging
import time
import pandas as pd
from typing import Dict, List, Optional
from pytubefix import YouTube
from pytubefix.cli import on_progress
from google import genai
import csv
import argparse
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
VIDEOS_DIR = "videos"
METADATA_FILE = "video_metadata.csv"
VIDEO_TRACKING_FILE = "downloaded_videos.json"

def setup_argparse():
    """Setup command line arguments."""
    parser = argparse.ArgumentParser(description="Download videos and upload them to Google's File API")
    parser.add_argument("--api-key", type=str, help="Google Generative AI API key")
    parser.add_argument("--batch-size", type=int, default=10, help="Number of videos to process in one batch")
    parser.add_argument("--start-index", type=int, default=0, help="Starting index in the dataset")
    parser.add_argument("--max-videos", type=int, default=None, help="Maximum number of videos to process")
    parser.add_argument("--input-csv", type=str, default=DATASET_CSV, help="Input CSV file with dataset")
    return parser.parse_args()

def get_api_key(args):
    """Get API key from arguments or environment variables."""
    api_key = args.api_key or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("Please provide a Google Generative AI API key via --api-key or GEMINI_API_KEY environment variable")
    return api_key

def load_dataset_from_csv(csv_path: str, start_idx: int = 0, limit: Optional[int] = None) -> List[Dict]:
    """
    Load the dataset from a CSV file.
    
    Args:
        csv_path: Path to the CSV file.
        start_idx: Starting index for pagination.
        limit: Maximum number of rows to fetch.
    
    Returns:
        List[Dict]: A list of dictionaries with video information.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset CSV file {csv_path} not found. Run fetch_dataset.py first.")
    
    logger.info(f"Loading dataset from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Apply pagination if specified
    if limit is not None:
        df = df.iloc[start_idx:start_idx + limit]
    else:
        df = df.iloc[start_idx:]
    
    logger.info(f"Loaded {len(df)} entries from dataset")
    
    # Convert to list of dictionaries
    videos = []
    for _, row in df.iterrows():
        row_dict = row.to_dict()
        
        # Extract QID for folder organization
        qid = row_dict.get("qid", "")
        # Extract the first 4 digits of the QID for organizing videos
        qid_prefix = qid.split("-")[0] if "-" in qid else qid[:4]
        
        # Check if youtube_url is valid, if not construct from video_id
        video_id = row_dict.get("video_id", "")
        youtube_url = row_dict.get("youtube_url", "")
        if not youtube_url or not youtube_url.startswith("http"):
            youtube_url = f"https://www.youtube.com/watch?v={video_id}"
            row_dict["youtube_url"] = youtube_url
        
        # Add folder information
        row_dict["folder"] = qid_prefix
        
        videos.append(row_dict)
    
    return videos

def ensure_directory(directory: str) -> None:
    """Ensure the directory exists."""
    if not os.path.exists(directory):
        os.makedirs(directory)
        logger.info(f"Created directory: {directory}")

def load_video_tracking() -> Dict:
    """Load tracking info of downloaded videos."""
    if os.path.exists(VIDEO_TRACKING_FILE):
        with open(VIDEO_TRACKING_FILE, 'r') as f:
            return json.load(f)
    return {"downloaded": {}, "uploaded": {}}

def save_video_tracking(tracking: Dict) -> None:
    """Save tracking info of downloaded videos."""
    with open(VIDEO_TRACKING_FILE, 'w') as f:
        json.dump(tracking, f, indent=2)

def load_metadata() -> Dict[str, Dict]:
    """Load existing video metadata from CSV."""
    metadata = {}
    if os.path.exists(METADATA_FILE):
        with open(METADATA_FILE, 'r', newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                if "qid" in row:
                    metadata[row["qid"]] = row
    return metadata

def save_metadata(metadata: Dict[str, Dict]) -> None:
    """Save video metadata to CSV."""
    if not metadata:
        return
    
    # Get all possible field names from the metadata dictionaries
    fieldnames = set()
    for info in metadata.values():
        fieldnames.update(info.keys())
    fieldnames = sorted(list(fieldnames))
    
    try:
        with open(METADATA_FILE, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for info in metadata.values():
                # Clean dictionary values to ensure they can be encoded
                cleaned_info = {}
                for key, value in info.items():
                    if value is not None:
                        if isinstance(value, str):
                            # Replace problematic characters with their closest equivalents
                            cleaned_info[key] = value.encode('utf-8', 'replace').decode('utf-8')
                        else:
                            cleaned_info[key] = value
                    else:
                        cleaned_info[key] = ""
                writer.writerow(cleaned_info)
        logger.info(f"Successfully saved metadata to {METADATA_FILE}")
    except Exception as e:
        logger.error(f"Error saving metadata: {str(e)}")
      
def download_video(video_info: Dict, tracking: Dict) -> Optional[str]:
    """
    Download a video if it doesn't already exist.
    
    Args:
        video_info: Information about the video.
        tracking: Dictionary tracking downloaded videos.
    
    Returns:
        Optional[str]: The path to the downloaded video, or None if download failed.
    """
    qid = video_info["qid"]
    video_id = video_info["video_id"]
    folder = video_info["folder"]
    
    # Create directory structure
    video_dir = os.path.join(VIDEOS_DIR, folder)
    ensure_directory(video_dir)
    
    # Define output filename and path
    filename = f"{qid}_{video_id}.mp4"
    output_path = os.path.join(video_dir, filename)
    
    # Check if already downloaded for this QID
    if qid in tracking["downloaded"]:
        local_path = tracking["downloaded"][qid]
        if os.path.exists(local_path):
            logger.info(f"Video for QID {qid} already downloaded at {local_path}")
            return local_path
    
    # Check if video_id already downloaded for another QID
    for existing_qid, path in tracking["downloaded"].items():
        if video_id in path and os.path.exists(path):
            logger.info(f"Video ID {video_id} already downloaded for QID {existing_qid}, reusing for QID {qid}")
            tracking["downloaded"][qid] = path
            return path
    
    # Download the video
    youtube_url = video_info["youtube_url"]
    try:
        logger.info(f"Downloading video for QID {qid} from {youtube_url}")
        yt = YouTube(youtube_url, on_progress_callback=on_progress)
        logger.info(f"Title: {yt.title} (Duration: {yt.length}s)")
        
        # Get the highest resolution stream
        stream = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first()
        
        if not stream:
            logger.warning(f"No suitable stream found for QID {qid}")
            return None
        
        stream.download(output_path=video_dir, filename=filename)
        logger.info(f"Downloaded video for QID {qid} to {output_path}")
        
        # Update tracking
        tracking["downloaded"][qid] = output_path
        save_video_tracking(tracking)
        
        return output_path
    except Exception as e:
        logger.error(f"Error downloading video for QID {qid}: {str(e)}")
        return None

def upload_to_google_api(video_path: str, qid: str, video_id: str, tracking: Dict, client) -> Optional[str]:
    """
    Upload a video to Google's File API if it hasn't been uploaded.
    
    Args:
        video_path: The path to the video file.
        qid: The QID for the video.
        video_id: The video ID.
        tracking: Dictionary tracking uploaded videos.
        client: Google Generative AI client.
    
    Returns:
        Optional[str]: The URI of the uploaded file, or None if upload failed.
    """
    # Check if this QID has already been uploaded
    if qid in tracking["uploaded"]:
        uri_info = tracking["uploaded"][qid]
        
        # Verify the file still exists in Google's File API
        try:
            file = client.files.get(name=uri_info["name"])
            if file.state.name == "ACTIVE":
                logger.info(f"Video for QID {qid} already uploaded with URI {uri_info['uri']}")
                return uri_info["uri"]
            else:
                logger.warning(f"File exists but state is {file.state.name}, re-uploading")
        except Exception as e:
            logger.warning(f"File not found in Google's File API, re-uploading: {str(e)}")
    
    # Check if video_id already uploaded for another QID
    for existing_qid, uri_info in tracking["uploaded"].items():
        if video_id in existing_qid or video_id in uri_info.get("video_id", ""):
            try:
                file = client.files.get(name=uri_info["name"])
                if file.state.name == "ACTIVE":
                    logger.info(f"Video ID {video_id} already uploaded for QID {existing_qid}, reusing for QID {qid}")
                    tracking["uploaded"][qid] = uri_info
                    save_video_tracking(tracking)
                    return uri_info["uri"]
            except Exception:
                pass
    
    # Upload to Google's File API
    try:
        logger.info(f"Uploading video for QID {qid} to Google's File API")
        video_file = client.files.upload(file=video_path)
        uri = video_file.uri
        
        # Wait for the file to be processed
        logger.info(f"Waiting for file to be processed...")
        while video_file.state.name == "PROCESSING":
            time.sleep(2)
            video_file = client.files.get(name=video_file.name)
            logger.info(f"File state: {video_file.state.name}")
        
        if video_file.state.name != "ACTIVE":
            logger.error(f"File upload failed: {video_file.state.name}")
            return None
        
        logger.info(f"File {uri} is now ACTIVE and ready for inference")
        
        # Update tracking
        tracking["uploaded"][qid] = {
            "uri": uri,
            "name": video_file.name,
            "video_id": video_id
        }
        save_video_tracking(tracking)
        
        return uri
    except Exception as e:
        logger.error(f"Error uploading video for QID {qid} to Google API: {str(e)}")
        return None

def update_metadata(metadata: Dict[str, Dict], video_info: Dict, local_path: str, google_uri: str, google_name: str) -> None:
    """Update metadata with download and upload information."""
    qid = video_info["qid"]
    
    if qid not in metadata:
        metadata[qid] = {}
    
    metadata[qid].update(video_info)
    metadata[qid]["local_path"] = local_path
    metadata[qid]["google_uri"] = google_uri
    metadata[qid]["google_name"] = google_name

def process_videos(videos: List[Dict], metadata: Dict[str, Dict], tracking: Dict, client, batch_size: int = 10) -> None:
    """Process videos in batches."""
    for i, video_info in enumerate(tqdm(videos, desc="Processing videos")):
        qid = video_info["qid"]
        video_id = video_info["video_id"]
        
        logger.info(f"Processing video {i+1}/{len(videos)} with QID {qid}")
        
        # Download video
        local_path = download_video(video_info, tracking)
        if not local_path:
            logger.warning(f"Skipping upload for QID {qid} as download failed")
            continue
        
        # Upload to Google API
        google_uri = upload_to_google_api(local_path, qid, video_id, tracking, client)
        if not google_uri:
            logger.warning(f"Failed to upload video for QID {qid} to Google API")
            continue
        
        # Get the Google file name from the URI
        google_name = tracking["uploaded"][qid]["name"]
        
        # Update metadata
        update_metadata(metadata, video_info, local_path, google_uri, google_name)
        
        # Save metadata periodically
        if (i + 1) % batch_size == 0 or i == len(videos) - 1:
            save_metadata(metadata)
            logger.info(f"Saved metadata for {len(metadata)} videos")

def main():
    """Main function to download and upload videos."""
    # Parse command line arguments
    args = setup_argparse()
    api_key = get_api_key(args)
    
    # Initialize Google Generative AI client
    genai_client = genai.Client(api_key=api_key)
    
    # Ensure videos directory exists
    ensure_directory(VIDEOS_DIR)
    
    # Load existing metadata and tracking
    metadata = load_metadata()
    tracking = load_video_tracking()
    
    logger.info(f"Loaded metadata for {len(metadata)} videos")
    logger.info(f"Found {len(tracking['downloaded'])} previously downloaded videos")
    logger.info(f"Found {len(tracking['uploaded'])} previously uploaded videos to Google API")
    
    # Load dataset from CSV
    start_idx = args.start_index
    batch_size = args.batch_size
    max_videos = args.max_videos
    
    videos = load_dataset_from_csv(args.input_csv, start_idx, max_videos)
    
    if not videos:
        logger.info("No videos to process")
        return
    
    logger.info(f"Processing {len(videos)} videos starting at index {start_idx}")
    
    # Process videos
    process_videos(videos, metadata, tracking, genai_client, batch_size)
    
    logger.info(f"Total processed: {len(videos)} videos")
    logger.info("Finished downloading and uploading videos")

if __name__ == "__main__":
    main()