# Video LLM Analysis Platform

A comprehensive platform for analyzing video content using Google's Gemini models. This project enables downloading YouTube videos, processing them with Gemini Flash 2.0, and generating intelligent responses to questions about the videos.

![Video Analysis](https://img.shields.io/badge/Video-Analysis-blue)
![Gemini Flash 2.0](https://img.shields.io/badge/Gemini-Flash%202.0-orange)
![Python 3.8+](https://img.shields.io/badge/Python-3.8+-green)

## Overview

This platform streamlines the process of working with video datasets by:

1. Fetching datasets from HuggingFace
2. Downloading videos using pytubefix
3. Uploading videos to Google's File API
4. Performing inference with Gemini Flash 2.0
5. Saving results to CSV for analysis

The modular architecture separates concerns, making the workflow maintainable and efficient.

## Architecture

The system is composed of three main components:

```
┌───────────────────┐      ┌───────────────────┐      ┌───────────────────┐
│  fetch_dataset.py │ ──► │ download_upload.py │ ──► │   inference.py    │
└───────────────────┘      └───────────────────┘      └───────────────────┘
       │                         │                          │
       ▼                         ▼                          ▼
┌─────────────┐         ┌─────────────────┐        ┌─────────────────┐
│ dataset.csv │         │ videos + metadata│        │   results.csv   │
└─────────────┘         └─────────────────┘        └─────────────────┘
```

## Features

- **Efficient Processing**: Checks for previously downloaded/uploaded videos to avoid duplication
- **Robust Error Handling**: Automatic retries with exponential backoff
- **Interactive Progress**: Detailed logging and progress bars
- **Flexible Configuration**: Extensive command-line options
- **Metadata Tracking**: Comprehensive tracking of video data
- **Modular Design**: Separates fetching, downloading, and inference
- **Resume Support**: Can be stopped and resumed at any point

## Installation

### Prerequisites

- Python 3.8+
- [HuggingFace Account](https://huggingface.co/) with access to the dataset
- [Google API Key](https://ai.google.dev/) with access to Gemini models

### Steps

1. Clone this repository:
   ```bash
   git clone https://github.com/dadevchia/tiktokllm.git
   cd tiktokllm
   ```

2. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Log in to HuggingFace (required to access datasets):
   ```bash
   huggingface-cli login
   ```

4. Create a `.env` file with your API credentials:
   ```bash
   touch .env
   ```

5. Add the following to your `.env` file:
   ```
   GEMINI_API_KEY=your_google_api_key_here
   ```

## Usage

### Step 1: Fetch the Dataset

```bash
# Basic usage
python fetch_dataset.py

# Advanced usage
python fetch_dataset.py --split test --output custom_dataset.csv --cache-dir ./hf_cache
```

**Options:**
- `--output`: Output CSV file path (default: dataset.csv)
- `--split`: Dataset split to use (default: test)
- `--cache-dir`: Cache directory for HuggingFace datasets

### Step 2: Download and Upload Videos

```bash
# Basic usage
python download_upload.py

# Advanced usage
python download_upload.py --batch-size 20 --input-csv custom_dataset.csv
```

**Options:**
- `--api-key`: Your Google Generative AI API key (uses .env by default)
- `--batch-size`: Number of videos to process in one batch (default: 10)
- `--start-index`: Index to start from in the dataset (default: 0)
- `--max-videos`: Maximum number of videos to process (default: all)
- `--input-csv`: Path to input CSV file (default: dataset.csv)

### Step 3: Perform Inference

```bash
# Basic usage
python inference.py

# Advanced usage
python inference.py --model gemini-2.0-flash --retry 5 --output custom_results.csv
```

**Options:**
- `--api-key`: Your Google Generative AI API key (uses .env by default)
- `--model`: Model name to use (default: gemini-2.0-flash)
- `--retry`: Number of retries for failed inferences (default: 3)
- `--start-index`: Index to start from in the list of videos (default: 0)
- `--max-videos`: Maximum number of videos to process (default: all)
- `--output`: Output CSV file for results (default: results.csv)

## Output Files

- `dataset.csv`: HuggingFace dataset in CSV format
- `video_metadata.csv`: Metadata about downloaded and uploaded videos
- `downloaded_videos.json`: Tracking info for downloaded and uploaded videos
- `results.csv`: Final inference results with qid and pred columns
- `videos/`: Directory containing downloaded videos organized by QID prefix

## Example Workflow

A typical workflow might look like:

```bash
# 1. Set up environment and fetch dataset
python fetch_dataset.py

# 2. Download first 10 videos for testing
python download_upload.py --max-videos 10

# 3. Run inference on these 10 videos
python inference.py --max-videos 10

# 4. Process the remaining videos in batches
python download_upload.py --start-index 10 --batch-size 20
python inference.py --start-index 10
```

## Troubleshooting

### Common Issues

1. **Video Download Failures**
   - Check internet connection
   - Verify video still exists on YouTube
   - Try updating pytubefix: `pip install --upgrade pytubefix`

2. **Google API Errors**
   - Verify your API key is correct in the .env file
   - Check if you have access to the specified model
   - Ensure your Google Cloud billing is properly set up

3. **HuggingFace Access Issues**
   - Run `huggingface-cli login` again
   - Verify you have access to the dataset
   - Check your internet connection

### Logs

All scripts create detailed logs in the console. For persistent logs, redirect output:

```bash
python download_upload.py > download_log.txt 2>&1
```

## Project Structure

```
tiktokllm/
├── fetch_dataset.py     # Script to fetch HuggingFace dataset
├── download_upload.py   # Script to download/upload videos
├── inference.py         # Script for Gemini inference
├── .env                 # Environment variables (API keys)
├── requirements.txt     # Project dependencies
├── dataset.csv          # Generated by fetch_dataset.py
├── video_metadata.csv   # Generated by download_upload.py
├── downloaded_videos.json # Generated by download_upload.py
├── results.csv          # Generated by inference.py
└── videos/              # Downloaded video files
    └── [QID_PREFIX]/    # Organized by QID prefix
```

## Best Practices

- Start with a small batch (e.g., `--max-videos 5`) to test your setup
- Run scripts using screen or tmux for long-running processes
- Regularly backup your metadata and tracking files
- Monitor disk space, especially if downloading many videos
- Consider setting up a cron job for large datasets

## Contributors

[Your Name/Organization]

## License

[Specify your license here]