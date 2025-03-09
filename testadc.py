from google import genai
from google.genai.types import HttpOptions, Part

from dotenv import load_dotenv

load_dotenv()

client = genai.Client()
response = client.models.generate_content(
    model="gemini-2.0-flash-001",
    contents=[
        "What is shown in this image?",
        Part.from_uri(
            file_uri="gs://cloud-samples-data/generative-ai/image/scones.jpg",
            mime_type="image/jpeg",
        ),
    ],
)
print(response.text)
# Example response:
# The image shows a flat lay of blueberry scones arranged on parchment paper. There are ...