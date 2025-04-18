# Paper to Speech

An AI-powered web application that converts academic papers and documents to speech, using text processing and natural-sounding speech synthesis.

## Features

- **Document Processing**: Upload PDF or Word (.docx) documents 
- **AI Text Processing**: Uses Gemini AI to:
  - Clean and sanitize text for better speech quality
  - Generate structured summaries with background, methods, results, and conclusions
- **High-Quality Speech Synthesis**: Converts text to speech using Kokoro TTS with natural-sounding voices
- **Processing Modes**:
  - Full Text: Converts the entire document while removing elements that sound awkward when read aloud
  - Detailed Summary: Creates a concise but comprehensive summary optimized for listening

## Requirements

- Python
- Google API key for Gemini AI (set in `.env` file)
- Required Python packages (see `requirements.txt`):
  - fastapi
  - uvicorn
  - PyPDF2
  - python-docx
  - google-generativeai
  - torch
  - soundfile
  - kokoro
  - requests
  - ffmpeg-python

## Setup

1. Clone the repository
2. Create a Python virtual environment:
   ```
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Create a `.env` file with your Google API key:
   ```
   GOOGLE_API_KEY=your_api_key_here
   ```

## Running the Application

Run the application using uvicorn:

```
python app.py
```

The application will be available at http://localhost:6123

Alternatively, you can use VS Code's Run and Debug feature to start the app.

## How It Works

1. Users upload a PDF or Word document
2. The app extracts text from the document
3. Gemini AI processes the text (either sanitizing for full-text mode or summarizing)
4. Kokoro TTS converts the processed text to natural-sounding speech
5. The user receives an MP3 file (or ZIP containing MP3 and transcript for summary mode)

## Privacy

Files are automatically deleted from the server after download to protect user privacy.

## License

This project is licensed under the GNU General Public License v3.0 - see the LICENSE file for details.
