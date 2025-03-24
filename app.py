import os
import json
import time
from typing import Optional, List
import zipfile
import uvicorn
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request
from werkzeug.utils import secure_filename
import atexit
import file_utils
from file_utils import change_file_extension, delayed_cleanup, clean_upload_folder, process_document, text_to_speech_kokoro, text_to_speech

app = FastAPI(title="Paper to Speech API")

# Configure CORS to allow LAN connections
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'docx', 'doc'}

# Create uploads folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Clean upload folder on startup
clean_upload_folder(UPLOAD_FOLDER)

# Register cleanup function to run on app exit
atexit.register(lambda: clean_upload_folder(UPLOAD_FOLDER))

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Templates
templates = Jinja2Templates(directory="templates")

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/")
async def convert_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    tts_engine: str = Form("kokoro"),
    processing_mode: str = Form("full"),
    sanitizer: str = Form("gemini")
):
    # Check if file is empty
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file selected")
    
    # Log the selected options
    print(f"TTS Engine: {tts_engine}")
    print(f"Processing Mode: {processing_mode}")
    print(f"Sanitizer: {sanitizer}")
    
    # Save the uploaded file
    filename = secure_filename(file.filename)
    file_content = await file.read()
    
    # Create a temporary file
    original_path = os.path.join(UPLOAD_FOLDER, filename)
    with open(original_path, "wb") as f:
        f.write(file_content)
    
    # Process the document and get sections
    sections = process_document(original_path, sanitizer=sanitizer, processing_mode=processing_mode)
    
    # If processing failed, return error
    if not sections:
        raise HTTPException(status_code=500, detail="Failed to process document")
    
    # Use only the content from the first section
    _, content = sections[0]
    
    # Convert text to speech using selected engine
    audio_file = text_to_speech(content, UPLOAD_FOLDER, secure_filename(f"output_{int(time.time())}"), tts_engine)
    
    if not audio_file:
        raise HTTPException(status_code=500, detail="Failed to convert text to speech")
    
    # Handle summary mode with transcript
    if processing_mode == "summary":
        # Create a text file with the transcript
        timestamp = int(time.time())
        transcript_filename = secure_filename(f"transcript_{timestamp}.txt")
        transcript_path = os.path.join(UPLOAD_FOLDER, transcript_filename)
        
        with open(transcript_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        # Create a zip file containing both the audio and transcript
        zip_filename = secure_filename(f"output_with_transcript_{timestamp}.zip")
        zip_path = os.path.join(UPLOAD_FOLDER, zip_filename)
        
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            zipf.write(audio_file, os.path.basename(audio_file))
            zipf.write(transcript_path, os.path.basename(transcript_path))
        
        # Schedule cleanup
        background_tasks.add_task(delayed_cleanup, [original_path, audio_file, transcript_path, zip_path])
        
        return FileResponse(
            path=zip_path, 
            filename=os.path.basename(zip_path),
            media_type="application/zip"
        )
    else:
        # Schedule cleanup
        background_tasks.add_task(delayed_cleanup, [original_path, audio_file])
        
        return FileResponse(
            path=audio_file, 
            filename=os.path.basename(audio_file),
            media_type="audio/mpeg"
        )

@app.get("/about")
async def about(request: Request):
    return templates.TemplateResponse("about.html", {"request": request})

@app.get("/cleanup")
async def cleanup():
    clean_upload_folder(UPLOAD_FOLDER)
    return {"status": "success", "message": "Upload folder cleaned"}

@app.get('/tts-engines')
async def get_tts_engines():
    try:
        engines = file_utils.get_available_tts_engines()
        print(f"Available TTS engines: {engines}")
        return {"engines": engines}
    except Exception as e:
        print(f"Error getting TTS engines: {str(e)}")
        return {"engines": ["kokoro"], "error": str(e)}

@app.get('/sanitizers')
async def get_sanitizers():
    try:
        sanitizers = file_utils.get_available_sanitizers()
        print(f"Available text sanitizers: {sanitizers}")
        return {"sanitizers": sanitizers}
    except Exception as e:
        print(f"Error getting text sanitizers: {str(e)}")
        return {"sanitizers": ["gemini"], "error": str(e)}

if __name__ == "__main__":
    # Run with uvicorn, binding to all interfaces (0.0.0.0) to allow LAN access
    uvicorn.run("app:app", host="0.0.0.0", port=6123, reload=True)
