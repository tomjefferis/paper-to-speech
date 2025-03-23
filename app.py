from flask import Flask, render_template, request, send_file, flash, jsonify, after_this_request, session
import os
import json
import time  # Add the missing time import
from werkzeug.utils import secure_filename
import atexit
import file_utils  # Fix the missing import
from file_utils import change_file_extension, delayed_cleanup, clean_upload_folder, process_document, text_to_speech_kokoro, text_to_speech

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'txt', 'pdf', 'docx', 'doc'}
app.secret_key = 'your_secret_key'  # needed for flashing messages and session

# Create uploads folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Clean upload folder on startup
clean_upload_folder(app.config['UPLOAD_FOLDER'])

# Register cleanup function to run on app exit
atexit.register(lambda: clean_upload_folder(app.config['UPLOAD_FOLDER']))

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route("/", methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        # Check if file was uploaded
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
        
        file = request.files['file']
        
        # Check if file is empty
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Get TTS engine and processing mode from form
        tts_engine = request.form.get('tts_engine', 'kokoro')
        processing_mode = request.form.get('processing_mode', 'full')
        sanitizer = request.form.get('sanitizer', 'gemini')
        
        # Log the selected options
        app.logger.info(f"TTS Engine: {tts_engine}")
        app.logger.info(f"Processing Mode: {processing_mode}")
        app.logger.info(f"Sanitizer: {sanitizer}")
        
        # Create upload folder if it doesn't exist
        if not os.path.exists(app.config['UPLOAD_FOLDER']):
            os.makedirs(app.config['UPLOAD_FOLDER'])
        
        # Clean previous uploads
        clean_upload_folder(app.config['UPLOAD_FOLDER'])
        
        # Save the uploaded file and get its path
        original_path, _, _ = change_file_extension(file, app.config['UPLOAD_FOLDER'], 'original')
        
        # Process the document and get sections
        sections = process_document(original_path, sanitizer=sanitizer, processing_mode=processing_mode)
        
        # If processing failed, return error
        if not sections:
            return jsonify({'error': 'Failed to process document'}), 500
        
        # Use only the content from the first section
        _, content = sections[0]
        
        # Convert text to speech using selected engine
        audio_file = text_to_speech(content, app.config['UPLOAD_FOLDER'], secure_filename(f"output_{int(time.time())}"), tts_engine)
        
        if not audio_file:
            return jsonify({'error': 'Failed to convert text to speech'}), 500
            
        # Set up response for file download
        response = send_file(
            audio_file,
            as_attachment=True,
            download_name=os.path.basename(audio_file),
            mimetype='audio/mpeg'
        )
        
        # Schedule cleanup of files after download
        delayed_cleanup([original_path, audio_file])
        
        return response
        
    # GET request: show the upload form
    return render_template('home.html')

@app.route("/convert", methods=['POST'])
def convert_to_speech():
    """Convert selected chapters to speech"""
    if 'document_chapters' not in session:
        return jsonify({'error': 'No document has been uploaded'}), 400
    
    # Get selected chapters from request
    data = request.get_json()
    if not data or 'selected_chapters' not in data:
        return jsonify({'error': 'No chapters selected'}), 400
    
    selected_indices = data['selected_chapters']
    chapters = json.loads(session['document_chapters'])
    
    # Filter chapters based on selected indices
    selected_chapters = [chapters[i] for i in selected_indices if i < len(chapters)]

    
    # Combine all selected chapter texts
    # Check if chapters are in dict format or tuple format
    if isinstance(selected_chapters[0], dict):
        combined_text = "\n\n".join([chapter["content"] for chapter in selected_chapters])
    else:
        # Handle tuple format (title, content)
        combined_text = "\n\n".join([chapter[1] for chapter in selected_chapters])
    
    # Get original file path and create a base name for the output file
    original_path = session['original_file']
    file_basename = os.path.splitext(os.path.basename(original_path))[0]
    
    print(combined_text)
    
    # Convert text to speech using Kokoro TTS
    mp3_path = text_to_speech_kokoro(combined_text, app.config['UPLOAD_FOLDER'], file_basename)
    
    if not mp3_path or not os.path.exists(mp3_path):
        return jsonify({'error': 'Failed to convert text to speech'}), 500
    
    new_filename = os.path.basename(mp3_path)
    
    # @after_this_request
    # def cleanup_after_request(response):
    #     # Clean up the original document and keep the MP3 file for a short time to ensure download
    #     delayed_cleanup([original_path, mp3_path])
    #     return response
    
    return send_file(
        mp3_path,
        as_attachment=True,
        download_name=new_filename
    )

@app.route("/about")
def about():
    return render_template("about.html")

# Add a cleanup endpoint that can be called to manually clean the upload folder
@app.route("/cleanup")
def cleanup():
    clean_upload_folder(app.config['UPLOAD_FOLDER'])
    session.clear()
    return jsonify({"status": "success", "message": "Upload folder cleaned"})

# Add or update the route for TTS engines
@app.route('/tts-engines', methods=['GET'])
def get_tts_engines():
    try:
        engines = file_utils.get_available_tts_engines()
        print(f"Available TTS engines: {engines}")  # Debug log
        return jsonify({"engines": engines})
    except Exception as e:
        print(f"Error getting TTS engines: {str(e)}")
        # Always return a valid response even if there's an error
        return jsonify({"engines": ["kokoro"], "error": str(e)})

# Add route for text sanitizers
@app.route('/sanitizers', methods=['GET'])
def get_sanitizers():
    try:
        sanitizers = file_utils.get_available_sanitizers()
        print(f"Available text sanitizers: {sanitizers}")  # Debug log
        return jsonify({"sanitizers": sanitizers})
    except Exception as e:
        print(f"Error getting text sanitizers: {str(e)}")
        # Always return a valid response even if there's an error
        return jsonify({"sanitizers": ["gemini"], "error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
