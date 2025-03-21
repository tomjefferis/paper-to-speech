from flask import Flask, render_template, request, send_file, flash, jsonify, after_this_request, session
import os
import json
from werkzeug.utils import secure_filename
import atexit
from file_utils import change_file_extension, delayed_cleanup, clean_upload_folder, process_document, text_to_speech_apple

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
        # Check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return jsonify({'error': 'No file part'}), 400
        
        file = request.files['file']
        
        # If user does not select file, browser also
        # submits an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return jsonify({'error': 'No selected file'}), 400
        
        if file and allowed_file(file.filename):
            # Save the uploaded file
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            # Process the document to extract chapters
            chapters = process_document(file_path)
            
            if not chapters:
                return jsonify({'error': 'Failed to extract text from document'}), 500
            
            # Extract the sanitized text (first chapter's content)
            sanitized_text = chapters[0][1]
            
            # Get base filename for output
            file_basename = os.path.splitext(filename)[0]
            
            # Convert text to speech
            mp3_path = text_to_speech_apple(sanitized_text, app.config['UPLOAD_FOLDER'], file_basename)
            
            if not mp3_path or not os.path.exists(mp3_path):
                return jsonify({'error': 'Failed to convert text to speech'}), 500
            
            new_filename = os.path.basename(mp3_path)
            
            # Register function to clean up files after download
            @after_this_request
            def cleanup_after_request(response):
                delayed_cleanup([file_path, mp3_path])
                return response
            
            # Return the MP3 file
            return send_file(
                mp3_path,
                as_attachment=True,
                download_name=new_filename
            )
        else:
            flash('File type not allowed')
            return jsonify({'error': 'File type not allowed'}), 400
    
    return render_template("home.html")

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
    
    # Convert text to speech using Apple's say command and lame for MP3 conversion
    mp3_path = text_to_speech_apple(combined_text, app.config['UPLOAD_FOLDER'], file_basename)
    
    if not mp3_path or not os.path.exists(mp3_path):
        return jsonify({'error': 'Failed to convert text to speech'}), 500
    
    new_filename = os.path.basename(mp3_path)
    
    @after_this_request
    def cleanup_after_request(response):
        # Clean up the original document and keep the MP3 file for a short time to ensure download
        delayed_cleanup([original_path, mp3_path])
        return response
    
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

if __name__ == '__main__':
    app.run(debug=True)
