import os
import shutil
import time
import threading
import subprocess
from werkzeug.utils import secure_filename
import PyPDF2
from docx import Document
from google import genai
from config import get_google_api_key
import re

def change_file_extension(file, upload_folder, new_extension="mp3"):
    """
    Save uploaded file and change its extension.
    
    Args:
        file: The uploaded file object
        upload_folder: Path to the upload directory
        new_extension: The new extension to apply (default is mp3)
        
    Returns:
        tuple: (original_path, new_path, new_filename)
    """
    # Save the original file
    filename = secure_filename(file.filename)
    original_path = os.path.join(upload_folder, filename)
    file.save(original_path)
    
    # Create the new filename with the desired extension
    file_basename = os.path.splitext(filename)[0]
    new_filename = f"{file_basename}.{new_extension}"
    new_path = os.path.join(upload_folder, new_filename)
    
    # Copy the file with new extension
    # In a real app, here you would convert the file to the actual format
    shutil.copy2(original_path, new_path)
    
    return original_path, new_path, new_filename

def cleanup_files(files_to_remove):
    """
    Remove files from the filesystem.
    
    Args:
        files_to_remove: List of file paths to remove
    """
    for file_path in files_to_remove:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"Removed file: {file_path}")
            else:
                print(f"File not found: {file_path}")
        except Exception as e:
            print(f"Error removing file {file_path}: {e}")

def delayed_cleanup(files_to_remove, delay=2):
    """
    Clean up files after a delay to ensure the download is complete.
    
    Args:
        files_to_remove: List of file paths to remove
        delay: Delay in seconds before cleaning up
    """
    def _cleanup():
        time.sleep(delay)
        cleanup_files(files_to_remove)
    
    # Start a new thread for cleanup
    threading.Thread(target=_cleanup).start()

def clean_upload_folder(upload_folder):
    """
    Clean all files in the upload folder.
    
    Args:
        upload_folder: Path to the upload directory
    """
    try:
        for filename in os.listdir(upload_folder):
            file_path = os.path.join(upload_folder, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
                print(f"Removed file during folder cleanup: {file_path}")
    except Exception as e:
        print(f"Error cleaning upload folder: {e}")

def extract_text_from_pdf(file_path):
    """
    Extract text from a PDF file.
    
    Args:
        file_path: Path to the PDF file
        
    Returns:
        str: Extracted text from the PDF
    """
    text = ""
    try:
        with open(file_path, 'rb') as pdf_file:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text += page.extract_text() + "\n"
        return text
    except Exception as e:
        print(f"Error extracting text from PDF {file_path}: {e}")
        return ""

def extract_text_from_docx(file_path):
    """
    Extract text from a Word document.
    
    Args:
        file_path: Path to the Word document
        
    Returns:
        str: Extracted text from the Word document
    """
    text = ""
    try:
        doc = Document(file_path)
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text
    except Exception as e:
        print(f"Error extracting text from Word document {file_path}: {e}")
        return ""

def extract_text_from_document(file_path):
    """
    Extract text from a document based on its file extension.
    
    Args:
        file_path: Path to the document
        
    Returns:
        str: Sanitized text extracted from the document:
             - linebreaks replaced by spaces
             - non-alphanumeric characters removed (except basic punctuation)
             - converted to lowercase
    """
    _, file_extension = os.path.splitext(file_path)
    file_extension = file_extension.lower()
    
    if file_extension == '.pdf':
        text = extract_text_from_pdf(file_path)
    elif file_extension in ['.docx', '.doc']:
        text = extract_text_from_docx(file_path)
    else:
        print(f"Unsupported file type: {file_extension}")
        return ""
    
    
    # Step 1: Replace linebreaks with spaces
    sanitized_text = text.replace('\n', ' ')
    
    # Step 2: Keep only alphanumeric characters, spaces, and basic punctuation
    # Basic punctuation: period, comma, semicolon, colon, question mark, exclamation mark,
    # quotation marks, parentheses, dash, apostrophe
    sanitized_text = re.sub(r'[^\w\s.,;:?!"\'\(\)\-]', '', sanitized_text)
    
    # Step 3: Convert to lowercase
    sanitized_text = sanitized_text.lower()
    
    # Step 4: Remove multiple spaces
    sanitized_text = re.sub(r'\s+', ' ', sanitized_text)
    
    return sanitized_text.strip()

def count_tokens(text, model_name="gemini-2.0-flash"):
    """
    Count tokens in text using Google's Generative AI API.
    
    Args:
        text: The input text to count tokens for
        model_name: The model to use for token counting
        
    Returns:
        int: Accurate token count from the API
    """
    try:
        api_key = get_google_api_key()
        client = genai.Client(api_key=api_key, http_options=genai.types.HttpOptions(api_version="v1"))
        response = client.models.count_tokens(
            model=model_name,
            contents=text,
        )
        return response.total_tokens
    except Exception as e:
        print(f"Error counting tokens: {e}")
        # Fall back to estimation if the API call fails
        return len(text) // 4  # Simple approximation as fallback

def sanitize_text_for_speech(text, model_name="gemini-2.0-flash"):
    """
    Use Gemini AI to sanitize text for speech synthesis - adding proper punctuation,
    removing content that would sound strange when read aloud, and improving flow.
    
    Args:
        text: Input text to sanitize
        model_name: The model to use (default is gemini-2.0-flash)
        
    Returns:
        str: Text sanitized and optimized for speech synthesis
    """
    try:
        print("Using Gemini for text sanitization...")
        # Count tokens before processing using the API method
        token_count = count_tokens(text)
        print(f"Token count before processing: {token_count}")
        
        # For very large texts, you might want to add handling here
        # For example, splitting into chunks or truncating
        
        api_key = get_google_api_key()
        
        # Initialize the client with API key
        client = genai.Client(api_key=api_key)
        
        # Create a prompt that asks for text sanitization for speech synthesis
        # This is the original sanitization prompt for full text processing
        prompt = """
        Sanitize the following text for speech synthesis. Your task is to:
        
        1. Add proper punctuation where it's missing
        2. Remove elements that would sound awkward when read aloud (citations, figure references, equations, etc.)
        3. Fix any formatting issues that would cause unnatural pauses or readings
        4. Ensure proper sentence structure for natural-sounding speech
        5. Keep the core content and meaning intact
        6. Remove the list of references at the end of the document
        7. Ensure the text flows well and is easy to read aloud
        8. remove the list of authors and affiliations
        9. do not include line breaks or anything that would sound weird when read aloud in the final text, such as "slash n"
        10. make sure all acronyms are expanded to be read properly, for example ERP should be expanded to Event-Related Potential
        
        Return ONLY the cleaned, speech-ready text without any explanations or comments. 
        The speech-ready text should of a similar or the same length to the original text, whilst keeping all meaning intact.
        
        Here is the text to sanitize:
        
        """
        
        # Generate content using the specified model
        response = client.models.generate_content(
            model=model_name,
            contents=prompt + text
        )
        
        return response.text.strip()
            
    except Exception as e:
        print(f"Error sanitizing text for speech: {e}")
        return text  # Return original text if sanitization fails

def summarize_text_for_speech(text, model_name="gemini-2.0-flash"):
    """
    Use Gemini AI to create a detailed, structured summary of the text suitable for speech synthesis.
    
    Args:
        text: Input text to summarize
        model_name: The model to use (default is gemini-2.0-flash)
        
    Returns:
        str: A detailed summary structured with background, methods, results and conclusion
    """
    try:
        print("Creating detailed summary...")
        # Count tokens before processing
        token_count = count_tokens(text)
        print(f"Token count before processing: {token_count}")
        
        api_key = get_google_api_key()
        
        # Initialize the client with API key
        client = genai.Client(api_key=api_key)
        
        # Create a prompt specifically for detailed summarization with structured sections
        prompt = """
        Create a detailed summary of the following academic paper or document. Your summary should:
        
        1. Be well-structured and comprehensive while being more concise than the original
        2. Include clear sections covering:
           - Background and context
           - Methods and approach
           - Key results and findings
           - Conclusions and implications
        3. Maintain academic language but optimize for speech readability
        4. Expand acronyms on first mention
        5. Remove citations, figure references, and other elements that would sound awkward when read aloud
        6. Include proper punctuation and transitions for natural-sounding speech
        7. Preserve the most important numerical results and statistics
        8. Do not include formatting instructions or comments in the summary
        9. Do not include headings formatting such as '#' or '##'
        
        Format the summary in a way that flows naturally when read aloud, with clear section transitions.
        
        RETURN ONLY THE SUMMARY TEXT WITHOUT EXPLANATION OR COMMENTS.
        
        Here is the document to summarize:
        
        """
        
        # Generate content using the specified model
        response = client.models.generate_content(
            model=model_name,
            contents=prompt + text
        )
        
        return response.text.strip()
            
    except Exception as e:
        print(f"Error creating summary for speech: {e}")
        return text  # Return original text if summarization fails

def process_document(file_path, model_name="gemini-2.0-flash", sanitizer="gemini", processing_mode="full"):
    """
    Process a document end-to-end: extract text and prepare it for speech synthesis.
    
    Args:
        file_path: Path to the document file
        model_name: The Gemini model to use for text processing
        sanitizer: Parameter kept for backward compatibility
        processing_mode: Either "full" for direct synthesis or "summary" for detailed summarization
        
    Returns:
        list: A list containing a single tuple with ("Full Document", processed_text)
               formatted as (title, content) for consistency
    """
    # Step 1: Extract text from the document
    text = extract_text_from_document(file_path)
    if not text:
        print(f"Failed to extract text from document: {file_path}")
        return []
    
    # Count tokens using the API method
    token_count = count_tokens(text)
    print(f"Document token count: {token_count}")
    
    # Step 2: Process the text based on selected mode
    if processing_mode == "summary":
        print("Creating detailed summary for speech synthesis")
        processed_text = summarize_text_for_speech(text, model_name)
        title = "Document Summary"
    else:
        print("Sanitizing full text for speech synthesis")
        processed_text = sanitize_text_for_speech(text, model_name)
        title = "Full Document"
    
    # Return the processed document as a single section
    # Using tuple format (title, content) for consistency
    return [(title, processed_text)]

def text_to_speech_apple(text, output_folder, file_basename):
    """
    Convert text to speech using Apple's 'say' command and convert to MP3 using 'lame'.
    
    Args:
        text: The text to convert to speech
        output_folder: The folder to save the audio files
        file_basename: The base filename to use for output files
        
    Returns:
        str: Path to the created MP3 file, or None if conversion failed
    """
    try:
        # Create temporary text file
        temp_text_file = os.path.join(output_folder, f"{file_basename}_text.txt")
        with open(temp_text_file, 'w') as f:
            f.write(text)
        
        # Create paths for temporary AIFF file and final MP3 file
        temp_aiff_file = os.path.join(output_folder, f"{file_basename}.aiff")
        output_mp3_file = os.path.join(output_folder, f"{file_basename}.mp3")
        
        # Use Apple's 'say' command to generate speech and save as AIFF
        say_cmd = ['say', '-f', temp_text_file, '-o', temp_aiff_file]
        subprocess.run(say_cmd, check=True)
        
        # Convert AIFF to MP3 using lame
        lame_cmd = ['lame', '-m', 'm', temp_aiff_file, output_mp3_file]
        subprocess.run(lame_cmd, check=True)
        
        # Clean up temporary files
        cleanup_files([temp_text_file, temp_aiff_file])
        
        # Return path to the created MP3 file
        return output_mp3_file
    
    except Exception as e:
        print(f"Error converting text to speech: {e}")
        return None

def text_to_speech_kokoro(text, output_folder, file_basename):
    """
    Convert text to speech using Kokoro TTS with sentence-level chunking.
    
    Args:
        text: The text to convert to speech
        output_folder: The folder to save the audio files
        file_basename: The base filename to use for output files
        
    Returns:
        str: Path to the created MP3 file, or None if conversion failed
    """
    try:
        import torch
        import soundfile as sf
        from kokoro import KPipeline
        import subprocess
        import re
        
        # Create temporary WAV file path and final MP3 file path
        temp_wav_file = os.path.join(output_folder, f"{file_basename}.wav")
        output_mp3_file = os.path.join(output_folder, f"{file_basename}.mp3")
        
        # Initialize Kokoro pipeline
        pipeline = KPipeline(lang_code='b')  # 'a' for American English
        
        # Split text into sentences for better processing
        # Use regex to find sentences ending with period followed by space or end of string
        sentences = re.split(r'(?<=\. )|\n', text)
        # Filter out empty sentences
        sentences = [sentence.strip() for sentence in sentences if sentence.strip()]
        
        print(f"Split text into {len(sentences)} sentences for Kokoro TTS")
        
        # If we have no sentences after splitting, treat the whole text as one sentence
        if not sentences:
            sentences = [text]
        
        # Process each sentence and collect the audio
        full_audio = []
        
        for i, sentence in enumerate(sentences):
            if not sentence:
                continue
                
            print(f"Processing sentence {i+1}/{len(sentences)} with Kokoro")
            
            # Generate speech for this sentence
            generator = pipeline(sentence, voice='af_heart', speed=1)
            
            # Collect audio segments for this sentence
            for j, (gs, ps, audio) in enumerate(generator):
                full_audio.append(audio)
        
        # If there's audio, save it
        if full_audio:
            combined_audio = torch.cat(full_audio, dim=0).numpy()
            sf.write(temp_wav_file, combined_audio, 24000)
            
            # Convert WAV to MP3 using lame
            lame_cmd = ['lame', '-m', 'm', temp_wav_file, output_mp3_file]
            subprocess.run(lame_cmd, check=True)
            
            # Clean up temporary WAV file
            cleanup_files([temp_wav_file])
            
            print(f"Successfully created MP3 using Kokoro TTS with sentence splitting: {output_mp3_file}")
            return output_mp3_file
        else:
            print("No audio was generated by Kokoro TTS")
            return None
    
    except Exception as e:
        print(f"Error converting text to speech with Kokoro TTS: {e}")
        import traceback
        traceback.print_exc()  # Print the full stack trace for debugging
        return None

def text_to_speech(text, output_folder, file_basename, engine="kokoro"):
    """
    Convert text to speech using the specified TTS engine.
    
    Args:
        text: The text to convert to speech
        output_folder: The folder to save the audio files
        file_basename: The base filename to use for output files
        engine: The TTS engine to use ('kokoro' or 'apple')
        
    Returns:
        str: Path to the created MP3 file, or None if conversion failed
    """
    # Select the appropriate TTS engine
    if engine.lower() == 'apple':
        return text_to_speech_apple(text, output_folder, file_basename)
    else:
        # Default to Kokoro for any other value
        return text_to_speech_kokoro(text, output_folder, file_basename)

def get_available_tts_engines():
    """
    Returns a list of available TTS engines based on installed dependencies.
    
    Returns:
        list: List of available TTS engine names
    """
    available_engines = ['kokoro']  # Kokoro is always available
    
    # Check for Apple 'say' command (macOS only)
    try:
        print("Checking for Apple 'say' command...")
        result = subprocess.run(['which', 'say'], capture_output=True)
        if result.returncode == 0:
            available_engines.append('apple')
            print("Apple 'say' command found")
        else:
            print("Apple 'say' command not found")
    except Exception as e:
        print(f"Apple 'say' command not available: {str(e)}")
    
    print(f"Final available engines: {available_engines}")
    return available_engines

def get_available_sanitizers():
    """
    Returns a list of available text sanitization services.
    
    Returns:
        list: List of available text sanitization service names
    """
    # Only Gemini is available now
    return ["gemini"]



