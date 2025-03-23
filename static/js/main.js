document.addEventListener('DOMContentLoaded', function() {
    // Load available TTS engines and sanitizers
    fetchTTSEngines();
    fetchSanitizers();
    
    // Set up form submission handler
    const uploadForm = document.getElementById('uploadForm');
    if (uploadForm) {
        uploadForm.addEventListener('submit', handleFormSubmit);
    }
});

// Function to fetch available TTS engines
function fetchTTSEngines() {
    console.log('Fetching TTS engines...');
    const engineOptions = document.getElementById('ttsEngineOptions');
    
    fetch('/tts-engines')
        .then(response => {
            console.log('Response status:', response.status);
            // Always parse the response, even if it's an error
            return response.json();
        })
        .then(data => {
            console.log('TTS engines data:', data);
            // Clear loading message
            engineOptions.innerHTML = '';
            
            // Add available engines as radio buttons
            if (data.engines && data.engines.length > 0) {
                data.engines.forEach((engine, index) => {
                    const radioDiv = document.createElement('div');
                    radioDiv.className = 'form-check mb-2';
                    
                    const radioInput = document.createElement('input');
                    radioInput.type = 'radio';
                    radioInput.className = 'form-check-input';
                    radioInput.name = 'tts_engine';
                    radioInput.id = `engine-${engine}`;
                    radioInput.value = engine;
                    radioInput.checked = index === 0; // Select first option by default
                    
                    const radioLabel = document.createElement('label');
                    radioLabel.className = 'form-check-label';
                    radioLabel.htmlFor = `engine-${engine}`;
                    
                    // Capitalize engine name and add description
                    const capitalizedEngine = engine.charAt(0).toUpperCase() + engine.slice(1);
                    let description = '';
                    
                    switch(engine) {
                        case 'kokoro':
                            description = 'Natural-sounding TTS with multi-voice support';
                            break;
                        case 'coqui':
                            description = 'XTTSv2 voice cloning using davidA voice sample';
                            break;
                        case 'apple':
                            description = 'MacOS built-in TTS (fast but less natural)';
                            break;
                        case 'ollama':
                            description = 'Local AI-powered TTS using Gemma3:12b model';
                            break;
                        default:
                            description = 'Text-to-Speech engine';
                    }
                    
                    radioLabel.innerHTML = `<strong>${capitalizedEngine}</strong> - ${description}`;
                    
                    radioDiv.appendChild(radioInput);
                    radioDiv.appendChild(radioLabel);
                    engineOptions.appendChild(radioDiv);
                });
            } else {
                // If no engines available, show a message
                console.warn('No TTS engines found in data');
                engineOptions.innerHTML = '<div class="alert alert-warning">No TTS engines available. Using default.</div>' +
                                          '<input type="hidden" name="tts_engine" value="kokoro">';
            }
        })
        .catch(error => {
            console.error('Error fetching TTS engines:', error);
            // Display error and set default
            engineOptions.innerHTML = '<div class="alert alert-danger">Failed to load TTS engines. Using default.</div>' +
                                      '<input type="hidden" name="tts_engine" value="kokoro">';
        });
}

// Function to fetch available text sanitizers
function fetchSanitizers() {
    console.log('Fetching text sanitizers...');
    const sanitizerOptions = document.getElementById('sanitizerOptions');
    
    fetch('/sanitizers')
        .then(response => {
            console.log('Sanitizers response status:', response.status);
            // Always parse the response, even if it's an error
            return response.json();
        })
        .then(data => {
            console.log('Text sanitizers data:', data);
            // Clear loading message
            sanitizerOptions.innerHTML = '';
            
            // Add available sanitizers as radio buttons
            if (data.sanitizers && data.sanitizers.length > 0) {
                data.sanitizers.forEach((sanitizer, index) => {
                    const radioDiv = document.createElement('div');
                    radioDiv.className = 'form-check mb-2';
                    
                    const radioInput = document.createElement('input');
                    radioInput.type = 'radio';
                    radioInput.className = 'form-check-input';
                    radioInput.name = 'sanitizer';
                    radioInput.id = `sanitizer-${sanitizer}`;
                    radioInput.value = sanitizer;
                    radioInput.checked = index === 0; // Select first option by default
                    
                    const radioLabel = document.createElement('label');
                    radioLabel.className = 'form-check-label';
                    radioLabel.htmlFor = `sanitizer-${sanitizer}`;
                    
                    // Capitalize sanitizer name and add description
                    const capitalizedSanitizer = sanitizer.charAt(0).toUpperCase() + sanitizer.slice(1);
                    let description = '';
                    
                    switch(sanitizer) {
                        case 'gemini':
                            description = 'Google\'s Gemini API for text sanitization (requires API key)';
                            break;
                        case 'ollama':
                            description = 'Local AI-powered text sanitization using Gemma3:12b model';
                            break;
                        default:
                            description = 'Text sanitization service';
                    }
                    
                    radioLabel.innerHTML = `<strong>${capitalizedSanitizer}</strong> - ${description}`;
                    
                    radioDiv.appendChild(radioInput);
                    radioDiv.appendChild(radioLabel);
                    sanitizerOptions.appendChild(radioDiv);
                });
            } else {
                // If no sanitizers available, show a message
                console.warn('No text sanitizers found in data');
                sanitizerOptions.innerHTML = '<div class="alert alert-warning">No text sanitizers available. Using default.</div>' +
                                           '<input type="hidden" name="sanitizer" value="gemini">';
            }
        })
        .catch(error => {
            console.error('Error fetching text sanitizers:', error);
            // Display error and set default
            sanitizerOptions.innerHTML = '<div class="alert alert-danger">Failed to load text sanitizers. Using default.</div>' +
                                       '<input type="hidden" name="sanitizer" value="gemini">';
        });
}

function handleFormSubmit(event) {
    event.preventDefault();
    
    const uploadBtn = document.getElementById('uploadBtn');
    const uploadSpinner = document.getElementById('uploadSpinner');
    
    // Show spinner and disable button
    uploadBtn.disabled = true;
    uploadSpinner.classList.remove('d-none');
    
    // Create FormData object
    const formData = new FormData(event.target);
    
    // Check if a TTS engine is selected
    if (!formData.get('tts_engine')) {
        // If no engine is selected, use a default
        formData.append('tts_engine', 'kokoro');
    }
    
    // Check if processing mode is selected
    if (!formData.get('processing_mode')) {
        // If no processing mode is selected, use full as default
        formData.append('processing_mode', 'full');
    }
    
    console.log("Processing mode:", formData.get('processing_mode'));
    
    // Submit form data to server
    fetch('/', {
        method: 'POST',
        body: formData
    })
    .then(response => {
        if (response.headers.get('Content-Type') && response.headers.get('Content-Type').includes('application/json')) {
            // Handle JSON response (error)
            return response.json().then(data => {
                if (!response.ok) {
                    throw new Error(data.error || 'Failed to process document');
                }
                return data;
            });
        } else {
            // Handle non-JSON response (file download)
            const contentDisposition = response.headers.get('Content-Disposition');
            let filename = 'speech.mp3';
            
            if (contentDisposition) {
                const filenameMatch = contentDisposition.match(/filename="?([^"]+)"?/);
                if (filenameMatch && filenameMatch[1]) {
                    filename = filenameMatch[1];
                }
            }
            
            // Trigger file download
            return response.blob().then(blob => {
                const url = window.URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.style.display = 'none';
                a.href = url;
                a.download = filename;
                document.body.appendChild(a);
                a.click();
                window.URL.revokeObjectURL(url);
                return { success: true, filename: filename };
            });
        }
    })
    .catch(error => {
        // Handle errors
        alert('Error: ' + error.message);
        console.error('Error:', error);
    })
    .finally(() => {
        // Hide spinner and enable button
        uploadBtn.disabled = false;
        uploadSpinner.classList.add('d-none');
    });
}