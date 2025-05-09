// Add progress bar functionality for uploads and background processing indicators
document.addEventListener('DOMContentLoaded', function() {
    // --- Upload form handling ---
    const uploadForm = document.getElementById('upload-form');
    const imageInput = document.getElementById('image-upload');
    const imagePreview = document.getElementById('image-preview');
    const previewContainer = document.getElementById('preview-container');

    // Handle image preview
    if (imageInput) {
        imageInput.addEventListener('change', function() {
            if (this.files && this.files[0]) {
                const reader = new FileReader();
                
                reader.onload = function(e) {
                    imagePreview.src = e.target.result;
                    previewContainer.classList.remove('d-none');
                };
                
                reader.readAsDataURL(this.files[0]);
            }
        });
    }

    // Create processing overlay
    function createProcessingOverlay() {
        const overlay = document.createElement('div');
        overlay.className = 'processing-overlay';
        overlay.innerHTML = `
            <div class="processing-content">
                <div class="spinner-border text-light" role="status">
                    <span class="visually-hidden">Processing...</span>
                </div>
                <div class="processing-text mt-3">
                    <h5>AI Model Working</h5>
                    <p class="text-light">Analyzing license plate with advanced CNN model...</p>
                    <div class="progress mb-2">
                        <div class="progress-bar progress-bar-striped progress-bar-animated" 
                             role="progressbar" style="width: 0%"></div>
                    </div>
                    <div class="processing-steps">
                        <div class="step">Detection</div>
                        <div class="step">Preprocessing</div>
                        <div class="step">Recognition</div>
                        <div class="step">Verification</div>
                    </div>
                </div>
            </div>
        `;
        
        document.body.appendChild(overlay);
        return overlay;
    }

    // Animate the processing steps
    function simulateProcessingSteps(overlay) {
        const progressBar = overlay.querySelector('.progress-bar');
        const steps = overlay.querySelectorAll('.step');
        
        // Reset
        steps.forEach(step => step.classList.remove('active', 'completed'));
        
        // Animate each step
        const stepDuration = 750; // milliseconds per step
        let currentProgress = 0;
        
        // Step 1: Detection
        steps[0].classList.add('active');
        progressBar.style.width = '25%';
        
        setTimeout(() => {
            steps[0].classList.remove('active');
            steps[0].classList.add('completed');
            steps[1].classList.add('active');
            progressBar.style.width = '50%';
            
            // Step 2: Preprocessing
            setTimeout(() => {
                steps[1].classList.remove('active');
                steps[1].classList.add('completed');
                steps[2].classList.add('active');
                progressBar.style.width = '75%';
                
                // Step 3: Recognition
                setTimeout(() => {
                    steps[2].classList.remove('active');
                    steps[2].classList.add('completed');
                    steps[3].classList.add('active');
                    progressBar.style.width = '90%';
                    
                    // Step 4: Verification
                    setTimeout(() => {
                        steps[3].classList.remove('active');
                        steps[3].classList.add('completed');
                        progressBar.style.width = '100%';
                    }, stepDuration);
                }, stepDuration);
            }, stepDuration);
        }, stepDuration);
    }

    // Handle form submission
    if (uploadForm) {
        uploadForm.addEventListener('submit', function(e) {
            e.preventDefault();
            
            const formData = new FormData(this);
            const xhr = new XMLHttpRequest();
            
            // Get loading indicator
            const loadingIndicator = document.getElementById('loading-indicator');
            if (loadingIndicator) {
                loadingIndicator.classList.remove('d-none');
            }
            
            // Disable submit button
            const submitButton = uploadForm.querySelector('button[type="submit"]');
            if (submitButton) {
                submitButton.disabled = true;
                submitButton.textContent = 'Processing...';
            }
            
            // Create and show the processing overlay
            const overlay = createProcessingOverlay();
            simulateProcessingSteps(overlay);
            
            xhr.open('POST', this.action, true);
            
            // Set headers to indicate this is an AJAX request
            xhr.setRequestHeader('X-Requested-With', 'XMLHttpRequest');
            
            xhr.onload = function() {
                // Remove the overlay
                setTimeout(() => {
                    document.body.removeChild(overlay);
                    
                    // Re-enable submit button and hide loading indicator
                    if (submitButton) {
                        submitButton.disabled = false;
                        submitButton.innerHTML = '<span class="spinner-border spinner-border-sm d-none me-2" role="status"></span>Upload & Process';
                    }
                    if (loadingIndicator) {
                        loadingIndicator.classList.add('d-none');
                    }
                    
                    if (xhr.status >= 200 && xhr.status < 300) {
                        window.location.href = '/';
                    } else {
                        console.error('Upload failed with status:', xhr.status);
                        
                        // Try to parse error message if available
                        let errorMessage = 'Upload failed. Please try again with a smaller image.';
                        try {
                            if (xhr.responseText) {
                                const response = JSON.parse(xhr.responseText);
                                if (response.error) {
                                    errorMessage = response.error;
                                }
                            }
                        } catch (e) {
                            console.error('Error parsing response:', e);
                        }
                        
                        alert(errorMessage);
                    }
                }, 500); // Short delay to show 100% progress
            };
            
            xhr.onerror = function() {
                // Remove the overlay
                document.body.removeChild(overlay);
                
                // Re-enable submit button and hide loading indicator
                if (submitButton) {
                    submitButton.disabled = false;
                    submitButton.innerHTML = '<span class="spinner-border spinner-border-sm d-none me-2" role="status"></span>Upload & Process';
                }
                if (loadingIndicator) {
                    loadingIndicator.classList.add('d-none');
                }
                
                console.error('Request failed');
                alert('Upload failed. Please try again.');
            };
            
            xhr.send(formData);
        });
    }

    // --- Video page handling ---
    const videoPage = document.querySelector('.video-page');
    if (videoPage) {
        // Add processing indicators for the video analysis
        const detectButton = document.getElementById('detect-button');
        const saveButton = document.getElementById('save-button');
        
        if (detectButton) {
            detectButton.addEventListener('click', function() {
                const processingStatus = document.getElementById('processing-status');
                if (processingStatus) {
                    processingStatus.classList.remove('d-none');
                    processingStatus.innerHTML = `
                        <div class="alert alert-info d-flex align-items-center" role="alert">
                            <div class="spinner-border spinner-border-sm me-2" role="status"></div>
                            <div>
                                Advanced ANPR model processing image...
                            </div>
                        </div>
                    `;
                    
                    // The status will be updated by the detect function in the video page
                }
            });
        }
    }

    // --- Clear search functionality ---
    const clearSearchBtn = document.getElementById('clear-search');
    if (clearSearchBtn) {
        clearSearchBtn.addEventListener('click', function() {
            window.location.href = '/';
        });
    }
});