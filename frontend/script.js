// LinkedIn Photo Optimizer Frontend (Step-by-Step)
class LinkedInOptimizer {
    constructor() {
        this.referenceImage = null;
        this.uploadedImages = [];
        this.topImages = [];
        this.finalImages = [];
        this.sessionId = null;
        this.currentStep = 0;
        
        this.initializeEventListeners();
    }

    initializeEventListeners() {
        // Reference image upload
        const referenceBtn = document.getElementById('reference-btn');
        const referenceInput = document.getElementById('reference-input');
        const referenceUpload = document.getElementById('reference-upload');

        referenceBtn.addEventListener('click', () => referenceInput.click());
        referenceUpload.addEventListener('click', () => referenceInput.click());
        referenceInput.addEventListener('change', (e) => this.handleReferenceUpload(e));

        // Drag and drop for reference
        referenceUpload.addEventListener('dragover', (e) => this.handleDragOver(e));
        referenceUpload.addEventListener('drop', (e) => this.handleReferenceDrop(e));

        // Batch images upload
        const uploadBtn = document.getElementById('upload-btn');
        const fileInput = document.getElementById('file-input');
        const uploadZone = document.getElementById('upload-zone');

        uploadBtn.addEventListener('click', () => fileInput.click());
        uploadZone.addEventListener('click', () => fileInput.click());
        fileInput.addEventListener('change', (e) => this.handleBatchUpload(e));

        // Drag and drop for batch images
        uploadZone.addEventListener('dragover', (e) => this.handleDragOver(e));
        uploadZone.addEventListener('drop', (e) => this.handleBatchDrop(e));

        // Download and restart buttons
        const downloadBtn = document.getElementById('download-btn');
        const restartBtn = document.getElementById('restart-btn');
        
        downloadBtn.addEventListener('click', () => this.downloadAllImages());
        restartBtn.addEventListener('click', () => this.restart());
    }

    // Reference Image Handling
    async handleReferenceUpload(event) {
        const file = event.target.files[0];
        if (file) {
            await this.processReferenceImage(file);
        }
    }

    async handleReferenceDrop(event) {
        event.preventDefault();
        const file = event.dataTransfer.files[0];
        if (file && file.type.startsWith('image/')) {
            await this.processReferenceImage(file);
        }
    }

    async processReferenceImage(file) {
        // Show preview
        const preview = document.getElementById('reference-preview');
        const image = document.getElementById('reference-image');
        const status = document.getElementById('reference-status');
        
        image.src = URL.createObjectURL(file);
        preview.classList.remove('hidden');
        
        // Show validating status
        status.className = 'reference-status validating';
        status.innerHTML = '<div class="status-spinner"></div><span>Validating reference image...</span>';

        try {
            // Simulate API call to validate reference image
            const isValid = await this.validateReferenceImage(file);
            
            if (isValid) {
                this.referenceImage = file;
                status.className = 'reference-status valid';
                status.innerHTML = '<span>✅ Great reference image!</span>';
                
                // Show upload section after 1 second
                setTimeout(() => {
                    document.getElementById('upload-section').classList.remove('hidden');
                    document.getElementById('upload-section').scrollIntoView({ behavior: 'smooth' });
                }, 1000);
            } else {
                status.className = 'reference-status invalid';
                status.innerHTML = `
                    <span>❌ Reference image not suitable</span>
                    <button class="retry-btn" onclick="this.parentElement.parentElement.classList.add('hidden')">Try Another</button>
                `;
            }
        } catch (error) {
            status.className = 'reference-status invalid';
            status.innerHTML = `
                <span>❌ Error validating image</span>
                <button class="retry-btn" onclick="this.parentElement.parentElement.classList.add('hidden')">Try Another</button>
            `;
        }
    }

    async validateReferenceImage(file) {
        const formData = new FormData();
        formData.append('reference_image', file);

        try {
            const response = await fetch('http://localhost:5002/api/validate-reference', {
                method: 'POST',
                body: formData
            });
            
            const result = await response.json();
            
            if (response.ok && result.valid) {
                return true;
            } else {
                console.log('Validation failed:', result.message);
                return false;
            }
        } catch (error) {
            console.error('Reference validation error:', error);
            return false;
        }
    }

    // Batch Images Handling
    handleDragOver(event) {
        event.preventDefault();
        event.currentTarget.classList.add('dragover');
    }

    async handleBatchUpload(event) {
        const files = Array.from(event.target.files);
        await this.processBatchImages(files);
    }

    async handleBatchDrop(event) {
        event.preventDefault();
        event.currentTarget.classList.remove('dragover');
        
        const files = Array.from(event.dataTransfer.files).filter(file => 
            file.type.startsWith('image/')
        );
        await this.processBatchImages(files);
    }

    async processBatchImages(files) {
        if (files.length === 0) return;
        
        this.uploadedImages = files;
        
        // Hide upload section and show processing
        document.getElementById('upload-section').classList.add('hidden');
        document.getElementById('processing-section').classList.remove('hidden');
        document.getElementById('processing-section').scrollIntoView({ behavior: 'smooth' });
        
        // Start processing pipeline
        await this.runProcessingPipeline();
    }

    async runProcessingPipeline() {
        const steps = [
            { id: 'step-1', title: '📷 Loading images...', api: 'load-images' },
            { id: 'step-2', title: '🎯 Detecting faces...', api: 'detect-and-crop' },
            { id: 'step-3', title: '🎨 Replacing backgrounds...', api: 'replace-backgrounds' },
            { id: 'step-4', title: '⭐ Scoring quality...', api: 'score-and-filter' },
            { id: 'step-5', title: '🏆 Selecting top 5...', api: null }
        ];

        const progressFill = document.getElementById('progress-fill');
        const progressText = document.getElementById('progress-text');
        const processingTitle = document.getElementById('processing-title');

        try {
            for (let i = 0; i < steps.length; i++) {
                const step = steps[i];
                
                // Update title
                processingTitle.textContent = step.title;
                
                // Update progress
                const progress = ((i + 1) / steps.length) * 100;
                progressFill.style.width = `${progress}%`;
                progressText.textContent = `Step ${i + 1} of ${steps.length}`;
                
                // Update step status
                const stepElement = document.getElementById(step.id);
                stepElement.classList.add('active');
                stepElement.querySelector('.step-status').textContent = '🔄';
                
                // Run actual pipeline step
                if (step.api) {
                    await this.runPipelineStep(step.api);
                }
                
                // Mark as completed
                stepElement.classList.remove('active');
                stepElement.classList.add('completed');
                stepElement.querySelector('.step-status').textContent = '✅';
            }

            // Pipeline completed - show results
            await this.showTopImages();
            
        } catch (error) {
            console.error('Pipeline error:', error);
            this.showProcessingError(error.message || 'Pipeline processing failed');
        }
    }

    async runPipelineStep(stepName) {
        let response;
        
        switch (stepName) {
            case 'load-images':
                response = await this.loadImages();
                break;
            case 'detect-and-crop':
                response = await this.detectAndCrop();
                break;
            case 'replace-backgrounds':
                response = await this.replaceBackgrounds();
                break;
            case 'score-and-filter':
                response = await this.scoreAndFilter();
                break;
        }
        
        if (!response.success) {
            throw new Error(response.message);
        }
        
        return response;
    }

    async loadImages() {
        const formData = new FormData();
        formData.append('reference_image', this.referenceImage);
        
        if (this.sessionId) {
            formData.append('session_id', this.sessionId);
        }
        
        this.uploadedImages.forEach((file, index) => {
            formData.append(`image_${index}`, file);
        });

        const response = await fetch('http://localhost:5002/api/load-images', {
            method: 'POST',
            body: formData
        });
        
        const result = await response.json();
        
        if (result.success) {
            this.sessionId = result.session_id;
            this.currentStep = 1;
        }
        
        return result;
    }

    async detectAndCrop() {
        const response = await fetch('http://localhost:5002/api/detect-and-crop', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ session_id: this.sessionId })
        });
        
        const result = await response.json();
        
        if (result.success) {
            this.currentStep = 2;
        }
        
        return result;
    }

    async replaceBackgrounds() {
        const response = await fetch('http://localhost:5002/api/replace-backgrounds', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ session_id: this.sessionId })
        });
        
        const result = await response.json();
        
        if (result.success) {
            this.currentStep = 3;
        }
        
        return result;
    }

    async scoreAndFilter() {
        const response = await fetch('http://localhost:5002/api/score-and-filter', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ session_id: this.sessionId })
        });
        
        const result = await response.json();
        
        if (result.success) {
            this.currentStep = 4;
            this.topImages = result.top_images.map(apiResult => ({
                url: `http://localhost:5002${apiResult.image_url}`,
                score: apiResult.final_score,
                linkedin: apiResult.linkedin_score,
                attire: apiResult.attire_score,
                neutrality: apiResult.neutrality_score,
                filename: apiResult.filename
            }));
        }
        
        return result;
    }

    async showTopImages() {
        // Hide processing and show results
        document.getElementById('processing-section').classList.add('hidden');
        document.getElementById('results-section').classList.remove('hidden');
        document.getElementById('results-section').scrollIntoView({ behavior: 'smooth' });

        // Populate photos grid
        const photosGrid = document.getElementById('photos-grid');
        photosGrid.innerHTML = '';

        this.topImages.forEach((imageData, index) => {
            const photoCard = document.createElement('div');
            photoCard.className = 'photo-card fade-in';
            photoCard.style.animationDelay = `${index * 0.1}s`;
            
            photoCard.innerHTML = `
                <img src="${imageData.url}" alt="Photo ${index + 1}">
                <div class="photo-info">
                    <div class="photo-score">Score: ${imageData.score}</div>
                    <div class="photo-details">
                        LinkedIn: ${imageData.linkedin} | 
                        Attire: ${imageData.attire} | 
                        Non-neutrality: ${imageData.neutrality}%
                    </div>
                </div>
                <div class="enhancement-overlay" id="overlay-${index}">
                    <div class="enhancement-spinner"></div>
                    <span>Enhancing...</span>
                </div>
            `;
            
            photosGrid.appendChild(photoCard);
        });
        
        // Start countdown and auto-enhancement
        this.startCountdown();
    }

    startCountdown() {
        let timeLeft = 5;
        const countdownElement = document.getElementById('countdown');
        
        const timer = setInterval(() => {
            timeLeft--;
            countdownElement.textContent = timeLeft;
            
            if (timeLeft <= 0) {
                clearInterval(timer);
                document.getElementById('auto-enhance-message').style.display = 'none';
                this.startEnhancement();
            }
        }, 1000);
    }

    async startEnhancement() {
        // No enhance button to hide anymore
        
        try {
            // Show enhancement overlays
            for (let i = 0; i < this.topImages.length; i++) {
                const overlay = document.getElementById(`overlay-${i}`);
                overlay.classList.add('active');
            }
            
            // Call enhancement API
            const response = await fetch('http://localhost:5002/api/enhance-images', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ session_id: this.sessionId })
            });
            
            const result = await response.json();
            
            // Hide enhancement overlays
            for (let i = 0; i < this.topImages.length; i++) {
                const overlay = document.getElementById(`overlay-${i}`);
                overlay.classList.remove('active');
            }
            
            if (result.success) {
                // Process enhanced images for display
                this.finalImages = result.enhanced_images.map(apiResult => ({
                    originalUrl: `http://localhost:5002${apiResult.original_url}`,
                    enhancedUrl: `http://localhost:5002${apiResult.enhanced_url}`,
                    score: apiResult.final_score,
                    linkedin: apiResult.linkedin_score,
                    attire: apiResult.attire_score,
                    neutrality: apiResult.neutrality_score
                }));
                
                this.showFinalResults();
            } else {
                throw new Error(result.message);
            }
            
        } catch (error) {
            console.error('Enhancement error:', error);
            this.showProcessingError('Enhancement failed: ' + error.message);
        }
    }

    showFinalResults() {
        // Hide results and show final section
        document.getElementById('results-section').classList.add('hidden');
        document.getElementById('final-section').classList.remove('hidden');
        document.getElementById('final-section').scrollIntoView({ behavior: 'smooth' });

        // Populate comparison grid
        const comparisonGrid = document.getElementById('comparison-grid');
        comparisonGrid.innerHTML = '';

        this.finalImages.forEach((imageData, index) => {
            const comparisonCard = document.createElement('div');
            comparisonCard.className = 'comparison-card fade-in';
            comparisonCard.style.animationDelay = `${index * 0.2}s`;
            
            comparisonCard.innerHTML = `
                <div class="before-image">
                    <img src="${imageData.originalUrl}" alt="Original">
                    <h4>Original</h4>
                </div>
                <div class="arrow">➜</div>
                <div class="after-image">
                    <img src="${imageData.enhancedUrl}" alt="Enhanced">
                    <h4>Enhanced</h4>
                </div>
            `;
            
            comparisonGrid.appendChild(comparisonCard);
        });
    }

    downloadAllImages() {
        // In a real application, this would download the enhanced images
        this.finalImages.forEach((imageData, index) => {
            const link = document.createElement('a');
            link.href = imageData.enhancedUrl;
            link.download = `enhanced_photo_${index + 1}.jpg`;
            link.click();
        });
    }

    restart() {
        // Reset all state
        this.referenceImage = null;
        this.uploadedImages = [];
        this.topImages = [];
        this.finalImages = [];

        // Hide all sections except reference
        document.getElementById('upload-section').classList.add('hidden');
        document.getElementById('processing-section').classList.add('hidden');
        document.getElementById('results-section').classList.add('hidden');
        document.getElementById('final-section').classList.add('hidden');

        // Reset reference section
        document.getElementById('reference-preview').classList.add('hidden');
        document.getElementById('reference-input').value = '';
        document.getElementById('file-input').value = '';

        // Scroll to top
        document.querySelector('.header').scrollIntoView({ behavior: 'smooth' });
    }

    showProcessingError(message) {
        // Hide processing section and show error
        document.getElementById('processing-section').classList.add('hidden');
        
        // Create error display
        const errorSection = document.createElement('section');
        errorSection.className = 'error-section';
        errorSection.innerHTML = `
            <div class="error-container">
                <h2>❌ Processing Error</h2>
                <p>${message}</p>
                <button class="restart-btn" onclick="this.parentElement.parentElement.remove(); new LinkedInOptimizer().restart();">
                    Try Again
                </button>
            </div>
        `;
        
        document.querySelector('.container').appendChild(errorSection);
        errorSection.scrollIntoView({ behavior: 'smooth' });
    }

    // Utility function
    delay(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }
}

// Initialize the application
document.addEventListener('DOMContentLoaded', () => {
    new LinkedInOptimizer();
});

// Remove dragover class when dragging leaves
document.addEventListener('dragleave', (e) => {
    if (e.target.classList.contains('upload-zone') || e.target.classList.contains('reference-upload')) {
        e.target.classList.remove('dragover');
    }
});