// Canvas setup
const canvas = document.getElementById('drawingCanvas');
const ctx = canvas.getContext('2d');
let isDrawing = false;
let lastX = 0;
let lastY = 0;

// Set up canvas
ctx.fillStyle = 'white';
ctx.fillRect(0, 0, canvas.width, canvas.height);
ctx.strokeStyle = 'black';
ctx.lineWidth = 20; // Original thin pen size
ctx.lineCap = 'round';
ctx.lineJoin = 'round';

// Drawing functions
function startDrawing(e) {
    isDrawing = true;
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    
    if (e.type === 'mousedown') {
        lastX = (e.clientX - rect.left) * scaleX;
        lastY = (e.clientY - rect.top) * scaleY;
    } else if (e.type === 'touchstart') {
        e.preventDefault();
        lastX = (e.touches[0].clientX - rect.left) * scaleX;
        lastY = (e.touches[0].clientY - rect.top) * scaleY;
    }
}

function draw(e) {
    if (!isDrawing) return;
    
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    
    let currentX, currentY;
    
    if (e.type === 'mousemove') {
        currentX = (e.clientX - rect.left) * scaleX;
        currentY = (e.clientY - rect.top) * scaleY;
    } else if (e.type === 'touchmove') {
        e.preventDefault();
        currentX = (e.touches[0].clientX - rect.left) * scaleX;
        currentY = (e.touches[0].clientY - rect.top) * scaleY;
    }
    
    ctx.beginPath();
    ctx.moveTo(lastX, lastY);
    ctx.lineTo(currentX, currentY);
    ctx.stroke();
    
    lastX = currentX;
    lastY = currentY;
}

function stopDrawing() {
    isDrawing = false;
}

// Event listeners for mouse
canvas.addEventListener('mousedown', startDrawing);
canvas.addEventListener('mousemove', draw);
canvas.addEventListener('mouseup', stopDrawing);
canvas.addEventListener('mouseout', stopDrawing);

// Event listeners for touch
canvas.addEventListener('touchstart', startDrawing);
canvas.addEventListener('touchmove', draw);
canvas.addEventListener('touchend', stopDrawing);

// Clear canvas button
document.getElementById('clearBtn').addEventListener('click', () => {
    ctx.fillStyle = 'white';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    clearResults();
});

// Predict button
document.getElementById('predictBtn').addEventListener('click', async () => {
    // Get canvas data
    const imageData = canvas.toDataURL('image/png');
    
    // Show loading state
    showLoading();
    
    try {
        // Send to server
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ image: imageData })
        });
        
        if (!response.ok) {
            throw new Error('Prediction failed');
        }
        
        const result = await response.json();
        displayResults(result);
    } catch (error) {
        console.error('Error:', error);
        showError('Failed to get prediction. Please try again.');
    }
});

// File upload
const fileInput = document.getElementById('fileInput');
const fileName = document.getElementById('fileName');

fileInput.addEventListener('change', async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    
    fileName.textContent = file.name;
    
    // Show loading state
    showLoading();
    
    // Create FormData
    const formData = new FormData();
    formData.append('file', file);
    
    try {
        // Send to server
        const response = await fetch('/upload', {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            throw new Error('Upload failed');
        }
        
        const result = await response.json();
        displayResults(result);
        
        // Load image to canvas for visualization
        const reader = new FileReader();
        reader.onload = function(event) {
            const img = new Image();
            img.onload = function() {
                // Clear canvas
                ctx.fillStyle = 'white';
                ctx.fillRect(0, 0, canvas.width, canvas.height);
                
                // Draw image centered and scaled
                const scale = Math.min(canvas.width / img.width, canvas.height / img.height);
                const x = (canvas.width - img.width * scale) / 2;
                const y = (canvas.height - img.height * scale) / 2;
                ctx.drawImage(img, x, y, img.width * scale, img.height * scale);
            };
            img.src = event.target.result;
        };
        reader.readAsDataURL(file);
        
    } catch (error) {
        console.error('Error:', error);
        showError('Failed to process uploaded image. Please try again.');
        fileName.textContent = 'No file chosen';
    }
});

// Display functions
function showLoading() {
    const resultDiv = document.getElementById('predictionResult');
    resultDiv.innerHTML = `
        <div class="loading"></div>
        <p style="margin-top: 15px; color: #6c757d;">Processing...</p>
    `;
    document.getElementById('probabilityBars').style.display = 'none';
}

function displayResults(result) {
    const resultDiv = document.getElementById('predictionResult');
    const probabilityBars = document.getElementById('probabilityBars');
    const barsContainer = document.getElementById('barsContainer');
    
    // Check if we have multiple models
    if (result.models && result.models.length > 0) {
        // Store models data globally for click handlers
        window.currentModels = result.models;
        window.selectedModelIndex = 0; // Default to first model
        
        // Display predictions from all models
        let modelsHTML = '<h3 style="color: #495057; margin-bottom: 20px;">🎯 Predictions from All Models <span style="font-size: 0.7em; color: #7f8c8d;">(Click to view probabilities)</span></h3>';
        modelsHTML += '<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin-bottom: 20px;">';
        
        result.models.forEach((model, index) => {
            const confidencePercent = (model.confidence * 100).toFixed(2);
            const borderColor = model.color || '#3498db';
            modelsHTML += `
                <div class="model-card" data-model-index="${index}" style="border: 3px solid ${borderColor}; border-radius: 15px; padding: 20px; text-align: center; background: linear-gradient(135deg, #ffffff 0%, ${borderColor}15 100%); cursor: pointer; transition: all 0.3s;">
                    <div style="font-size: 2em; margin-bottom: 5px;">${model.icon}</div>
                    <div style="font-weight: 600; color: #2c3e50; margin-bottom: 10px;">${model.name}</div>
                    <div style="font-size: 3em; font-weight: bold; color: ${borderColor}; margin: 10px 0;">${model.prediction}</div>
                    <div style="font-size: 0.9em; color: #7f8c8d;">Confidence: <strong>${confidencePercent}%</strong></div>
                    <div style="margin-top: 10px; font-size: 0.8em; color: #3498db;">👆 Click to view details</div>
                </div>
            `;
        });
        
        modelsHTML += '</div>';
        
        // Check if all models agree
        const predictions = result.models.map(m => m.prediction);
        const allAgree = predictions.every(p => p === predictions[0]);
        
        if (allAgree) {
            modelsHTML += '<div style="background: #d4edda; border: 2px solid #28a745; border-radius: 10px; padding: 15px; text-align: center; color: #155724; font-weight: 600;">✅ All models agree! High confidence prediction.</div>';
        } else {
            modelsHTML += '<div style="background: #fff3cd; border: 2px solid #ffc107; border-radius: 10px; padding: 15px; text-align: center; color: #856404; font-weight: 600;">⚠️ Models disagree. Click each model to compare probabilities.</div>';
        }
        
        resultDiv.innerHTML = modelsHTML;
        resultDiv.classList.add('show');
        
        // Add click handlers to model cards
        document.querySelectorAll('.model-card').forEach(card => {
            card.addEventListener('click', function() {
                const modelIndex = parseInt(this.getAttribute('data-model-index'));
                window.selectedModelIndex = modelIndex;
                
                // Highlight selected card
                document.querySelectorAll('.model-card').forEach(c => {
                    c.style.transform = 'scale(1)';
                    c.style.boxShadow = 'none';
                });
                this.style.transform = 'scale(1.05)';
                this.style.boxShadow = '0 8px 20px rgba(0,0,0,0.2)';
                
                // Update probability bars
                updateProbabilityBars(window.currentModels[modelIndex]);
            });
        });
        
        // Display probability bars for the first model initially
        updateProbabilityBars(result.models[0]);
    } else {
        // Fallback to single model display
        const confidencePercent = (result.confidence * 100).toFixed(2);
        resultDiv.innerHTML = `
            <h3 style="color: #495057; margin-bottom: 15px;">Predicted Digit</h3>
            <div class="predicted-digit">${result.prediction}</div>
            <div class="confidence">
                Confidence: <span class="confidence-value">${confidencePercent}%</span>
            </div>
        `;
        resultDiv.classList.add('show');
        
        // Display probability bars
        barsContainer.innerHTML = '';
        const probabilities = result.probabilities;
        
        for (let i = 0; i < 10; i++) {
            const prob = probabilities[i.toString()];
            const percentage = (prob * 100).toFixed(1);
            const isMax = i === result.prediction;
            
            const barItem = document.createElement('div');
            barItem.className = 'bar-item';
            barItem.innerHTML = `
                <div class="bar-label">
                    <span class="bar-digit">Digit ${i}</span>
                    <span class="bar-percentage">${percentage}%</span>
                </div>
                <div class="bar-background">
                    <div class="bar-fill ${isMax ? 'highlight' : ''}" style="width: ${percentage}%">
                        ${percentage > 5 ? percentage + '%' : ''}
                    </div>
                </div>
            `;
            barsContainer.appendChild(barItem);
        }
    }
    
    probabilityBars.style.display = 'block';
}

function updateProbabilityBars(model) {
    const barsContainer = document.getElementById('barsContainer');
    const probabilityBars = document.getElementById('probabilityBars');
    
    // Clear and update with selected model's probabilities
    barsContainer.innerHTML = `
        <h4 style="margin-bottom: 15px; color: #495057;">
            ${model.icon} ${model.name} - Detailed Probabilities
        </h4>
    `;
    
    const probabilities = model.probabilities;
    
    for (let i = 0; i < 10; i++) {
        const prob = probabilities[i.toString()];
        const percentage = (prob * 100).toFixed(1);
        const isMax = i === model.prediction;
        
        const barItem = document.createElement('div');
        barItem.className = 'bar-item';
        barItem.innerHTML = `
            <div class="bar-label">
                <span class="bar-digit">Digit ${i}</span>
                <span class="bar-percentage">${percentage}%</span>
            </div>
            <div class="bar-background">
                <div class="bar-fill ${isMax ? 'highlight' : ''}" style="width: ${percentage}%; background: ${model.color};">
                    ${percentage > 5 ? percentage + '%' : ''}
                </div>
            </div>
        `;
        barsContainer.appendChild(barItem);
    }
    
    probabilityBars.style.display = 'block';
}

function clearResults() {
    const resultDiv = document.getElementById('predictionResult');
    resultDiv.innerHTML = '<p class="no-prediction">Draw a digit and click "Predict Digit" to see results</p>';
    resultDiv.classList.remove('show');
    document.getElementById('probabilityBars').style.display = 'none';
    window.currentModels = null;
    window.selectedModelIndex = 0;
}

function showError(message) {
    const resultDiv = document.getElementById('predictionResult');
    resultDiv.innerHTML = `<div class="error-message">${message}</div>`;
    document.getElementById('probabilityBars').style.display = 'none';
}

// Reset file input label when cleared
fileInput.addEventListener('click', () => {
    fileInput.value = '';
    fileName.textContent = 'No file chosen';
});
