# NLP Review Analysis - Startup Script
# Run this script to start the application

Write-Host "NLP Review Analysis Dashboard" -ForegroundColor Cyan
Write-Host "=================================" -ForegroundColor Cyan
Write-Host ""

# Check if node_modules exists
if (-not (Test-Path "node_modules")) {
    Write-Host "Installing Node.js dependencies..." -ForegroundColor Yellow
    npm install
    Write-Host "Node.js dependencies installed!" -ForegroundColor Green
    Write-Host ""
}

# Check if virtual environment exists
if (-not (Test-Path "venv")) {
    Write-Host "Creating Python virtual environment..." -ForegroundColor Yellow
    python -m venv venv
    Write-Host "Virtual environment created!" -ForegroundColor Green
    Write-Host ""
}

# Activate virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Yellow
& ".\venv\Scripts\Activate.ps1"

# Install Python dependencies
Write-Host "Installing Python dependencies..." -ForegroundColor Yellow
pip install -r requirements.txt --quiet
Write-Host "Python dependencies installed!" -ForegroundColor Green
Write-Host ""

# Download NLTK data
Write-Host "Downloading NLTK data..." -ForegroundColor Yellow
python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('punkt_tab', quiet=True); nltk.download('stopwords', quiet=True); nltk.download('wordnet', quiet=True); nltk.download('averaged_perceptron_tagger', quiet=True)"
Write-Host "NLTK data downloaded!" -ForegroundColor Green
Write-Host ""

# Start server
Write-Host "Starting server..." -ForegroundColor Cyan
Write-Host ""
Write-Host "Dashboard will be available at: http://localhost:3000" -ForegroundColor Green
Write-Host "Press Ctrl+C to stop the server" -ForegroundColor Yellow
Write-Host ""

npm start
