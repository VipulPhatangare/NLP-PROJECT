# Quick Installation Script
# Run this once before starting the application

Write-Host "🎯 NLP Review Analysis - Installation" -ForegroundColor Cyan
Write-Host "======================================" -ForegroundColor Cyan
Write-Host ""

# Install Node dependencies
Write-Host "📦 Installing Node.js dependencies..." -ForegroundColor Yellow
npm install
if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ Node.js dependencies installed!" -ForegroundColor Green
} else {
    Write-Host "❌ Failed to install Node.js dependencies" -ForegroundColor Red
    exit 1
}
Write-Host ""

# Create virtual environment
Write-Host "🐍 Creating Python virtual environment..." -ForegroundColor Yellow
python -m venv venv
if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ Virtual environment created!" -ForegroundColor Green
} else {
    Write-Host "❌ Failed to create virtual environment" -ForegroundColor Red
    exit 1
}
Write-Host ""

# Activate and install Python dependencies
Write-Host "📦 Installing Python dependencies..." -ForegroundColor Yellow
& ".\venv\Scripts\Activate.ps1"
pip install -r requirements.txt
if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ Python dependencies installed!" -ForegroundColor Green
} else {
    Write-Host "❌ Failed to install Python dependencies" -ForegroundColor Red
    exit 1
}
Write-Host ""

# Download NLTK data
Write-Host "📚 Downloading NLTK data..." -ForegroundColor Yellow
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('averaged_perceptron_tagger'); nltk.download('averaged_perceptron_tagger_eng')"
if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ NLTK data downloaded!" -ForegroundColor Green
} else {
    Write-Host "⚠️  NLTK data download had issues (might work anyway)" -ForegroundColor Yellow
}
Write-Host ""

Write-Host "🎉 Installation complete!" -ForegroundColor Green
Write-Host ""
Write-Host "To start the application, run: .\start.ps1" -ForegroundColor Cyan
Write-Host "Or simply run: npm start" -ForegroundColor Cyan
