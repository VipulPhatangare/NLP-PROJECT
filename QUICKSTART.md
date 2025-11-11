# 🚀 Quick Start Guide

## Step 1: Installation (One-time setup)

Open PowerShell in the project directory and run:

```powershell
.\install.ps1
```

This will:
- Install Node.js dependencies
- Create Python virtual environment
- Install Python packages
- Download NLTK data

## Step 2: Start the Application

```powershell
.\start.ps1
```

Or simply:

```powershell
npm start
```

## Step 3: Open Dashboard

Open your browser and navigate to:
```
http://localhost:3000
```

## Step 4: Run Analysis

1. **Scrape Reviews** (15-30 minutes)
   - Click "Scrape Reviews" button
   - Wait for completion
   - Reviews saved to `data/flipkart_boat_raw.csv`

2. **Run Phase 1** (1-2 minutes)
   - Click "Run Phase 1" button
   - Preprocessing completes
   - View language distribution and stats

3. **Run Phase 2** (2-3 minutes)
   - Click "Run Phase 2" button
   - Analysis completes
   - View POS tags, entities, topics, similarities

4. **Explore Results**
   - Navigate through tabs
   - View charts and graphs
   - Export data as CSV

## Dashboard Features

### 📈 Overview Tab
- Total reviews count
- Language statistics
- Topics discovered
- Vocabulary size

### 🧹 Phase 1 Tab
- Language distribution pie chart
- Preprocessing statistics table
- Sample processed reviews
- Before/after comparison

### 📊 Phase 2 Tab
- POS tag distribution (bar chart)
- Top adjectives (horizontal bar)
- Top verbs (horizontal bar)
- Named entities (brands, features, etc.)
- LSA topics with keywords
- Word2Vec semantic similarities

### 📄 Data View Tab
- Browse complete raw data
- Browse cleaned data
- Export to CSV
- Searchable table

## Keyboard Shortcuts

- `Ctrl + C` - Stop server
- `F5` - Refresh dashboard
- `Ctrl + Shift + I` - Open developer tools

## Troubleshooting

### Cannot run PowerShell scripts
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Port 3000 already in use
Change port in `server.js`:
```javascript
const PORT = 3001; // Change to any available port
```

### Python packages not found
```powershell
.\venv\Scripts\Activate
pip install -r requirements.txt
```

### Charts not displaying
- Clear browser cache
- Check browser console for errors
- Ensure Chart.js CDN is accessible

## File Locations

- **Raw Data**: `data/flipkart_boat_raw.csv`
- **Cleaned Data**: `data/flipkart_boat_cleaned.csv`
- **Analysis Results**: `results/phase2_results.json`
- **Server Logs**: Console output

## API Endpoints (for testing)

Test with PowerShell:

```powershell
# Check status
Invoke-RestMethod -Uri "http://localhost:3000/api/status"

# Get raw data
Invoke-RestMethod -Uri "http://localhost:3000/api/data/raw"

# Get cleaned data
Invoke-RestMethod -Uri "http://localhost:3000/api/data/cleaned"

# Get Phase 2 results
Invoke-RestMethod -Uri "http://localhost:3000/api/results/phase2"
```

## Performance Tips

1. **Scraping**: Use smaller `totalPages` for testing (e.g., 10 pages)
2. **Memory**: Close other applications during Phase 2 analysis
3. **Browser**: Use Chrome or Firefox for best performance

## Next Steps

After completing all phases:
1. Export cleaned data for further analysis
2. Use results in your project report
3. Create visualizations for presentation
4. Document insights discovered

---

Need help? Check `README.md` for detailed documentation.
