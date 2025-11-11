const express = require('express');
const cors = require('cors');
const { spawn } = require('child_process');
const fs = require('fs').promises;
const path = require('path');
const csvParser = require('csv-parser');
const { Readable } = require('stream');

const app = express();
const PORT = 3000;

// Middleware
app.use(cors());
app.use(express.json());
app.use(express.static('public'));

// Utility function to run Python scripts
function runPythonScript(scriptName, args = []) {
    return new Promise((resolve, reject) => {
        const pythonProcess = spawn('python', [path.join('scripts', scriptName), ...args]);
        
        let stdout = '';
        let stderr = '';
        
        pythonProcess.stdout.on('data', (data) => {
            stdout += data.toString();
            console.log(data.toString());
        });
        
        pythonProcess.stderr.on('data', (data) => {
            stderr += data.toString();
            console.error(data.toString());
        });
        
        pythonProcess.on('close', (code) => {
            if (code !== 0) {
                reject(new Error(`Python script exited with code ${code}: ${stderr}`));
            } else {
                try {
                    // Try to parse JSON output from last line
                    const lines = stdout.trim().split('\n');
                    const lastLine = lines[lines.length - 1];
                    const result = JSON.parse(lastLine);
                    resolve(result);
                } catch (e) {
                    resolve({ success: true, output: stdout });
                }
            }
        });
    });
}

// Check if file exists
async function fileExists(filePath) {
    try {
        await fs.access(filePath);
        return true;
    } catch {
        return false;
    }
}

// API Routes

// Check data status
app.get('/api/status', async (req, res) => {
    try {
        const rawDataExists = await fileExists('data/flipkart_boat_raw.csv');
        const cleanedDataExists = await fileExists('data/flipkart_boat_cleaned.csv');
        const phase2ResultsExist = await fileExists('results/phase2_results.json');
        const phase3ResultsExist = await fileExists('results/phase3_results.json');
        
        res.json({
            rawData: rawDataExists,
            cleanedData: cleanedDataExists,
            phase2Results: phase2ResultsExist,
            phase3Results: phase3ResultsExist
        });
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
});

// Scrape reviews with real-time progress
app.get('/api/scrape', async (req, res) => {
    try {
        const { totalPages = 60, batchSize = 15, delay = 60 } = req.query;
        
        // Set headers for SSE (Server-Sent Events)
        res.setHeader('Content-Type', 'text/event-stream');
        res.setHeader('Cache-Control', 'no-cache');
        res.setHeader('Connection', 'keep-alive');
        res.setHeader('Access-Control-Allow-Origin', '*');
        
        console.log('Starting scraping with real-time progress...');
        
        const pythonProcess = spawn('python', ['scripts/scraper.py', totalPages, batchSize, delay]);
        
        pythonProcess.stdout.on('data', (data) => {
            const output = data.toString();
            const lines = output.split('\n').filter(line => line.trim());
            
            lines.forEach(line => {
                if (line.includes('PROGRESS|') || line.includes('SUCCESS|') || line.includes('ERROR|')) {
                    const parts = line.split('|');
                    const type = parts[0];
                    
                    if (type === 'PROGRESS' && parts.length >= 3 && parts[1].includes('%')) {
                        // Format: PROGRESS|25%|100 reviews|message
                        const progress = {
                            type: 'progress',
                            percent: parseInt(parts[1]),
                            reviews: parts[2],
                            message: parts[3] || parts[1]
                        };
                        res.write(`data: ${JSON.stringify(progress)}\n\n`);
                    } else if (type === 'SUCCESS') {
                        // Format: SUCCESS|100%|350 reviews|message
                        const success = {
                            type: 'complete',
                            percent: 100,
                            reviews: parts[2],
                            message: parts[3] || 'Scraping complete!'
                        };
                        res.write(`data: ${JSON.stringify(success)}\n\n`);
                    } else if (type === 'PROGRESS') {
                        // Simple progress message
                        const progress = {
                            type: 'info',
                            message: parts.slice(1).join('|')
                        };
                        res.write(`data: ${JSON.stringify(progress)}\n\n`);
                    }
                }
                console.log(line);
            });
        });
        
        pythonProcess.stderr.on('data', (data) => {
            console.error(`Scraper error: ${data}`);
        });
        
        pythonProcess.on('close', (code) => {
            if (code === 0) {
                res.write(`data: ${JSON.stringify({ type: 'done' })}\n\n`);
            } else {
                res.write(`data: ${JSON.stringify({ type: 'error', message: 'Scraping failed' })}\n\n`);
            }
            res.end();
        });
        
        // Handle client disconnect
        req.on('close', () => {
            pythonProcess.kill();
            res.end();
        });
        
    } catch (error) {
        res.write(`data: ${JSON.stringify({ type: 'error', message: error.message })}\n\n`);
        res.end();
    }
});

// Run Phase 1 preprocessing
app.post('/api/phase1', async (req, res) => {
    try {
        console.log('Starting Phase 1 preprocessing...');
        const result = await runPythonScript('phase1_preprocessing.py');
        
        res.json({
            success: true,
            message: 'Phase 1 preprocessing completed',
            data: result
        });
    } catch (error) {
        res.status(500).json({
            success: false,
            error: error.message
        });
    }
});

// Run Phase 2 analysis
app.post('/api/phase2', async (req, res) => {
    try {
        console.log('Starting Phase 2 analysis...');
        const result = await runPythonScript('phase2_analysis.py');
        
        res.json({
            success: true,
            message: 'Phase 2 analysis completed',
            data: result
        });
    } catch (error) {
        res.status(500).json({
            success: false,
            error: error.message
        });
    }
});

// Run Phase 3 analysis
app.post('/api/phase3', async (req, res) => {
    try {
        console.log('Starting Phase 3 advanced analysis...');
        const result = await runPythonScript('phase3_advanced.py');
        
        res.json({
            success: true,
            message: 'Phase 3 advanced analysis completed',
            data: result
        });
    } catch (error) {
        res.status(500).json({
            success: false,
            error: error.message
        });
    }
});

// Clear all data (raw, cleaned, results)
app.post('/api/clear-data', async (req, res) => {
    try {
        console.log('Clearing all data files...');
        
        // List of files to delete
        const filesToDelete = [
            'data/flipkart_boat_raw.csv',
            'data/flipkart_boat_cleaned.csv',
            'data/phase1_stats.json',
            'results/phase2_results.json',
            'results/phase3_results.json'
        ];
        
        // Delete each file if it exists
        for (const file of filesToDelete) {
            try {
                await fs.unlink(file);
                console.log(`Deleted: ${file}`);
            } catch (error) {
                // Ignore error if file doesn't exist
                if (error.code !== 'ENOENT') {
                    console.error(`Error deleting ${file}:`, error);
                }
            }
        }
        
        res.json({
            success: true,
            message: 'All data files cleared successfully'
        });
    } catch (error) {
        res.status(500).json({
            success: false,
            error: error.message
        });
    }
});

// Get raw data
app.get('/api/data/raw', async (req, res) => {
    try {
        const filePath = 'data/flipkart_boat_raw.csv';
        const exists = await fileExists(filePath);
        
        if (!exists) {
            return res.status(404).json({ error: 'Raw data not found. Please scrape first.' });
        }
        
        const data = await fs.readFile(filePath, 'utf-8');
        const rows = [];
        
        Readable.from(data)
            .pipe(csvParser())
            .on('data', (row) => rows.push(row))
            .on('end', () => {
                res.json({
                    total: rows.length,
                    data: rows
                });
            })
            .on('error', (error) => {
                res.status(500).json({ error: error.message });
            });
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
});

// Get cleaned data
app.get('/api/data/cleaned', async (req, res) => {
    try {
        const filePath = 'data/flipkart_boat_cleaned.csv';
        const exists = await fileExists(filePath);
        
        if (!exists) {
            return res.status(404).json({ error: 'Cleaned data not found. Please run Phase 1 first.' });
        }
        
        const data = await fs.readFile(filePath, 'utf-8');
        const rows = [];
        
        Readable.from(data)
            .pipe(csvParser())
            .on('data', (row) => rows.push(row))
            .on('end', () => {
                res.json({
                    total: rows.length,
                    data: rows
                });
            })
            .on('error', (error) => {
                res.status(500).json({ error: error.message });
            });
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
});

// Get Phase 2 results
app.get('/api/results/phase2', async (req, res) => {
    try {
        const filePath = 'results/phase2_results.json';
        const exists = await fileExists(filePath);
        
        if (!exists) {
            return res.status(404).json({ error: 'Phase 2 results not found. Please run Phase 2 first.' });
        }
        
        const data = await fs.readFile(filePath, 'utf-8');
        const results = JSON.parse(data);
        
        res.json(results);
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
});

// Get Phase 1 stats (preprocessing steps)
app.get('/api/results/phase1', async (req, res) => {
    try {
        const filePath = 'data/phase1_stats.json';
        const exists = await fileExists(filePath);
        
        if (!exists) {
            return res.status(404).json({ error: 'Phase 1 stats not found. Please run Phase 1 first.' });
        }
        
        const data = await fs.readFile(filePath, 'utf-8');
        const stats = JSON.parse(data);
        
        res.json(stats);
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
});

// Get Phase 3 results
app.get('/api/results/phase3', async (req, res) => {
    try {
        const filePath = 'results/phase3_results.json';
        const exists = await fileExists(filePath);
        
        if (!exists) {
            return res.status(404).json({ error: 'Phase 3 results not found. Please run Phase 3 first.' });
        }
        
        const data = await fs.readFile(filePath, 'utf-8');
        const results = JSON.parse(data);
        
        res.json(results);
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
});

// Get dynamic stats
app.get('/api/stats', async (req, res) => {
    try {
        const stats = {
            totalReviews: 0,
            languages: 0,
            topicCount: 0,
            vocabSize: 0,
            avgReviewLength: 0,
            topLanguage: '',
            topFeature: '',
            uniqueEntities: 0
        };
        
        // Check if Phase 1 results exist (cleaned data)
        const cleanedPath = 'data/flipkart_boat_cleaned.csv';
        if (await fileExists(cleanedPath)) {
            const data = await fs.readFile(cleanedPath, 'utf-8');
            const lines = data.trim().split('\n');
            const rows = lines.slice(1); // Skip header
            stats.totalReviews = rows.length;
            
            // Calculate average review length
            let totalWords = 0;
            const languageCounts = {};
            
            rows.forEach(row => {
                // Parse CSV row (handle quoted fields)
                const matches = row.match(/(".*?"|[^,]+)(?=\s*,|\s*$)/g);
                if (matches && matches.length >= 3) {
                    const processedReview = matches[2].replace(/^"|"$/g, '');
                    const words = processedReview.split(/\s+/).filter(w => w.length > 0);
                    totalWords += words.length;
                    
                    // Count languages
                    if (matches.length >= 4) {
                        const lang = matches[3].replace(/^"|"$/g, '').trim();
                        languageCounts[lang] = (languageCounts[lang] || 0) + 1;
                    }
                }
            });
            
            if (stats.totalReviews > 0) {
                stats.avgReviewLength = Math.round(totalWords / stats.totalReviews);
            }
            
            // Get top language
            if (Object.keys(languageCounts).length > 0) {
                const topLang = Object.entries(languageCounts)
                    .sort((a, b) => b[1] - a[1])[0];
                
                const langMap = {
                    'en': 'English', 'hi': 'Hindi', 'es': 'Spanish', 'fr': 'French', 
                    'de': 'German', 'pt': 'Portuguese', 'ru': 'Russian', 'ja': 'Japanese'
                };
                stats.topLanguage = langMap[topLang[0]] || topLang[0].toUpperCase();
            }
        }
        
        // Check if Phase 2 results exist
        const phase2Path = 'results/phase2_results.json';
        if (await fileExists(phase2Path)) {
            const data = await fs.readFile(phase2Path, 'utf-8');
            const results = JSON.parse(data);
            
            // Count unique languages
            if (results.language_distribution) {
                stats.languages = Object.keys(results.language_distribution).length;
            }
            
            // Get topic count from LSA
            if (results.lsa_topics && results.lsa_topics.topics) {
                stats.topicCount = results.lsa_topics.topics.length;
            }
            
            // Get vocabulary size
            if (results.bow_tfidf && results.bow_tfidf.vocab_size) {
                stats.vocabSize = results.bow_tfidf.vocab_size;
            }
            
            // Get top feature from NER analysis
            if (results.ner_analysis && results.ner_analysis.top_entities) {
                const topEntity = results.ner_analysis.top_entities.find(e => e.type === 'FEATURE');
                if (topEntity) {
                    stats.topFeature = topEntity.value.charAt(0).toUpperCase() + topEntity.value.slice(1);
                }
            }
            
            // Count unique entities
            if (results.ner_analysis && results.ner_analysis.entity_stats) {
                let uniqueCount = 0;
                Object.values(results.ner_analysis.entity_stats).forEach(stat => {
                    uniqueCount += stat.unique || 0;
                });
                stats.uniqueEntities = uniqueCount;
            }
        }
        
        res.json(stats);
    } catch (error) {
        console.error('Error fetching stats:', error);
        res.status(500).json({ error: error.message });
    }
});

// Create necessary directories
async function initializeDirectories() {
    const dirs = ['data', 'results', 'public'];
    for (const dir of dirs) {
        try {
            await fs.mkdir(dir, { recursive: true });
        } catch (error) {
            console.error(`Error creating ${dir}:`, error);
        }
    }
}

// Start server
initializeDirectories().then(() => {
    app.listen(PORT, () => {
        console.log(`🚀 Server running on http://localhost:${PORT}`);
        console.log(`📊 NLP Review Analysis System ready!`);
    });
});
