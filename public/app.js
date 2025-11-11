const API_BASE = 'http://localhost:3000/api';

let currentData = {
    phase1: null,
    phase2: null,
    raw: null,
    cleaned: null
};

let charts = {};

// Language code to name mapping
const LANGUAGE_NAMES = {
    'en': 'English', 'hi': 'Hindi', 'es': 'Spanish', 'fr': 'French', 'de': 'German',
    'it': 'Italian', 'pt': 'Portuguese', 'ru': 'Russian', 'ja': 'Japanese', 'ko': 'Korean',
    'zh': 'Chinese', 'ar': 'Arabic', 'tr': 'Turkish', 'nl': 'Dutch', 'pl': 'Polish',
    'sv': 'Swedish', 'no': 'Norwegian', 'da': 'Danish', 'fi': 'Finnish', 'cs': 'Czech',
    'hu': 'Hungarian', 'ro': 'Romanian', 'sk': 'Slovak', 'bg': 'Bulgarian', 'hr': 'Croatian',
    'sl': 'Slovenian', 'sr': 'Serbian', 'uk': 'Ukrainian', 'el': 'Greek', 'he': 'Hebrew',
    'th': 'Thai', 'vi': 'Vietnamese', 'id': 'Indonesian', 'ms': 'Malay', 'tl': 'Tagalog',
    'sw': 'Swahili', 'af': 'Afrikaans', 'sq': 'Albanian', 'ca': 'Catalan', 'cy': 'Welsh',
    'et': 'Estonian', 'lv': 'Latvian', 'lt': 'Lithuanian', 'mk': 'Macedonian', 'mt': 'Maltese',
    'is': 'Icelandic', 'ga': 'Irish', 'eu': 'Basque', 'gl': 'Galician', 'so': 'Somali'
};

// Toast Notification System
function showToast(title, message, type = 'info') {
    const container = document.getElementById('toastContainer');
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    
    const icons = {
        success: '✅',
        error: '❌',
        warning: '⚠️',
        info: 'ℹ️'
    };
    
    toast.innerHTML = `
        <div class="toast-icon">${icons[type] || icons.info}</div>
        <div class="toast-content">
            <div class="toast-title">${title}</div>
            <div class="toast-message">${message}</div>
        </div>
    `;
    
    container.appendChild(toast);
    
    // Auto remove after 3 seconds
    setTimeout(() => {
        toast.style.animation = 'slideIn 0.3s ease reverse';
        setTimeout(() => toast.remove(), 300);
    }, 3000);
}

// Animate stat value update
function updateStatValue(elementId, newValue) {
    const element = document.getElementById(elementId);
    element.classList.add('updating');
    setTimeout(() => {
        element.textContent = newValue;
        element.classList.remove('updating');
    }, 150);
}

// Initialize on load
document.addEventListener('DOMContentLoaded', () => {
    checkStatus();
    loadDynamicStats();
});

// API Functions
async function apiCall(endpoint, method = 'GET', body = null, showAlert = true) {
    const options = {
        method,
        headers: {
            'Content-Type': 'application/json'
        }
    };
    
    if (body) {
        options.body = JSON.stringify(body);
    }
    
    try {
        const response = await fetch(`${API_BASE}${endpoint}`, options);
        const data = await response.json();
        
        if (!response.ok) {
            throw new Error(data.error || 'API request failed');
        }
        
        return data;
    } catch (error) {
        console.error('API Error:', error);
        if (showAlert) {
            alert(`Error: ${error.message}`);
        }
        throw error;
    }
}

// Check Status
async function checkStatus() {
    try {
        const status = await apiCall('/status', 'GET', null, false);
        
        updateStatusBadge('statusRaw', status.rawData);
        updateStatusBadge('statusPhase1', status.cleanedData);
        updateStatusBadge('statusPhase2', status.phase2Results);
        updateStatusBadge('statusPhase3', status.phase3Results);
        
        // Only load data that exists, without showing errors
        if (status.cleanedData) {
            try {
                await apiCall('/data/cleaned', 'GET', null, false);
                // Only load dynamic stats if cleaned data exists
                await loadDynamicStats();
            } catch (e) {
                // Silently ignore - data might not be ready
            }
        }
        
        if (status.phase2Results) {
            document.getElementById('tabsContainer').style.display = 'flex';
            try {
                await loadDynamicStats();
            } catch (e) {
                // Silently ignore
            }
        }
        
        if (status.phase3Results) {
            try {
                await loadPhase3Results();
            } catch (e) {
                // Silently ignore
            }
        }
    } catch (error) {
        console.error('Status check failed:', error);
    }
}

// Load dynamic statistics
async function loadDynamicStats() {
    try {
        // Call the /stats endpoint which reads from Phase 1 & 2 results
        const stats = await apiCall('/stats');
        
        // Update and show/hide Total Reviews
        const totalReviews = stats.totalReviews || 0;
        document.getElementById('totalReviews').textContent = totalReviews;
        document.getElementById('totalReviewsCard').style.display = totalReviews > 0 ? 'block' : 'none';
        
        // Update and show/hide Languages Detected
        const languages = stats.languages || 0;
        document.getElementById('languageCount').textContent = languages;
        document.getElementById('languageCountCard').style.display = languages > 0 ? 'block' : 'none';
        
        // Update and show/hide Topics Found
        const topics = stats.topicCount || 0;
        document.getElementById('topicCount').textContent = topics;
        document.getElementById('topicCountCard').style.display = topics > 0 ? 'block' : 'none';
        
        // Update and show/hide Vocabulary Size
        const vocabSize = stats.vocabSize || 0;
        document.getElementById('vocabSize').textContent = vocabSize;
        document.getElementById('vocabSizeCard').style.display = vocabSize > 0 ? 'block' : 'none';
        
    // (avg review length card removed)
        
        // Update and show/hide Top Language
        // const topLanguage = stats.topLanguage || '';
        // document.getElementById('topLanguage').textContent = topLanguage || '-';
        // document.getElementById('topLanguageCard').style.display = topLanguage ? 'block' : 'none';
        
        // Update and show/hide Top Feature
        const topFeature = stats.topFeature || '';
        document.getElementById('topFeature').textContent = topFeature || '-';
        document.getElementById('topFeatureCard').style.display = topFeature ? 'block' : 'none';
        
        // Update and show/hide Unique Entities
        const uniqueEntities = stats.uniqueEntities || 0;
        document.getElementById('uniqueEntities').textContent = uniqueEntities;
        document.getElementById('uniqueEntitiesCard').style.display = uniqueEntities > 0 ? 'block' : 'none';
        
        // Load Phase 2 results to get language details
        if (languages > 0) {
            try {
                const phase2 = await apiCall('/results/phase2');
                if (phase2.language_distribution) {
                    displayLanguageTags(phase2.language_distribution);
                }
            } catch (error) {
                console.error('Failed to load language details:', error);
            }
        }
    } catch (error) {
        console.error('Failed to load dynamic stats:', error);
        // Hide all cards on error
        document.getElementById('totalReviewsCard').style.display = 'none';
        document.getElementById('languageCountCard').style.display = 'none';
        document.getElementById('topicCountCard').style.display = 'none';
        document.getElementById('vocabSizeCard').style.display = 'none';
    // avg review length card removed
        // document.getElementById('topLanguageCard').style.display = 'none';
        document.getElementById('topFeatureCard').style.display = 'none';
        document.getElementById('uniqueEntitiesCard').style.display = 'none';
    }
}

// Display language tags with full names
function displayLanguageTags(languages) {
    if (!languages || Object.keys(languages).length === 0) {
        document.getElementById('languageDetailsSection').style.display = 'none';
        return;
    }
    
    const container = document.getElementById('languageTags');
    const section = document.getElementById('languageDetailsSection');
    container.innerHTML = '';
    
    // Sort by count (descending)
    const sortedLangs = Object.entries(languages)
        .sort((a, b) => b[1] - a[1])
        .slice(0, 10); // Show top 10
    
    sortedLangs.forEach(([code, count]) => {
        const langName = LANGUAGE_NAMES[code] || code.toUpperCase();
        const tag = document.createElement('div');
        tag.className = 'language-tag';
        tag.innerHTML = `
            <span class="language-tag-flag">🌐</span>
            <span class="language-tag-name">${langName}</span>
            <span class="language-tag-count">${count}</span>
        `;
        container.appendChild(tag);
    });
    
    section.style.display = 'block';
}

function updateStatusBadge(elementId, isComplete) {
    const element = document.getElementById(elementId);
    const badge = element.querySelector('.badge');
    
    if (isComplete) {
        badge.textContent = 'Complete';
        badge.className = 'badge badge-complete';
    } else {
        badge.textContent = 'Pending';
        badge.className = 'badge badge-pending';
    }
}

// Scraping with real-time progress
async function startScraping() {
    const btn = document.getElementById('btnScrape');
    btn.disabled = true;
    
    // Reset all statuses to pending
    updateStatusBadge('statusRaw', false);
    updateStatusBadge('statusPhase1', false);
    updateStatusBadge('statusPhase2', false);
    
    // Reset stats
    updateStatValue('totalReviews', '0');
    updateStatValue('languageCount', '0');
    updateStatValue('topicCount', '0');
    updateStatValue('vocabSize', '0');
    document.getElementById('languageDetailsSection').style.display = 'none';
    
    // Clear all data from data folder
    try {
        await apiCall('/clear-data', 'POST');
        console.log('Data folder cleared');
    } catch (error) {
        console.error('Failed to clear data folder:', error);
    }
    
    // Reset current data in memory
    currentData = { raw: null, cleaned: null, phase2: null };
    
    showToast('Scraping Started', 'Initiating web scraper for Flipkart reviews', 'info');
    
    // Show progress container
    const loadingDiv = document.getElementById('loading');
    loadingDiv.style.display = 'block';
    loadingDiv.innerHTML = `
        <div class="loading-content">
            <div class="progress-container">
                <h3>🔍 Scraping Reviews from Flipkart</h3>
                <div class="progress-bar-container">
                    <div class="progress-bar" id="scrapeProgressBar" style="width: 0%"></div>
                </div>
                <div class="progress-text">
                    <span id="scrapePercent">0%</span>
                    <span id="scrapeReviews">0 reviews scraped</span>
                </div>
                <p id="scrapeMessage">Initializing scraper...</p>
            </div>
        </div>
    `;
    
    try {
        const eventSource = new EventSource(`${API_BASE}/scrape?totalPages=60&batchSize=15&delay=60`);
        
        eventSource.onmessage = (event) => {
            const data = JSON.parse(event.data);
            
            if (data.type === 'progress') {
                // Update progress bar
                const percent = data.percent || 0;
                const progressBar = document.getElementById('scrapeProgressBar');
                progressBar.style.width = percent + '%';
                progressBar.textContent = percent + '%';
                
                document.getElementById('scrapePercent').textContent = percent + '%';
                
                // Parse review count - handle different formats
                const reviewCount = data.reviews ? data.reviews.replace(' reviews', '') : '0';
                document.getElementById('scrapeReviews').textContent = reviewCount + ' reviews scraped';
                document.getElementById('scrapeMessage').textContent = data.message || 'Processing...';
                
                // Show milestone toasts
                if (percent === 25 || percent === 50 || percent === 75) {
                    showToast(`${percent}% Complete`, `${reviewCount} reviews collected`, 'info');
                }
            } else if (data.type === 'info') {
                document.getElementById('scrapeMessage').textContent = data.message;
            } else if (data.type === 'complete') {
                const progressBar = document.getElementById('scrapeProgressBar');
                progressBar.style.width = '100%';
                progressBar.textContent = '100%';
                
                document.getElementById('scrapePercent').textContent = '100%';
                
                // Parse review count - handle different formats
                const reviewCount = data.reviews ? data.reviews.replace(' reviews', '') : (data.total || '0');
                document.getElementById('scrapeReviews').textContent = reviewCount + ' reviews scraped';
                document.getElementById('scrapeMessage').textContent = '✅ ' + data.message;
                
                setTimeout(async () => {
                    eventSource.close();
                    hideLoading();
                    btn.disabled = false;
                    showToast('Scraping Complete!', `Successfully scraped ${reviewCount} reviews`, 'success');
                    await checkStatus();
                    
                    // Wait a bit for file to be written, then try to load
                    setTimeout(async () => {
                        await loadRawData();
                    }, 1000);
                }, 2000);
            } else if (data.type === 'error') {
                eventSource.close();
                hideLoading();
                btn.disabled = false;
                showToast('Scraping Failed', data.message, 'error');
            } else if (data.type === 'done') {
                eventSource.close();
            }
        };
        
        eventSource.onerror = (error) => {
            console.error('EventSource error:', error);
            eventSource.close();
            hideLoading();
            btn.disabled = false;
            showToast('Connection Error', 'Lost connection during scraping', 'error');
        };
        
    } catch (error) {
        hideLoading();
        btn.disabled = false;
        showToast('Scraping Failed', error.message, 'error');
    }
}

// Phase 1
async function runPhase1() {
    const btn = document.getElementById('btnPhase1');
    btn.disabled = true;
    showLoading('Running Phase 1 preprocessing...');
    showToast('🔄 Phase 1 Started', 'Preprocessing and cleaning review data...', 'info');
    
    try {
        const result = await apiCall('/phase1', 'POST');
        currentData.phase1 = result.data;
        
        showToast('✅ Phase 1 Complete!', `Processed ${result.data.total_reviews} reviews successfully`, 'success', 7000);
        await checkStatus();
        await loadDynamicStats();
        await loadCleanedData(); // Load cleaned data after Phase 1
    } catch (error) {
        showToast('❌ Phase 1 Failed', 'Make sure you have scraped data first', 'error');
    } finally {
        hideLoading();
        btn.disabled = false;
    }
}

// Phase 2
async function runPhase2() {
    const btn = document.getElementById('btnPhase2');
    btn.disabled = true;
    showLoading('Running Phase 2 analysis... This may take a few minutes.');
    showToast('🔄 Phase 2 Started', 'Performing NLP analysis on cleaned data...', 'info');
    
    try {
        const result = await apiCall('/phase2', 'POST');
        
        showToast('✅ Phase 2 Complete!', 'Analysis results are ready! Check the tabs below.', 'success', 7000);
        await checkStatus();
        await loadDynamicStats();
        await loadPhase2Results(); // Load Phase 2 results after completion
    } catch (error) {
        showToast('❌ Phase 2 Failed', 'Make sure Phase 1 is completed first', 'error');
    } finally {
        hideLoading();
        btn.disabled = false;
    }
}

// Phase 3
async function runPhase3() {
    const btn = document.getElementById('btnPhase3');
    btn.disabled = true;
    showLoading('Running Phase 3 advanced analysis... This may take a few minutes.');
    showToast('🔄 Phase 3 Started', 'Performing review summarization and Q&A generation...', 'info');
    
    try {
        const result = await apiCall('/phase3', 'POST');
        
        showToast('✅ Phase 3 Complete!', 'Advanced analysis ready! Check Phase 3 tab.', 'success', 7000);
        await checkStatus();
        await loadPhase3Results(); // Load Phase 3 results after completion
    } catch (error) {
        showToast('❌ Phase 3 Failed', 'Make sure Phase 1 and Phase 2 are completed first', 'error');
    } finally {
        hideLoading();
        btn.disabled = false;
    }
}

// Load Data
async function loadRawData() {
    try {
        const data = await apiCall('/data/raw', 'GET', null, false); // Don't show alert
        currentData.raw = data;
        console.log('Raw data loaded successfully');
    } catch (error) {
        // Silently fail - data might not be ready yet
        console.log('Raw data not available yet');
    }
}

async function loadCleanedData() {
    try {
        const data = await apiCall('/data/cleaned', 'GET', null, false);
        currentData.cleaned = data;
        
        // Also load Phase 1 stats (preprocessing steps)
        try {
            const phase1Stats = await apiCall('/results/phase1', 'GET', null, false);
            currentData.phase1 = phase1Stats;
        } catch (error) {
            console.log('Phase 1 stats not available yet');
        }
        
        displayPhase1Results();
    } catch (error) {
        console.log('Cleaned data not available yet');
    }
}

async function loadPhase2Results() {
    try {
        const data = await apiCall('/results/phase2', 'GET', null, false);
        currentData.phase2 = data;
        displayPhase2Results();
        updateOverview();
    } catch (error) {
        console.log('Phase 2 results not available yet');
    }
}

async function loadPhase3Results() {
    try {
        const data = await apiCall('/results/phase3', 'GET', null, false);
        currentData.phase3 = data;
        displayPhase3Results();
    } catch (error) {
        console.log('Phase 3 results not available yet');
    }
}

// Display Functions
function displayPhase1Results() {
    if (!currentData.cleaned) return;
    
    const data = currentData.cleaned.data;
    
    // Display Preprocessing Steps
    if (currentData.phase1 && currentData.phase1.preprocessing_steps) {
        const stepsHtml = currentData.phase1.preprocessing_steps.map((step, index) => `
            <div class="step-card">
                <div class="step-header">
                    <div class="step-number">${index + 1}</div>
                    <div class="step-title">${step.step}</div>
                </div>
                <div class="step-description">${step.description}</div>
                <div class="step-stats">📊 ${step.stats}</div>
                <div class="step-technique"><strong>Technique:</strong> ${step.technique}</div>
                ${step.example_before ? `
                    <div class="step-example">
                        <div class="example-label">Example:</div>
                        <div class="example-before">
                            <span class="example-tag">Before:</span> ${step.example_before}
                        </div>
                        <div class="example-after">
                            <span class="example-tag">After:</span> ${step.example_after}
                        </div>
                    </div>
                ` : ''}
            </div>
        `).join('');
        document.getElementById('preprocessingSteps').innerHTML = stepsHtml;
    }
    
    // Language Chart - Convert language codes to full names
    const languages = {};
    data.forEach(row => {
        const langCode = row.language || 'unknown';
        const langName = LANGUAGE_NAMES[langCode] || langCode;
        languages[langName] = (languages[langName] || 0) + 1;
    });
    
    createPieChart('languageChart', 'Language Distribution', languages);
    
    // Stats Table
    const statsHtml = `
        <table>
            <tr><th>Metric</th><th>Value</th></tr>
            <tr><td>Total Reviews</td><td>${data.length}</td></tr>
            <tr><td>Unique Languages</td><td>${Object.keys(languages).length}</td></tr>
            <tr><td>Avg Review Length</td><td>${calculateAvgLength(data)} words</td></tr>
        </table>
    `;
    document.getElementById('preprocessingStats').innerHTML = statsHtml;
    
    // Sample Transformations - Show complete pipeline
    const samplesHtml = data.slice(0, 5).map(row => {
        const langCode = row.language || 'unknown';
        const langName = LANGUAGE_NAMES[langCode] || langCode;
        
        return `
            <div class="sample-card">
                <div class="sample-original">
                    <strong>Original (${langName}):</strong> ${row.Review || ''}
                </div>
                ${row.Translated_Review && row.Translated_Review !== row.Review ? 
                    `<div class="sample-processed"><strong>Translated:</strong> ${row.Translated_Review}</div>` : ''}
                <div class="sample-processed">
                    <strong>Final Processed:</strong> ${row.Processed_Review || ''}
                </div>
            </div>
        `;
    }).join('');
    document.getElementById('sampleReviews').innerHTML = samplesHtml;
}

function displayPhase2Results() {
    if (!currentData.phase2) return;
    
    const data = currentData.phase2;
    // Show top language in Phase 2 tab if available
    try {
        const topLangEl = document.getElementById('phase2TopLanguage');
        let topLangText = '-';
        if (data.language_distribution) {
            const entries = Object.entries(data.language_distribution);
            if (entries.length > 0) {
                entries.sort((a, b) => b[1] - a[1]);
                const code = entries[0][0];
                topLangText = LANGUAGE_NAMES[code] || code;
            }
        }
        if (topLangEl) topLangEl.textContent = topLangText;
    } catch (e) {
        console.error('Failed to set Phase 2 top language:', e);
    }
    
    // Display Sentiment Analysis
    if (data.overall_sentiment) {
        displayOverallSentiment(data.overall_sentiment);
    }
    
    if (data.sentiment_lexicon) {
        displaySentimentChart('vaderSentimentChart', 'VADER Sentiment', data.sentiment_lexicon.sentiment_distribution);
    }
    
    if (data.sentiment_lstm) {
        displaySentimentChart('lstmSentimentChart', 'LSTM Sentiment', data.sentiment_lstm.sentiment_distribution);
    }
    
    if (data.sentiment_lexicon && data.sentiment_lstm) {
        displaySentimentDetails(data.sentiment_lexicon, data.sentiment_lstm);
    }
    
    if (data.key_sentiment_phrases) {
        displayKeyPhrases(data.key_sentiment_phrases);
    }
    
    // POS Chart
    if (data.pos_analysis && data.pos_analysis.pos_distribution) {
        const posData = data.pos_analysis.pos_distribution.slice(0, 15);
        createBarChart('posChart', 'POS Tag Distribution', 
            posData.map(d => d.POS_Tag),
            posData.map(d => d.Count)
        );
    }
    
    // Adjectives & Verbs
    if (data.pos_analysis) {
        if (data.pos_analysis.top_adjectives) {
            createHorizontalBarChart('adjectivesChart', 'Top Adjectives',
                data.pos_analysis.top_adjectives.slice(0, 10).map(d => d[0]),
                data.pos_analysis.top_adjectives.slice(0, 10).map(d => d[1])
            );
        }
        
        if (data.pos_analysis.top_verbs) {
            createHorizontalBarChart('verbsChart', 'Top Verbs',
                data.pos_analysis.top_verbs.slice(0, 10).map(d => d[0]),
                data.pos_analysis.top_verbs.slice(0, 10).map(d => d[1])
            );
        }
    }
    
    // NER Results
    if (data.ner_analysis && data.ner_analysis.entity_stats) {
        displayNERResults(data.ner_analysis.entity_stats);
    }
    
    // Topics
    if (data.lsa_topics && data.lsa_topics.topics) {
        displayTopics(data.lsa_topics.topics);
    }
    
    // Word2Vec
    if (data.word2vec_similarities) {
        displayWord2Vec(data.word2vec_similarities);
    }
}

function displayNERResults(entityStats) {
    let html = '<div class="ner-grid">';
    
    for (const [type, stats] of Object.entries(entityStats)) {
        html += `
            <div class="ner-card">
                <div class="ner-type">${type}</div>
                <p><strong>Total:</strong> ${stats.count}</p>
                <p><strong>Unique:</strong> ${stats.unique}</p>
                <div style="margin-top: 15px;">
                    <strong>Top Values:</strong>
                    ${Object.entries(stats.top_values).slice(0, 5).map(([val, count]) => `
                        <div class="ner-entity">
                            <span>${val}</span>
                            <span class="entity-count">${count}</span>
                        </div>
                    `).join('')}
                </div>
            </div>
        `;
    }
    
    html += '</div>';
    document.getElementById('nerResults').innerHTML = html;
}

function displayTopics(topics) {
    const html = topics.map(topic => `
        <div class="topic-card">
            <div class="topic-title">${topic.topic}</div>
            <div class="topic-words">
                ${topic.top_words.map(word => `<span class="topic-word">${word}</span>`).join('')}
            </div>
            <p style="margin-top: 10px; color: var(--text-light);">
                Variance Explained: ${(topic.variance_explained * 100).toFixed(2)}%
            </p>
        </div>
    `).join('');
    
    document.getElementById('topicsContainer').innerHTML = html;
}

function displayWord2Vec(similarities) {
    let html = '<div class="similarity-grid">';
    
    for (const [term, similar] of Object.entries(similarities)) {
        if (similar.length > 0) {
            html += `
                <div class="similarity-card">
                    <div class="similarity-term">🔤 ${term}</div>
                    ${similar.map(([word, score]) => `
                        <div class="similar-word">
                            <span>${word}</span>
                            <span class="similarity-score">${score.toFixed(3)}</span>
                        </div>
                    `).join('')}
                </div>
            `;
        }
    }
    
    html += '</div>';
    document.getElementById('word2vecContainer').innerHTML = html;
}

// Sentiment Analysis Display Functions
function displayOverallSentiment(sentiment) {
    const verdictClass = sentiment.verdict === 'Positive' ? 'positive' : (sentiment.verdict === 'Negative' ? 'negative' : 'neutral');
    
    const html = `
        <div class="overall-sentiment-card ${verdictClass}">
            <div class="sentiment-verdict">
                <h3>Overall Verdict: ${sentiment.verdict}</h3>
                <p class="sentiment-score">Score: ${sentiment.overall_score.toFixed(3)}</p>
            </div>
            <div class="sentiment-breakdown">
                <div class="sentiment-bar-container">
                    <div class="sentiment-label">Positive</div>
                    <div class="sentiment-bar-bg">
                        <div class="sentiment-bar positive-bar" style="width: ${sentiment.positive_percentage}%">
                            ${sentiment.positive_percentage}%
                        </div>
                    </div>
                </div>
                <div class="sentiment-bar-container">
                    <div class="sentiment-label">Neutral</div>
                    <div class="sentiment-bar-bg">
                        <div class="sentiment-bar neutral-bar" style="width: ${sentiment.neutral_percentage}%">
                            ${sentiment.neutral_percentage}%
                        </div>
                    </div>
                </div>
                <div class="sentiment-bar-container">
                    <div class="sentiment-label">Negative</div>
                    <div class="sentiment-bar-bg">
                        <div class="sentiment-bar negative-bar" style="width: ${sentiment.negative_percentage}%">
                            ${sentiment.negative_percentage}%
                        </div>
                    </div>
                </div>
            </div>
        </div>
    `;
    
    document.getElementById('overallSentiment').innerHTML = html;
}

function displaySentimentChart(canvasId, title, distribution) {
    const ctx = document.getElementById(canvasId).getContext('2d');
    
    const colors = {
        'positive': 'rgba(16, 185, 129, 0.8)',
        'neutral': 'rgba(107, 114, 128, 0.8)',
        'negative': 'rgba(239, 68, 68, 0.8)'
    };
    
    new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: Object.keys(distribution).map(k => k.charAt(0).toUpperCase() + k.slice(1)),
            datasets: [{
                data: Object.values(distribution),
                backgroundColor: Object.keys(distribution).map(k => colors[k]),
                borderWidth: 2,
                borderColor: '#fff'
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'bottom'
                },
                title: {
                    display: true,
                    text: title
                }
            }
        }
    });
}

function displaySentimentDetails(vader, lstm) {
    const html = `
        <div class="sentiment-methods">
            <div class="method-card">
                <h4>📖 VADER (Lexicon-based)</h4>
                <p class="method-name">${vader.method}</p>
                <div class="method-stats">
                    <div><strong>Avg Compound:</strong> ${vader.average_scores.avg_compound.toFixed(3)}</div>
                    <div><strong>Avg Positive:</strong> ${vader.average_scores.avg_positive.toFixed(3)}</div>
                    <div><strong>Avg Negative:</strong> ${vader.average_scores.avg_negative.toFixed(3)}</div>
                </div>
            </div>
            <div class="method-card">
                <h4>🧠 LSTM (Deep Learning)</h4>
                <p class="method-name">${lstm.method}</p>
                <div class="method-stats">
                    <div><strong>Architecture:</strong> ${lstm.architecture}</div>
                    <div><strong>Training Accuracy:</strong> ${(lstm.final_accuracy * 100).toFixed(2)}%</div>
                    <div><strong>Validation Accuracy:</strong> ${(lstm.final_val_accuracy * 100).toFixed(2)}%</div>
                    <div><strong>Avg Confidence:</strong> ${(lstm.avg_confidence * 100).toFixed(2)}%</div>
                    <div><strong>Agreement with VADER:</strong> ${(lstm.agreement_with_vader * 100).toFixed(2)}%</div>
                </div>
            </div>
        </div>
    `;
    
    document.getElementById('sentimentDetails').innerHTML = html;
}

function displayKeyPhrases(phrases) {
    const positiveHtml = phrases.most_positive_phrases.map((phrase, idx) => `
        <div class="phrase-card positive-phrase">
            <div class="phrase-rank">#${idx + 1}</div>
            <div class="phrase-text">${phrase.text}</div>
            <div class="phrase-score">Score: ${phrase.score.toFixed(3)}</div>
        </div>
    `).join('');
    
    const negativeHtml = phrases.most_negative_phrases.map((phrase, idx) => `
        <div class="phrase-card negative-phrase">
            <div class="phrase-rank">#${idx + 1}</div>
            <div class="phrase-text">${phrase.text}</div>
            <div class="phrase-score">Score: ${phrase.score.toFixed(3)}</div>
        </div>
    `).join('');
    
    document.getElementById('positivePhrases').innerHTML = positiveHtml;
    document.getElementById('negativePhrases').innerHTML = negativeHtml;
}

function updateOverview() {
    if (currentData.cleaned) {
        document.getElementById('totalReviews').textContent = currentData.cleaned.total;
    }
    
    if (currentData.cleaned && currentData.cleaned.data) {
        const languages = new Set(currentData.cleaned.data.map(r => r.language));
        document.getElementById('languageCount').textContent = languages.size;
    }
    
    if (currentData.phase2 && currentData.phase2.lsa_topics) {
        document.getElementById('topicCount').textContent = currentData.phase2.lsa_topics.topics.length;
    }
    
    if (currentData.phase2 && currentData.phase2.bow_tfidf) {
        document.getElementById('vocabSize').textContent = currentData.phase2.bow_tfidf.vocab_size;
    }
}

// Phase 3 Display Functions
function displayPhase3Results() {
    if (!currentData.phase3) return;
    
    const data = currentData.phase3;
    
    // Display Summarization Stats
    if (data.review_summarization) {
        const summ = data.review_summarization;
        const statsHtml = `
            <div class="stats-summary">
                <div class="stat-item">
                    <strong>Method:</strong> ${summ.method}
                </div>
                <div class="stat-item">
                    <strong>Total Reviews Analyzed:</strong> ${summ.total_reviews_analyzed}
                </div>
                <div class="stat-item">
                    <strong>Clusters Identified:</strong> ${summ.num_clusters}
                </div>
                <div class="stat-item">
                    <strong>Avg Similarity Score:</strong> ${summ.avg_similarity_score.toFixed(3)}
                </div>
            </div>
        `;
        document.getElementById('summarizationStats').innerHTML = statsHtml;
        
        // Display Representative Reviews
        const reviewsHtml = summ.representative_reviews.map((review, idx) => `
            <div class="representative-review-card">
                <div class="review-header">
                    <span class="review-badge">Cluster ${review.cluster_id + 1}</span>
                    <span class="cluster-size">Represents ${review.cluster_size} similar reviews</span>
                </div>
                <div class="review-content">
                    <p><strong>Original Review:</strong></p>
                    <p class="review-text">${review.review_original}</p>
                </div>
                <div class="review-meta">
                    <span>Similarity Score: ${review.similarity_score.toFixed(3)}</span>
                </div>
            </div>
        `).join('');
        document.getElementById('representativeReviews').innerHTML = reviewsHtml;
        
        // Display Cluster Distribution Chart
        const clusterData = summ.cluster_distribution;
        const ctx = document.getElementById('clusterChart').getContext('2d');
        new Chart(ctx, {
            type: 'bar',
            data: {
                labels: Object.keys(clusterData).map(k => `Cluster ${parseInt(k) + 1}`),
                datasets: [{
                    label: 'Number of Reviews',
                    data: Object.values(clusterData),
                    backgroundColor: 'rgba(79, 70, 229, 0.7)',
                    borderColor: 'rgba(79, 70, 229, 1)',
                    borderWidth: 2
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false }
                },
                scales: {
                    y: { beginAtZero: true }
                }
            }
        });
    }
    
    // Display Q&A Pairs
    if (data.question_answering) {
        const qa = data.question_answering;
        const qaHtml = `
            <div class="qa-method">
                <strong>Method:</strong> ${qa.method}
            </div>
            <div class="qa-grid">
                ${qa.qa_pairs.map((pair, idx) => `
                    <div class="qa-card">
                        <div class="question">
                            <span class="q-icon">❓</span>
                            <span class="q-text">${pair.question}</span>
                        </div>
                        <div class="answer">
                            <span class="a-icon">💡</span>
                            <span class="a-text">${pair.answer}</span>
                        </div>
                        <div class="supporting-data">
                            <strong>Supporting Data:</strong>
                            <ul>
                                ${Object.entries(pair.supporting_data).map(([key, value]) => {
                                    if (typeof value === 'string' && value.length > 0) {
                                        return `<li><strong>${key}:</strong> ${value}</li>`;
                                    } else if (typeof value === 'number') {
                                        return `<li><strong>${key}:</strong> ${value}</li>`;
                                    }
                                    return '';
                                }).join('')}
                            </ul>
                        </div>
                    </div>
                `).join('')}
            </div>
        `;
        document.getElementById('qaContainer').innerHTML = qaHtml;
    }
}

// View All Data
async function viewAllData() {
    showTab('data');
}

async function loadSelectedData() {
    const select = document.getElementById('dataTypeSelect');
    const type = select.value;
    
    if (!type) return;
    
    showLoading('Loading data...');
    
    try {
        const data = await apiCall(`/data/${type}`);
        displayDataTable(data.data);
    } catch (error) {
        alert('Failed to load data');
    } finally {
        hideLoading();
    }
}

function displayDataTable(data) {
    if (!data || data.length === 0) {
        document.getElementById('dataTableContainer').innerHTML = '<p>No data available</p>';
        return;
    }
    
    const columns = Object.keys(data[0]);
    
    let html = `
        <table class="data-table">
            <thead>
                <tr>${columns.map(col => `<th>${col}</th>`).join('')}</tr>
            </thead>
            <tbody>
                ${data.map(row => `
                    <tr>${columns.map(col => `<td>${row[col] || ''}</td>`).join('')}</tr>
                `).join('')}
            </tbody>
        </table>
    `;
    
    document.getElementById('dataTableContainer').innerHTML = html;
}

// Chart Functions
function createPieChart(canvasId, title, data) {
    const ctx = document.getElementById(canvasId);
    if (!ctx) return;
    
    if (charts[canvasId]) {
        charts[canvasId].destroy();
    }
    
    charts[canvasId] = new Chart(ctx, {
        type: 'pie',
        data: {
            labels: Object.keys(data),
            datasets: [{
                data: Object.values(data),
                backgroundColor: [
                    '#4f46e5', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6',
                    '#ec4899', '#14b8a6', '#f97316', '#06b6d4', '#84cc16'
                ]
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: {
                    display: true,
                    text: title,
                    font: { size: 16 }
                },
                legend: {
                    position: 'right'
                }
            }
        }
    });
}

function createBarChart(canvasId, title, labels, data) {
    const ctx = document.getElementById(canvasId);
    if (!ctx) return;
    
    if (charts[canvasId]) {
        charts[canvasId].destroy();
    }
    
    charts[canvasId] = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Count',
                data: data,
                backgroundColor: '#4f46e5'
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: {
                    display: true,
                    text: title,
                    font: { size: 16 }
                },
                legend: {
                    display: false
                }
            },
            scales: {
                y: {
                    beginAtZero: true
                }
            }
        }
    });
}

function createHorizontalBarChart(canvasId, title, labels, data) {
    const ctx = document.getElementById(canvasId);
    if (!ctx) return;
    
    if (charts[canvasId]) {
        charts[canvasId].destroy();
    }
    
    charts[canvasId] = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Frequency',
                data: data,
                backgroundColor: '#10b981'
            }]
        },
        options: {
            indexAxis: 'y',
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: {
                    display: true,
                    text: title,
                    font: { size: 16 }
                },
                legend: {
                    display: false
                }
            },
            scales: {
                x: {
                    beginAtZero: true
                }
            }
        }
    });
}

// UI Functions
function showTab(tabName) {
    // Hide all tabs
    document.querySelectorAll('.tab-pane').forEach(pane => {
        pane.classList.remove('active');
    });
    
    // Remove active from all buttons
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.classList.remove('active');
    });
    
    // Show selected tab
    document.getElementById(`${tabName}Tab`).classList.add('active');
    
    // Activate button
    event.target.classList.add('active');
    
    // Show tabs container
    document.getElementById('tabsContainer').style.display = 'flex';
}

function showLoading(text) {
    const loadingText = document.getElementById('loadingText');
    const loading = document.getElementById('loading');
    
    if (loadingText) {
        loadingText.textContent = text;
    }
    if (loading) {
        loading.style.display = 'block';
    }
}

function hideLoading() {
    const loading = document.getElementById('loading');
    if (loading) {
        loading.style.display = 'none';
    }
}

function calculateAvgLength(data) {
    const total = data.reduce((sum, row) => {
        const words = (row.Processed_Review || '').split(' ').length;
        return sum + words;
    }, 0);
    return Math.round(total / data.length);
}

function exportData() {
    const select = document.getElementById('dataTypeSelect');
    const type = select.value;
    
    if (!type) {
        alert('Please select a data type first');
        return;
    }
    
    window.open(`${API_BASE}/data/${type}`, '_blank');
}
