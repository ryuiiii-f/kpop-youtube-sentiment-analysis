#  K-pop YouTube Sentiment Analysis

Cross-cultural sentiment analysis of K-pop YouTube comments using AI and traditional NLP methods to understand global fan engagement patterns.

##  Key Findings

**NewJeans "Ditto" dominates engagement depth:**
- **905 comments analyzed** across 5 languages  
- **20.6% MV analysis** comments (vs typical 5-10% for K-pop)
- **Cross-cultural insight**: Korean fans 90% analytical, Global fans 43% emotional

**BLACKPINK "How You Like That":**
- **English-dominated** global fanbase analysis
- **Traditional NLP + rule-based sentiment classification**

## 📊 Project Comparison

| Project | Comments | Languages | Deep Analysis | Unique Finding |
|---------|----------|-----------|---------------|----------------|
| **Ditto** | 905 | 5 | 20.6% | 1.8% "melancholy-positive" emotion |
| **HYLT** | ~500 | 3 | 15% | English-dominated global response |

## 🔧 Tech Stack

| Component | Technologies |
|-----------|-------------|
| **Data Collection** | YouTube Data API, Custom scrapers |
| **Text Processing** | Pandas, NLTK, Multi-language preprocessing |
| **Sentiment Analysis** | VADER, Custom emotion lexicons |
| **Machine Learning** | Scikit-learn (K-means, PCA, t-SNE) |
| **Statistical Analysis** | TF-IDF, Correlation analysis |
| **Visualization** | Plotly, Matplotlib, Wordcloud |
| **Dashboard** | Interactive HTML/CSS/JavaScript |

## 📊 Visual Analysis

![Ditto Dashboard](images/ditto_dashboard_preview.png)
*Interactive cross-cultural analysis dashboard with 10+ visualizations*

![Emotion Wordcloud](images/ditto_emotion_wordcloud.png)
*Multi-language emotion keywords showing "nostalgic," "theory," "masterpiece"*

![Cross-Cultural Comparison](images/ditto_cross_cultural_analysis.png)
*Korean vs Global fan response patterns and engagement types*

## 📁 Files

- `youtube_comment_collector.py` - YouTube API data collection
- `hg_emotion_labeller.py` - Traditional sentiment analysis  
- `ditto_cross_cultural_analysis.py` - Cross-cultural pattern analysis
- `ditto_dashboard.html` - Interactive analysis dashboard
- Various processed datasets with sentiment labels

## 🎮 Interactive Dashboards

**NewJeans "Ditto"**: [Cross-Cultural Analysis Dashboard](results/ditto_dashboard.html)
- 5-language sentiment analysis
- Fan engagement type breakdowns  
- Cultural pattern visualization

**BLACKPINK**: Static visualizations with emotion distribution analysis

## 🔍 Methodology

**5-Phase Analysis Pipeline:**
1. **Baseline**: VADER + Rules (70% accuracy)
2. **AI**: HuggingFace Transformers (85-90% accuracy)  
3. **Hybrid**: Rule-based
4. **Discovery**: Unsupervised pattern identification
5. **Validation**: Cross-method comparison
