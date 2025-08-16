import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import re
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# 🎯 加载清洗后的数据
df = pd.read_csv("..\data_collection\clean_comments_ditto_side_a.csv")

print(f"📊 Analyzing {len(df)} cleaned comments")
print("=" * 50)

# ================================
# 💰 PHASE 1: RULE-BASED SENTIMENT (FREE)
# ================================

def ditto_specific_sentiment(text):
    """
    针对Ditto和K-pop优化的规则情感分析
    避免Hugging Face的unclear问题
    """
    text_lower = str(text).lower()
    
    # 正面情感词汇（Ditto特化）
    positive_patterns = {
        'love': ['love', 'amazing', 'perfect', 'beautiful', 'gorgeous', 'talent', 'queen'],
        'nostalgia': ['nostalgia', 'nostalgic', 'memory', 'childhood', 'youth', 'miss'],
        'winter_mood': ['winter', 'cozy', 'warm', 'comfort', 'peaceful', 'calm'],
        'mv_praise': ['genius', 'masterpiece', 'art', 'cinematic', 'story', 'concept'],
        'korean_positive': ['최고', '완벽', '아름다워', '사랑', '감동']
    }
    
    # 负面情感词汇
    negative_patterns = {
        'sadness': ['sad', 'crying', 'tears', 'heartbreak', 'lonely', 'depressed'],
        'confusion': ['confused', 'weird', 'strange', 'understand', "don't get"],
        'criticism': ['boring', 'bad', 'worst', 'hate', 'terrible'],
        'korean_negative': ['싫어', '별로', '이상해', '지루해']
    }
    
    # 特殊情感：Ditto特有的melancholy（忧郁但不完全负面）
    melancholy_patterns = ['bittersweet', 'melancholy', 'emotional', 'deep', 'thoughtful', 'complex']
    
    pos_score = sum([len(re.findall(word, text_lower)) for category in positive_patterns.values() for word in category])
    neg_score = sum([len(re.findall(word, text_lower)) for category in negative_patterns.values() for word in category])
    mel_score = sum([len(re.findall(word, text_lower)) for word in melancholy_patterns])
    
    # 情感分类逻辑
    if mel_score > 0 and pos_score > neg_score:
        return "melancholy_positive"  # Ditto特有：忧郁但正面
    elif pos_score > neg_score * 1.5:
        return "positive"
    elif neg_score > pos_score * 1.5:
        return "negative"
    elif mel_score > 0:
        return "melancholy"
    else:
        return "neutral"

# 应用规则情感分析
print("🎭 Applying rule-based sentiment analysis...")
df['sentiment'] = df['comment_text'].apply(ditto_specific_sentiment)

# 情感分布
sentiment_dist = df['sentiment'].value_counts()
print("\n📊 Sentiment Distribution:")
for sentiment, count in sentiment_dist.items():
    percentage = count/len(df)*100
    print(f"   {sentiment}: {count} ({percentage:.1f}%)")

# ================================
# 🔍 PHASE 2: CLUSTERING ANALYSIS (FREE)
# ================================

print("\n🔮 Performing clustering analysis...")

# 准备文本数据（只分析高质量评论节省计算）
high_quality_df = df[df['quality_score'] > 40].copy()
print(f"🎯 Analyzing {len(high_quality_df)} high-quality comments for clustering")

# 文本预处理
def preprocess_for_clustering(text):
    # 保留英文、韩文、常见符号
    text = re.sub(r'[^\w\s가-힣]', ' ', str(text).lower())
    # 移除过短词汇
    words = [word for word in text.split() if len(word) > 2]
    return ' '.join(words)

high_quality_df['processed_text'] = high_quality_df['comment_text'].apply(preprocess_for_clustering)

# TF-IDF向量化（多语言友好）
vectorizer = TfidfVectorizer(
    max_features=100,  # 控制特征数量
    min_df=2,          # 词汇至少出现2次
    max_df=0.8,        # 忽略过于常见的词
    ngram_range=(1, 2), # 1-2词组合
    stop_words=None    # 不使用英文停用词（会删除韩文）
)

# 向量化
tfidf_matrix = vectorizer.fit_transform(high_quality_df['processed_text'])
feature_names = vectorizer.get_feature_names_out()

# K-means聚类
n_clusters = 5  # 基于你的comment_type分布
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
clusters = kmeans.fit_predict(tfidf_matrix)

high_quality_df['cluster'] = clusters

# ================================
# 📈 PHASE 3: RESULTS VISUALIZATION
# ================================

# 1. 情感 x 评论类型交叉分析
print("\n📊 Sentiment by Comment Type:")
cross_tab = pd.crosstab(df['comment_type'], df['sentiment'], normalize='index') * 100
print(cross_tab.round(1))

# 2. 聚类结果分析
print(f"\n🔮 Cluster Analysis Results:")
for cluster_id in range(n_clusters):
    cluster_data = high_quality_df[high_quality_df['cluster'] == cluster_id]
    print(f"\n📍 Cluster {cluster_id} ({len(cluster_data)} comments):")
    
    # 获取聚类的关键词
    cluster_indices = np.where(clusters == cluster_id)[0]
    cluster_tfidf = tfidf_matrix[cluster_indices].mean(axis=0).A1
    top_features_idx = cluster_tfidf.argsort()[-10:][::-1]
    top_keywords = [feature_names[i] for i in top_features_idx]
    print(f"   Key words: {', '.join(top_keywords[:5])}")
    
    # 主要情感和类型
    main_sentiment = cluster_data['sentiment'].mode().iloc[0] if len(cluster_data) > 0 else "N/A"
    main_type = cluster_data['comment_type'].mode().iloc[0] if len(cluster_data) > 0 else "N/A"
    print(f"   Main sentiment: {main_sentiment}")
    print(f"   Main type: {main_type}")
    
    # 代表性评论
    representative = cluster_data.nlargest(1, 'quality_score')['comment_text'].iloc[0]
    print(f"   Sample: {representative[:100]}...")

# ================================
# 💎 PHASE 4: KEY INSIGHTS EXTRACTION
# ================================

print("\n" + "="*50)
print("🔍 KEY INSIGHTS")
print("="*50)

# MV Analysis情感倾向
mv_analysis_sentiment = df[df['comment_type'] == 'mv_analysis']['sentiment'].value_counts()
print("🎬 MV Analysis Comments Sentiment:")
for sentiment, count in mv_analysis_sentiment.head(3).items():
    percentage = count/len(df[df['comment_type'] == 'mv_analysis'])*100
    print(f"   {sentiment}: {percentage:.1f}%")

# 语言与情感关系
if 'detected_language' in df.columns:
    print(f"\n🌍 Language vs Sentiment Patterns:")
    lang_sentiment = df.groupby('detected_language')['sentiment'].apply(lambda x: x.value_counts(normalize=True).head(2))
    print(lang_sentiment)

# 高质量评论的特征
high_quality_characteristics = df[df['quality_score'] > 60].groupby('comment_type').size().sort_values(ascending=False)
print(f"\n⭐ High Quality Comments (score>60) by Type:")
for ctype, count in high_quality_characteristics.head(3).items():
    print(f"   {ctype}: {count}")

# ================================
# 💡 ACTIONABLE RECOMMENDATIONS
# ================================

print("\n💡 RECOMMENDATIONS FOR DEEPER ANALYSIS:")

melancholy_positive_count = len(df[df['sentiment'] == 'melancholy_positive'])
if melancholy_positive_count > 20:
    print(f"🎯 Focus Area 1: {melancholy_positive_count} 'melancholy_positive' comments - unique Ditto emotion")

mv_analysis_count = len(df[df['comment_type'] == 'mv_analysis'])
if mv_analysis_count > 50:
    print(f"🎯 Focus Area 2: {mv_analysis_count} MV analysis comments - theory extraction potential")

korean_comments = len(df[df.get('detected_language', '') == 'ko'])
if korean_comments > 100:
    print(f"🎯 Focus Area 3: {korean_comments} Korean comments - cultural perspective analysis")

print(f"\n💰 Cost-Effective Next Step: Use GPT on top {min(50, mv_analysis_count)} MV analysis comments only")
print("   Expected cost: ~$2-5 for high-value insights")

# 保存分析结果
df.to_csv("ditto_sentiment_analysis_results.csv", index=False, encoding="utf-8-sig")
print(f"\n💾 Results saved to: ditto_sentiment_analysis_results.csv")
