import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re

print("🧠 UNSUPERVISED DISCOVERY OF REAL FAN PATTERNS")
print("="*60)

# 读取数据
df = pd.read_csv("labeled_comments_hylt_hf.csv")

# 1. 清理和预处理
def clean_comment(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    # 保留字母、数字、空格，移除特殊符号
    text = re.sub(r'[^a-zA-Z0-9\s가-힣ひらがなカタカナ一-龯]', ' ', text)
    # 压缩多个空格
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# 清理评论
df['comment_clean'] = df['comment_text'].apply(clean_comment)

# 过滤太短的评论（这些通常就是general praise）
df_filtered = df[df['comment_clean'].str.len() > 10].copy()
print(f"📊 Filtered comments: {len(df_filtered)} / {len(df)} (removed short comments)")

# 2. TF-IDF向量化
print("\n🔧 Creating TF-IDF vectors...")
vectorizer = TfidfVectorizer(
    max_features=200,  # 限制特征数
    ngram_range=(1, 2),  # 包含单词和双词组合
    min_df=3,  # 词汇至少出现3次
    max_df=0.8,  # 排除过于常见的词
    stop_words=None  # 不使用停用词（因为多语言）
)

tfidf_matrix = vectorizer.fit_transform(df_filtered['comment_clean'])
feature_names = vectorizer.get_feature_names_out()

print(f"✅ TF-IDF matrix shape: {tfidf_matrix.shape}")

# 定义聚类解释函数
def interpret_cluster(keywords, sample_comments):
    """基于关键词和样本评论解释聚类含义"""
    keywords_str = ' '.join(keywords).lower()
    
    # Visual相关
    if any(word in keywords_str for word in ['look', 'visual', 'beautiful', 'pretty', 'hair', 'style', 'makeup']):
        return "Visual-Appearance Focused"
    
    # Music/Performance相关
    elif any(word in keywords_str for word in ['music', 'song', 'dance', 'performance', 'beat', 'vocals']):
        return "Content-Performance Focused"
    
    # 歌曲标题/歌词相关
    elif any(word in keywords_str for word in ['how you like that', 'like that', 'how you', 'lyrics', 'line']):
        return "Song-Title/Lyrics Focused"
    
    # 情感表达
    elif any(word in keywords_str for word in ['love', 'amazing', 'perfect', 'best', 'good', 'great']):
        return "Emotional-Praise Focused"
    
    # 互动/社区相关
    elif any(word in keywords_str for word in ['blink', 'fan', 'stream', 'support', 'forever']):
        return "Community-Support Focused"
    
    # 其他
    else:
        return "Mixed-General Content"

# 3. 尝试不同的聚类数量
print("\n🔍 Testing different cluster numbers...")

inertias = []
K_range = range(2, 8)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(tfidf_matrix)
    inertias.append(kmeans.inertia_)

# 绘制肘部法则图
plt.figure(figsize=(10, 6))
plt.plot(K_range, inertias, 'bo-')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal Cluster Number')
plt.grid(True, alpha=0.3)
plt.savefig('clustering_elbow_method.png', dpi=300, bbox_inches='tight')
plt.show()
print("✅ Saved: clustering_elbow_method.png")

# 4. 使用最优聚类数（假设是3-4）
optimal_k = 4  # 可以根据肘部图调整
print(f"\n🎯 Performing clustering with k={optimal_k}...")

kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(tfidf_matrix)

# 将聚类结果加入数据框
df_filtered['cluster'] = cluster_labels

# 5. 分析每个聚类的特征
print(f"\n🔍 ANALYZING DISCOVERED CLUSTERS:")
print("-" * 50)

cluster_insights = {}
for i in range(optimal_k):
    cluster_mask = cluster_labels == i
    cluster_size = np.sum(cluster_mask)
    
    if cluster_size > 5:  # 只分析有足够样本的聚类
        # 获取该聚类的代表性词汇
        cluster_center = kmeans.cluster_centers_[i]
        top_indices = cluster_center.argsort()[-15:][::-1]
        top_words = [feature_names[idx] for idx in top_indices]
        
        # 获取该聚类的样本评论
        cluster_comments = df_filtered[df_filtered['cluster'] == i]['comment_text'].tolist()
        cluster_emotions = df_filtered[df_filtered['cluster'] == i]['emotion'].value_counts()
        
        print(f"\n🔸 CLUSTER {i+1} ({cluster_size} comments, {cluster_size/len(df_filtered)*100:.1f}%):")
        print(f"   Key words: {', '.join(top_words[:8])}")
        print(f"   Top emotions: {dict(cluster_emotions.head(3))}")
        print(f"   Sample comments:")
        for j, comment in enumerate(cluster_comments[:2]):
            print(f"     '{comment[:60]}...'")
        
        # 尝试解释这个聚类
        interpretation = interpret_cluster(top_words, cluster_comments[:5])
        print(f"   💡 Possible interpretation: {interpretation}")
        
        cluster_insights[f"cluster_{i+1}"] = {
            'size': cluster_size,
            'percentage': cluster_size/len(df_filtered)*100,
            'keywords': top_words[:8],
            'interpretation': interpretation,
            'top_emotions': cluster_emotions.head(3).to_dict()
        }

# 6. 可视化聚类结果
print(f"\n📊 Creating cluster visualization...")

# PCA降维可视化
pca = PCA(n_components=2, random_state=42)
tfidf_2d = pca.fit_transform(tfidf_matrix.toarray())

plt.figure(figsize=(12, 8))
colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']

for i in range(optimal_k):
    cluster_points = tfidf_2d[cluster_labels == i]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], 
               c=colors[i], label=f'Cluster {i+1}', alpha=0.6)

plt.title('K-pop Comment Clusters (PCA Visualization)', fontsize=16, fontweight='bold')
plt.xlabel(f'First Principal Component ({pca.explained_variance_ratio_[0]:.1%} variance)')
plt.ylabel(f'Second Principal Component ({pca.explained_variance_ratio_[1]:.1%} variance)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('comment_clusters_pca.png', dpi=300, bbox_inches='tight')
plt.show()
print("✅ Saved: comment_clusters_pca.png")

# 7. 创建新的分类系统
print(f"\n🎯 CREATING DATA-DRIVEN CLASSIFICATION SYSTEM:")
print("-" * 50)

# 基于发现的聚类创建新的fan_type
def assign_data_driven_type(cluster_num, cluster_insights):
    """基于聚类分析结果分配类型"""
    if f"cluster_{cluster_num+1}" in cluster_insights:
        interpretation = cluster_insights[f"cluster_{cluster_num+1}"]['interpretation']
        
        if "Visual" in interpretation:
            return "visual-focused"
        elif "Content" in interpretation or "Performance" in interpretation:
            return "content-focused"
        elif "Song-Title" in interpretation or "Lyrics" in interpretation:
            return "song-reference-focused"  # 新发现的类型！
        elif "Community" in interpretation:
            return "community-focused"  # 新类型！
        else:
            return "general-praise"
    return "general-praise"

# 为过滤后的数据分配新类型
df_filtered['fan_type_discovered'] = df_filtered['cluster'].apply(
    lambda x: assign_data_driven_type(x, cluster_insights)
)

# 统计新的分类结果
discovered_counts = df_filtered['fan_type_discovered'].value_counts()
print(f"\n📊 DATA-DRIVEN CLASSIFICATION RESULTS:")
for category, count in discovered_counts.items():
    percentage = count/len(df_filtered)*100
    print(f"   {category}: {count} ({percentage:.1f}%)")

# 保存发现的分类结果
df_filtered[['comment_text', 'emotion', 'fan_type_discovered', 'cluster']].to_csv(
    "labeled_comments_discovered_types.csv", index=False, encoding="utf-8-sig"
)

print(f"\n💾 SAVED DATA-DRIVEN RESULTS:")
print(f"   ✅ labeled_comments_discovered_types.csv")
print(f"   ✅ clustering_elbow_method.png")
print(f"   ✅ comment_clusters_pca.png")

print(f"\n🤔 ANALYSIS CONCLUSION:")
if len(discovered_counts) > 1 and discovered_counts.iloc[0] / len(df_filtered) < 0.8:
    print(f"   ✅ Found meaningful patterns in the data!")
    print(f"   🎯 Use these discovered types for further analysis")
else:
    print(f"   📝 Comments are naturally homogeneous (mostly general praise)")
    print(f"   💡 This is a valid finding - most K-pop fans express general positive emotions")
    print(f"   🚀 Recommend focusing on emotion analysis instead of fan types")

print(f"\n🎵 READY FOR SPOTIFY AUDIO FEATURES ANALYSIS!")
