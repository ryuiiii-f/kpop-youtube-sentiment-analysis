import pandas as pd
import numpy as np
from collections import Counter, defaultdict
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# 读取数据
df = pd.read_csv("labeled_comments_hylt_hf.csv")

print("🔍 ANALYZING FAN TYPE CLASSIFICATION ISSUES")
print("="*60)

# 1. 分析unclear的评论，找出模式
unclear_comments = df[df['fan_type'] == 'unclear']['comment_text'].tolist()
print(f"📊 Unclear comments: {len(unclear_comments)} / {len(df)} ({len(unclear_comments)/len(df)*100:.1f}%)")

# ============================================
# 策略1: 从数据中发现关键词模式
# ============================================

def extract_keywords_from_comments(comments, top_n=20):
    """从评论中提取高频关键词"""
    # 清理和分词
    text = ' '.join([str(comment).lower() for comment in comments])
    text = re.sub(r'[^\w\s]', ' ', text)
    
    # 移除停用词
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them', 'my', 'your', 'his', 'her', 'its', 'our', 'their', 'so', 'very', 'really', 'just', 'too', 'now', 'here', 'there', 'when', 'where', 'how', 'what', 'who', 'which', 'why', 'all', 'some', 'any', 'no', 'not', 'only', 'also', 'even', 'more', 'most', 'much', 'many', 'few', 'little', 'good', 'bad', 'nice', 'great', 'best', 'better', 'well', 'like', 'love', 'know', 'think', 'see', 'get', 'go', 'come', 'want', 'need', 'make', 'take', 'give', 'say', 'tell', 'ask'}
    
    words = text.split()
    filtered_words = [word for word in words if len(word) > 2 and word not in stop_words]
    
    return Counter(filtered_words).most_common(top_n)

# 为每个成功标注的类型提取关键词
print("\n🔍 DISCOVERING PATTERNS FROM SUCCESSFULLY LABELED COMMENTS")
print("-" * 60)

for fan_type in ['lyrics-lover', 'visual-lover', 'dance-lover', 'personality-lover', 'music-lover']:
    if fan_type in df['fan_type'].values:
        type_comments = df[df['fan_type'] == fan_type]['comment_text'].tolist()
        if len(type_comments) > 0:
            keywords = extract_keywords_from_comments(type_comments, 15)
            print(f"\n💎 {fan_type.upper()} keywords:")
            print("   ", [f"{word}({count})" for word, count in keywords[:10]])

# ============================================
# 策略2: 改进的规则引擎
# ============================================

def improved_fan_type_classifier(comment):
    """改进的基于规则的分类器"""
    if pd.isna(comment):
        return "unclear"
    
    text = str(comment).lower()
    
    # 更精细的关键词规则
    rules = [
        # Visual-lover 规则 (最容易识别)
        {
            'type': 'visual-lover',
            'strong_indicators': ['handsome', 'beautiful', 'pretty', 'gorgeous', 'hot', 'sexy', 'cute', 'visual', 'look', 'face', 'hair', 'outfit', 'style', 'fashion', 'makeup', '帅', '美', '好看', '颜值', '造型', '외모', '잘생겼다', 'かっこいい', '美しい'],
            'weight': 3
        },
        
        # Dance-lover 规则
        {
            'type': 'dance-lover', 
            'strong_indicators': ['dance', 'dancing', 'choreography', 'choreo', 'moves', 'move', 'performance', 'perform', 'stage', 'rhythm', 'beat', '舞蹈', '编舞', '动作', '表演', '댄스', '안무', 'ダンス', '踊り'],
            'weight': 3
        },
        
        # Music-lover 规则  
        {
            'type': 'music-lover',
            'strong_indicators': ['beat', 'melody', 'music', 'sound', 'production', 'instrumental', 'vocals', 'voice', 'singing', 'audio', 'quality', 'mix', '音乐', '编曲', '旋律', '声音', '음악', '멜로디', '音楽', 'メロディー'],
            'weight': 3
        },
        
        # Lyrics-lover 规则
        {
            'type': 'lyrics-lover',
            'strong_indicators': ['lyrics', 'words', 'meaning', 'message', 'deep', 'story', 'relate', 'understand', 'poetry', 'verse', 'line', '歌词', '意思', '含义', '故事', '가사', '의미', '歌詞', '言葉'],
            'weight': 3
        },
        
        # Personality-lover 规则 (最难识别，用组合规则)
        {
            'type': 'personality-lover',
            'strong_indicators': ['funny', 'humor', 'sweet', 'kind', 'real', 'genuine', 'humble', 'personality', 'character', 'person', 'talented', 'skill', '性格', '人品', '搞笑', '善良', '真实', '성격', '인성', '面白い', '優しい'],
            'weight': 2,
            'combo_rules': [
                # 组合规则：提到人+正面词汇
                (['person', 'people', 'human', 'individual'], ['amazing', 'incredible', 'perfect', 'awesome', 'wonderful']),
                # 提到才能+个人特质
                (['talent', 'skill', 'ability', 'gift'], ['natural', 'born', 'amazing', 'incredible'])
            ]
        }
    ]
    
    # 计算每种类型的得分
    type_scores = defaultdict(float)
    
    for rule in rules:
        fan_type = rule['type']
        weight = rule['weight']
        
        # 基础关键词得分
        keyword_score = 0
        for keyword in rule['strong_indicators']:
            if keyword in text:
                keyword_score += 1
        
        # 组合规则得分 (如果有)
        combo_score = 0
        if 'combo_rules' in rule:
            for combo in rule['combo_rules']:
                group1, group2 = combo
                has_group1 = any(word in text for word in group1)
                has_group2 = any(word in text for word in group2)
                if has_group1 and has_group2:
                    combo_score += 2
        
        type_scores[fan_type] = (keyword_score + combo_score) * weight
    
    # 特殊规则：长度过短的评论
    if len(text.strip()) < 10:
        return "unclear"
    
    # 特殊规则：只有表情符号或重复字符
    if re.match(r'^[^\w\s]*$', text) or len(set(text.replace(' ', ''))) < 3:
        return "unclear"
    
    # 返回得分最高的类型
    if type_scores and max(type_scores.values()) > 0:
        return max(type_scores, key=type_scores.get)
    else:
        return "unclear"

# 重新分类unclear的评论
print("\n🔧 RE-CLASSIFYING UNCLEAR COMMENTS WITH IMPROVED RULES")
print("-" * 60)

unclear_mask = df['fan_type'] == 'unclear'
unclear_indices = df[unclear_mask].index

reclassified_count = 0
for idx in unclear_indices:
    comment = df.loc[idx, 'comment_text']
    new_type = improved_fan_type_classifier(comment)
    if new_type != 'unclear':
        df.loc[idx, 'fan_type'] = new_type
        reclassified_count += 1

print(f"📊 Reclassified {reclassified_count} comments from unclear to specific types")

# 新的分布统计
new_fan_type_counts = df["fan_type"].value_counts()
new_unclear_ratio = new_fan_type_counts.get('unclear', 0) / len(df) * 100

print(f"\n📈 IMPROVED RESULTS:")
print(f"   • Unclear ratio: {len(unclear_comments)/len(df)*100:.1f}% → {new_unclear_ratio:.1f}%")
print(f"   • Improvement: {len(unclear_comments)/len(df)*100 - new_unclear_ratio:.1f}% reduction in unclear")

print(f"\n🆕 NEW FAN TYPE DISTRIBUTION:")
for fan_type, count in new_fan_type_counts.items():
    percentage = count/len(df)*100
    print(f"   {fan_type}: {count} ({percentage:.1f}%)")

# ============================================
# 策略3: 文本聚类发现未知模式
# ============================================

print("\n🧠 DISCOVERING HIDDEN PATTERNS WITH TEXT CLUSTERING")
print("-" * 60)

# 对remaining unclear评论进行聚类分析
remaining_unclear = df[df['fan_type'] == 'unclear']['comment_text'].tolist()

if len(remaining_unclear) > 10:
    # TF-IDF向量化
    vectorizer = TfidfVectorizer(
        max_features=100, 
        stop_words='english',
        ngram_range=(1, 2),
        min_df=2
    )
    
    try:
        # 过滤空评论
        valid_comments = [str(comment) for comment in remaining_unclear if str(comment).strip() and len(str(comment)) > 5]
        
        if len(valid_comments) > 5:
            tfidf_matrix = vectorizer.fit_transform(valid_comments)
            
            # K-means聚类
            n_clusters = min(5, len(valid_comments) // 3)  # 动态确定聚类数
            if n_clusters >= 2:
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                clusters = kmeans.fit_predict(tfidf_matrix)
                
                # 分析每个聚类的特征词
                feature_names = vectorizer.get_feature_names_out()
                
                print(f"🔍 Found {n_clusters} hidden patterns in unclear comments:")
                
                for i in range(n_clusters):
                    cluster_comments = [valid_comments[j] for j in range(len(valid_comments)) if clusters[j] == i]
                    cluster_size = len(cluster_comments)
                    
                    if cluster_size > 1:
                        # 获取该聚类的代表性词汇
                        cluster_indices = np.where(clusters == i)[0]
                        cluster_center = kmeans.cluster_centers_[i]
                        top_features = cluster_center.argsort()[-10:][::-1]
                        top_words = [feature_names[idx] for idx in top_features]
                        
                        print(f"\n   Cluster {i+1} ({cluster_size} comments):")
                        print(f"   Key words: {', '.join(top_words[:5])}")
                        print(f"   Sample: {cluster_comments[0][:60]}...")
                        
                        # 基于关键词建议可能的类型
                        if any(word in ' '.join(top_words) for word in ['love', 'heart', 'feel', 'emotion']):
                            print(f"   💡 Suggests: Might be personality-lover or emotional response")
                        elif any(word in ' '.join(top_words) for word in ['song', 'music', 'sound']):
                            print(f"   💡 Suggests: Might be music-lover")
    
    except Exception as e:
        print(f"❌ Clustering analysis failed: {e}")

# ============================================
# 策略4: 手工规则优化建议
# ============================================

print(f"\n💡 MANUAL INSPECTION SUGGESTIONS:")
print("-" * 60)

# 显示一些unclear样本供手工检查
unclear_samples = df[df['fan_type'] == 'unclear'].head(10)
print(f"\n📝 Sample unclear comments for manual review:")

for idx, row in unclear_samples.iterrows():
    comment = str(row['comment_text'])[:80]
    emotion = row['emotion']
    print(f"\n   Comment: {comment}...")
    print(f"   Emotion: {emotion}")
    print(f"   💭 Manual suggestion: _____ (你可以手工判断)")

# 保存改进后的结果
df.to_csv("labeled_comments_hylt_improved.csv", index=False, encoding="utf-8-sig")

print(f"\n💾 SAVED IMPROVED RESULTS:")
print(f"   ✅ labeled_comments_hylt_improved.csv")

print(f"\n🎯 RECOMMENDATIONS FOR NEXT ITERATION:")
print(f"   1. 📝 Manually review top unclear samples to build better rules")
print(f"   2. 🔄 Add discovered keywords to the classifier") 
print(f"   3. 📊 Test on comments from different K-pop songs")
print(f"   4. 🤖 Consider using a different approach for personality-lover detection")
print(f"   5. 📈 Focus on the classifications that ARE working well")

# 生成改进建议报告
print(f"\n📋 CLASSIFICATION PERFORMANCE REPORT:")
print(f"   • Best performing types: {new_fan_type_counts.head(3).to_dict()}")
print(f"   • Most challenging type: Likely 'personality-lover' (too subjective)")
print(f"   • Recommended focus: Visual, Dance, Music (easier to identify)")