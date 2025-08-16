import pandas as pd
import re
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 尝试导入语言检测库（如果没有请pip install langdetect）
try:
    from langdetect import detect, DetectorFactory
    DetectorFactory.seed = 0  # 确保结果一致性
    LANG_DETECT_AVAILABLE = True
except ImportError:
    print("⚠️ langdetect not installed. Language detection will be skipped.")
    print("Install with: pip install langdetect")
    LANG_DETECT_AVAILABLE = False

# 🎯 加载Ditto评论数据
input_file = "newjeans_ditto_ditto_side_a_comments.csv"  # 根据你的文件名调整
df = pd.read_csv(input_file)

print(f"📊 Original dataset: {len(df)} comments")
print(f"📁 Source file: {input_file}")
print("-" * 50)

def detect_language(text):
    """检测文本语言"""
    if not LANG_DETECT_AVAILABLE:
        return "unknown"
    
    try:
        # 去掉过多的表情符号和符号
        clean_text = re.sub(r'[^\w\s가-힣ひらがなカタカナ一-龯]', ' ', str(text))
        if len(clean_text.strip()) < 3:
            return "emoji_only"
        return detect(clean_text)
    except:
        return "unknown"

def classify_comment_type(text):
    """基于内容分类评论类型（针对Ditto特色）"""
    text_lower = str(text).lower()
    
    # MV解读关键词
    mv_keywords = [
        'side a', 'side b', 'heesoo', 'theory', 'meaning', 'explain', 
        'story', 'mv', 'video', 'concept', 'interpretation', 'analysis',
        '이론', '의미', '스토리', '해석', '분석'
    ]
    
    # 情感共鸣关键词  
    emotion_keywords = [
        'nostalgia', 'winter', 'memory', 'feeling', 'mood', 'vibe',
        'relate', 'sad', 'beautiful', 'emotional', 'tears',
        '감정', '기억', '겨울', '추억', '느낌'
    ]
    
    # 音乐技术关键词
    music_keywords = [
        'vocal', 'voice', 'harmony', 'production', 'beat', 'sound',
        'mixing', 'melody', 'instrumental', 'choreography',
        '보컬', '목소리', '음악', '안무'
    ]
    
    # 纯应援关键词
    stan_keywords = [
        'queen', 'talent', 'visual', 'love', 'stan', 'amazing', 
        'perfect', 'best', 'bias', 'ot5',
        '최고', '사랑', '완벽', '여왕'
    ]
    
    if any(keyword in text_lower for keyword in mv_keywords):
        return "mv_analysis"
    elif any(keyword in text_lower for keyword in emotion_keywords):
        return "emotional_response" 
    elif any(keyword in text_lower for keyword in music_keywords):
        return "music_technical"
    elif any(keyword in text_lower for keyword in stan_keywords):
        return "fan_support"
    else:
        return "general"

def is_valid_ditto_comment(text):
    """改进的评论有效性检查（适配多语言）"""
    if pd.isna(text):
        return False
    
    text = str(text).strip()
    
    # 过短的评论
    if len(text) < 2:
        return False
    
    # 纯表情符号或符号的评论
    clean_text = re.sub(r'[^\w\s가-힣ひらがなカタカナ一-龯]', '', text)
    if len(clean_text.strip()) < 2:
        return False
        
    # 过滤明显的垃圾评论
    spam_patterns = [
        r'^.{1,3}$',  # 过短
        r'^[!@#$%^&*()_+={}[\]|\\:";\'<>?,.\/~`\-]*$',  # 纯符号
        r'^(\w)\1{10,}',  # 重复字符
    ]
    
    for pattern in spam_patterns:
        if re.match(pattern, text):
            return False
    
    return True

def advanced_deduplication(df):
    """高级去重：考虑相似评论和多语言"""
    
    # 1. 完全重复去重
    original_len = len(df)
    df = df.drop_duplicates(subset=["comment_text"])
    print(f"🔄 Exact duplicates removed: {original_len - len(df)}")
    
    # 2. 标准化后去重（去除空格、标点差异）
    def normalize_text(text):
        # 移除多余空格、标点，转小写
        normalized = re.sub(r'[^\w\s가-힣ひらがなカタカナ一-龯]', '', str(text).lower())
        return ' '.join(normalized.split())
    
    df['normalized_text'] = df['comment_text'].apply(normalize_text)
    before_norm = len(df)
    df = df.drop_duplicates(subset=['normalized_text'])
    print(f"🔄 Normalized duplicates removed: {before_norm - len(df)}")
    
    # 清理临时列
    df = df.drop('normalized_text', axis=1)
    
    return df

# 🧹 数据清洗流程
print("🧹 Starting data cleaning...")

# 1. 基础清洗
df["comment_text"] = df["comment_text"].astype(str)
original_count = len(df)
df = df[df["comment_text"].apply(is_valid_ditto_comment)]
print(f"📝 Valid comments: {len(df)} (removed {original_count - len(df)} invalid)")

# 2. 高级去重
df = advanced_deduplication(df)

# 3. 语言检测
if LANG_DETECT_AVAILABLE:
    print("🌍 Detecting languages...")
    df['detected_language'] = df['comment_text'].apply(detect_language)
    
    # 语言分布统计
    lang_dist = df['detected_language'].value_counts()
    print("📊 Language distribution:")
    for lang, count in lang_dist.head(5).items():
        percentage = count/len(df)*100
        print(f"   {lang}: {count} ({percentage:.1f}%)")

# 4. 评论类型分类
print("🎭 Classifying comment types...")
df['comment_type'] = df['comment_text'].apply(classify_comment_type)

# 评论类型分布
type_dist = df['comment_type'].value_counts()
print("📊 Comment type distribution:")
for ctype, count in type_dist.items():
    percentage = count/len(df)*100
    print(f"   {ctype}: {count} ({percentage:.1f}%)")

# 5. 评论长度和质量指标
df['comment_length'] = df['comment_text'].str.len()
df['word_count'] = df['comment_text'].apply(lambda x: len(str(x).split()))

# 6. 时间处理（如果有时间戳）
if 'published_at' in df.columns:
    df['published_at'] = pd.to_datetime(df['published_at'])
    df['hour_of_day'] = df['published_at'].dt.hour
    df['day_of_week'] = df['published_at'].dt.day_name()

# 7. 质量分数计算
def calculate_quality_score(row):
    """综合评估评论质量"""
    score = 0
    
    # 长度分数 (0-30分)
    length = row['comment_length']
    if length > 100:
        score += 30
    elif length > 50:
        score += 20
    elif length > 20:
        score += 10
    
    # 点赞分数 (0-40分)
    likes = row.get('like_count', 0)
    if likes > 50:
        score += 40
    elif likes > 10:
        score += 30
    elif likes > 5:
        score += 20
    elif likes > 0:
        score += 10
    
    # 回复分数 (0-20分)
    replies = row.get('reply_count', 0)
    if replies > 5:
        score += 20
    elif replies > 0:
        score += 10
    
    # 类型分数 (0-10分)
    if row['comment_type'] in ['mv_analysis', 'emotional_response']:
        score += 10
    elif row['comment_type'] == 'music_technical':
        score += 5
    
    return score

df['quality_score'] = df.apply(calculate_quality_score, axis=1)

# 📊 清洗结果统计
print("\n" + "="*50)
print("📈 CLEANING SUMMARY")
print("="*50)
print(f"✅ Final dataset: {len(df)} comments")
print(f"📏 Average comment length: {df['comment_length'].mean():.1f} characters")
print(f"📝 Average word count: {df['word_count'].mean():.1f} words")
print(f"👍 Average likes: {df.get('like_count', pd.Series([0])).mean():.1f}")
print(f"🏆 Average quality score: {df['quality_score'].mean():.1f}/100")

# 保存清洗后的数据
output_file = "clean_comments_ditto_side_a.csv"
df.to_csv(output_file, index=False, encoding="utf-8-sig")
print(f"\n💾 Cleaned data saved to: {output_file}")

# 8. 高质量评论预览
print("\n🔥 TOP 3 HIGHEST QUALITY COMMENTS:")
top_quality = df.nlargest(3, 'quality_score')[['comment_text', 'quality_score', 'comment_type', 'detected_language']]
for i, row in top_quality.iterrows():
    print(f"\nScore: {row['quality_score']}/100 | Type: {row['comment_type']} | Lang: {row.get('detected_language', 'N/A')}")
    print(f"Text: {row['comment_text'][:150]}...")

print("\n🎯 Ready for sentiment analysis and clustering!")
