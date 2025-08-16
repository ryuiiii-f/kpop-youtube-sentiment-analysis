import pandas as pd
from transformers import pipeline
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

print("🚀 LOADING HUGGINGFACE MODELS FOR 3-CATEGORY CLASSIFICATION...")

# 加载零样本分类模型 (多语言支持更好的版本)
classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli"  # 或者试试 "microsoft/DialoGPT-medium" 
)

print("✅ Model loaded successfully!")

# 读取原始数据
df = pd.read_csv("labeled_comments_hylt_hf.csv")
print(f"📊 Total comments to reclassify: {len(df)}")

def classify_fan_type_hf_3category(text):
    """
    使用HuggingFace进行3类分类
    针对英韩日语言优化的标签设计
    """
    
    if pd.isna(text) or not str(text).strip():
        return "general-praise"
    
    # 优化的候选标签 (针对多语言特点)
    candidate_labels = [
        # 更具体、更容易区分的标签描述
        "visual appearance styling fashion looks makeup outfits", # visual-focused
        "music lyrics dance choreography song performance content", # content-focused  
        "general praise love excitement emotional reaction" # general-praise
    ]
    
    try:
        result = classifier(str(text), candidate_labels)
        top_label = result['labels'][0]
        confidence = result['scores'][0]
        
        # 标签映射
        label_mapping = {
            "visual appearance styling fashion looks makeup outfits": "visual-focused",
            "music lyrics dance choreography song performance content": "content-focused", 
            "general praise love excitement emotional reaction": "general-praise"
        }
        
        # 置信度阈值 (3类应该更有信心)
        if confidence < 0.4:  # 降低阈值，因为3类更容易区分
            return "general-praise"  # 不确定的归为general-praise
        
        return label_mapping.get(top_label, "general-praise")
        
    except Exception as e:
        print(f"❌ Classification error: {str(e)}")
        return "general-praise"

# ============================================
# 批量处理 (优化版本)
# ============================================

def batch_classify_3category(comments, batch_size=16):
    """批量分类以提高效率"""
    results = []
    
    print("🔄 Starting 3-category classification...")
    
    for i in tqdm(range(0, len(comments), batch_size), desc="🎯 HF 3-category processing"):
        batch = comments[i:i+batch_size]
        
        for comment in batch:
            fan_type = classify_fan_type_hf_3category(comment)
            results.append(fan_type)
    
    return results

# 执行分类
comments = df['comment_text'].tolist()
new_fan_types = batch_classify_3category(comments)

# 更新数据框
df['fan_type_hf3'] = new_fan_types

# ============================================
# 结果分析
# ============================================

print("\n📊 HUGGINGFACE 3-CATEGORY RESULTS:")
print("="*50)

hf3_counts = df['fan_type_hf3'].value_counts()
for category, count in hf3_counts.items():
    percentage = count/len(df)*100
    print(f"   {category}: {count} ({percentage:.1f}%)")

# 计算具体分类成功率
specific_categories = hf3_counts.get('visual-focused', 0) + hf3_counts.get('content-focused', 0)
specific_ratio = specific_categories / len(df) * 100
general_ratio = hf3_counts.get('general-praise', 0) / len(df) * 100

print(f"\n🎯 PERFORMANCE METRICS:")
print(f"   ✅ Specific categorization: {specific_ratio:.1f}% (visual + content)")
print(f"   ✅ General praise: {general_ratio:.1f}%")
print(f"   ✅ Compared to original unclear: {100-50:.1f}% vs {specific_ratio:.1f}%")

# ============================================
# 质量检查
# ============================================

print(f"\n🔍 QUALITY CHECK - SAMPLE CLASSIFICATIONS:")
print("-" * 60)

for category in ['visual-focused', 'content-focused', 'general-praise']:
    if category in hf3_counts.index:
        samples = df[df['fan_type_hf3'] == category].head(3)
        print(f"\n💎 {category.upper()} examples:")
        for idx, row in samples.iterrows():
            comment = str(row['comment_text'])[:70]
            emotion = row['emotion']
            print(f"   '{comment}...'")
            print(f"   → Emotion: {emotion}, Type: {category}")

# ============================================
# 语言分布分析
# ============================================

def detect_language_pattern(text):
    """简单的语言模式检测"""
    text = str(text).lower()
    
    # 韩文字符
    korean_chars = re.findall(r'[가-힣]', text)
    # 日文字符  
    japanese_chars = re.findall(r'[ひらがなカタカナ]|[一-龯]', text)
    # 英文为主
    english_chars = re.findall(r'[a-zA-Z]', text)
    
    total_chars = len(korean_chars) + len(japanese_chars) + len(english_chars)
    
    if total_chars == 0:
        return "mixed"
    
    korean_ratio = len(korean_chars) / total_chars
    japanese_ratio = len(japanese_chars) / total_chars  
    english_ratio = len(english_chars) / total_chars
    
    if korean_ratio > 0.3:
        return "korean-dominant"
    elif japanese_ratio > 0.3:
        return "japanese-dominant"
    elif english_ratio > 0.7:
        return "english-dominant"
    else:
        return "mixed"

# 分析语言分布
df['language_pattern'] = df['comment_text'].apply(detect_language_pattern)
language_dist = df['language_pattern'].value_counts()

print(f"\n🌐 LANGUAGE DISTRIBUTION IN COMMENTS:")
for lang, count in language_dist.items():
    percentage = count/len(df)*100
    print(f"   {lang}: {count} ({percentage:.1f}%)")

# 语言 vs 分类效果分析
print(f"\n📊 CLASSIFICATION EFFECTIVENESS BY LANGUAGE:")
for lang in language_dist.index:
    lang_df = df[df['language_pattern'] == lang]
    lang_specific = (lang_df['fan_type_hf3'] != 'general-praise').sum()
    lang_ratio = lang_specific / len(lang_df) * 100 if len(lang_df) > 0 else 0
    print(f"   {lang}: {lang_ratio:.1f}% specific classification")

# ============================================
# 保存结果
# ============================================

# 保存HF 3类结果
df_hf3 = df[['comment_text', 'emotion', 'fan_type_hf3', 'language_pattern']].copy()
df_hf3.rename(columns={'fan_type_hf3': 'fan_type'}, inplace=True)
df_hf3.to_csv("labeled_comments_hylt_hf3category.csv", index=False, encoding="utf-8-sig")

print(f"\n💾 SAVED HUGGINGFACE 3-CATEGORY RESULTS:")
print(f"   ✅ labeled_comments_hylt_hf3category.csv")

# ============================================
# 对比分析建议
# ============================================

print(f"\n🎯 NEXT STEPS:")
print(f"   1. 📊 Compare HF 3-category vs Rule-based 3-category results")
print(f"   2. 🔄 Choose the better performing approach")
print(f"   3. 🎵 Proceed to Spotify audio features analysis")
print(f"   4. 📈 Build correlation analysis between audio features and fan reactions")

print(f"\n💡 RECOMMENDATION:")
if specific_ratio > 60:
    print(f"   🎉 HuggingFace 3-category works well! Use this for final analysis.")
elif specific_ratio > 45:
    print(f"   ✅ Decent results. Compare with rule-based approach.")
else:
    print(f"   🤔 May need to stick with rule-based approach or hybrid method.")

print(f"\n📈 HUGGINGFACE 3-CATEGORY CLASSIFICATION COMPLETE!")
