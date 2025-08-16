import pandas as pd
from transformers import pipeline
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# 初始化模型 (首次运行会自动下载)
print("🔥 Loading HuggingFace models...")

# 多语言情感分析模型
emotion_classifier = pipeline(
    "sentiment-analysis", 
    model="cardiffnlp/twitter-xlm-roberta-base-sentiment-multilingual",
    return_all_scores=True
)

# 零样本分类模型 (用于粉丝类型识别)
fan_type_classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli"
)

print("✅ Models loaded successfully!")

# 情感标签映射 (从HF标签到你的标签)
def map_emotion_to_custom(hf_result):
    """将HuggingFace的情感结果映射到自定义情感标签"""
    # HF返回: LABEL_0(negative), LABEL_1(neutral), LABEL_2(positive)
    scores = {item['label']: item['score'] for item in hf_result}
    
    # 获取最高分的情感
    max_label = max(scores, key=scores.get)
    max_score = scores[max_label]
    
    # 根据置信度和标签映射到具体情感
    if max_label == 'LABEL_2':  # Positive
        if max_score > 0.8:
            return "cool"  # 强正面
        else:
            return "cute"  # 轻正面
    elif max_label == 'LABEL_0':  # Negative
        return "critical"
    else:  # Neutral
        return "neutral"

def analyze_fan_type_with_hf(text):
    """使用零样本分类识别粉丝类型"""
    candidate_labels = [
        "lyrics and meaning focus",  # lyrics-lover
        "visual and appearance focus",  # visual-lover  
        "dance and performance focus",  # dance-lover
        "personality and character focus",  # personality-lover
        "music and production focus",  # music-lover
        "unclear or general comment"  # unclear
    ]
    
    try:
        result = fan_type_classifier(text, candidate_labels)
        top_label = result['labels'][0]
        confidence = result['scores'][0]
        
        # 映射到你的标签
        label_mapping = {
            "lyrics and meaning focus": "lyrics-lover",
            "visual and appearance focus": "visual-lover", 
            "dance and performance focus": "dance-lover",
            "personality and character focus": "personality-lover",
            "music and production focus": "music-lover",
            "unclear or general comment": "unclear"
        }
        
        # 如果置信度太低，标记为unclear
        if confidence < 0.3:
            return "unclear"
        
        return label_mapping.get(top_label, "unclear")
    
    except Exception as e:
        print(f"❌ Fan type analysis error: {str(e)}")
        return "unclear"

def analyze_emotion_with_keywords_fallback(text):
    """关键词回退方案（备用）"""
    text_lower = str(text).lower()
    
    # 强特征关键词
    if any(word in text_lower for word in ["cry", "tears", "감동", "泪", "感动"]):
        return "touching"
    elif any(word in text_lower for word in ["addicted", "loop", "replay", "중독", "循环"]):
        return "addictive"  
    elif any(word in text_lower for word in ["inspire", "hope", "strength", "희망"]):
        return "inspiring"
    elif any(word in text_lower for word in ["remember", "nostalgia", "miss", "그립다", "怀念"]):
        return "nostalgic"
    else:
        return None  # 让HF模型决定

def label_comment_hf(comment):
    """使用HuggingFace模型标注评论"""
    try:
        # 清理文本
        if pd.isna(comment) or comment.strip() == "":
            return "neutral", "unclear"
        
        comment_str = str(comment).strip()
        
        # 情感分析
        # 先尝试关键词快速识别
        emotion = analyze_emotion_with_keywords_fallback(comment_str)
        if emotion is None:
            # 使用HF模型
            emotion_result = emotion_classifier(comment_str)
            emotion = map_emotion_to_custom(emotion_result[0])
        
        # 粉丝类型识别
        fan_type = analyze_fan_type_with_hf(comment_str)
        
        return emotion, fan_type
        
    except Exception as e:
        print(f"❌ Error for comment: {str(comment)[:20]}... → {str(e)}")
        return "neutral", "unclear"

# 批量处理优化版本
def batch_label_comments(comments, batch_size=32):
    """批量处理评论以提高效率"""
    results = []
    
    for i in tqdm(range(0, len(comments), batch_size), desc="🚀 HuggingFace batch processing"):
        batch = comments[i:i+batch_size]
        batch_results = []
        
        for comment in batch:
            emotion, fan_type = label_comment_hf(comment)
            batch_results.append({
                "comment_text": comment,
                "emotion": emotion, 
                "fan_type": fan_type
            })
        
        results.extend(batch_results)
    
    return results

# 主程序
if __name__ == "__main__":
    # 读取评论数据
    print("📖 Reading comments...")
    df = pd.read_csv("clean_comments_how_you_like_that.csv")
    comments = df["comment_text"].tolist()
    
    print(f"📊 Total comments to process: {len(comments)}")
    
    # 批量处理评论
    labeled_data = batch_label_comments(comments)
    
    # 保存结果
    labeled_df = pd.DataFrame(labeled_data)
    labeled_df.to_csv("labeled_comments_hylt_hf.csv", index=False, encoding="utf-8-sig")
    
    # 统计结果
    print("\n📊 情感分布统计:")
    emotion_counts = labeled_df["emotion"].value_counts()
    print(emotion_counts)
    
    print("\n📊 粉丝类型分布统计:")
    fan_type_counts = labeled_df["fan_type"].value_counts()
    print(fan_type_counts)
    
    # 样本展示
    print("\n🔍 标注样本预览:")
    sample_df = labeled_df.head(10)[["comment_text", "emotion", "fan_type"]]
    for idx, row in sample_df.iterrows():
        print(f"Comment: {row['comment_text'][:50]}...")
        print(f"Emotion: {row['emotion']}, Fan Type: {row['fan_type']}\n")
    
    print(f"\n✅ 完成！共标注 {len(labeled_data)} 条评论")
    print("保存文件: 'labeled_comments_hylt_hf.csv'")
    print("💡 模型下载完成后，后续运行会更快！")