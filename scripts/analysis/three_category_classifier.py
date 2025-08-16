import pandas as pd
import numpy as np
from collections import defaultdict
import re
from tqdm import tqdm

# 读取原始标注数据
print("📖 Loading original emotion-labeled comments...")
df = pd.read_csv("labeled_comments_hylt_hf.csv")  # 只保留emotion标注，重做fan_type

print(f"📊 Total comments to reclassify: {len(df)}")

# ============================================
# 新的3类分类系统
# ============================================

def classify_fan_type_simple(comment):
    """
    简化的3类分类器：
    - visual-focused: 外貌、造型、视觉相关
    - content-focused: 音乐、歌词、舞蹈等作品内容
    - general-praise: 一般性赞美、情感表达
    """
    
    if pd.isna(comment) or not comment.strip():
        return "general-praise"
    
    text = str(comment).lower()
    
    # Visual-focused 强特征
    visual_keywords = [
        # 直接外貌词汇
        'beautiful', 'handsome', 'pretty', 'gorgeous', 'hot', 'sexy', 'cute', 'stunning', 
        'visual', 'look', 'looks', 'face', 'eyes', 'smile', 'hair', 'skin',
        # 造型相关
        'outfit', 'dress', 'clothes', 'style', 'fashion', 'makeup', 'clothing', 'costume',
        'styling', 'appearance', 'image', 'photoshoot', 'photo',
        # 中文
        '帅', '美', '好看', '漂亮', '颜值', '外貌', '长相', '造型', '服装', '穿搭', 
        '发型', '妆容', '妆', '衣服', '形象', '气质',
        # 日文  
        'かっこいい', '美しい', 'きれい', '可愛い', 'イケメン', 'ビジュアル', 'ファッション',
        # 韩文
        '잘생겼다', '예쁘다', '외모', '스타일', '패션', '헤어', '메이크업'
    ]
    
    # Content-focused 强特征
    content_keywords = [
        # 音乐制作相关
        'music', 'song', 'track', 'beat', 'rhythm', 'melody', 'harmony', 'vocals', 
        'voice', 'singing', 'production', 'sound', 'audio', 'instrumental', 'composition',
        # 歌词相关
        'lyrics', 'words', 'meaning', 'message', 'story', 'verse', 'line', 'poetry',
        # 舞蹈表演相关  
        'dance', 'dancing', 'choreography', 'choreo', 'moves', 'performance', 'stage',
        'dancing', 'routine', 'steps', 'rhythm',
        # 创作相关
        'written', 'composed', 'produced', 'created', 'made', 'directed',
        # 中文
        '音乐', '歌', '歌曲', '编曲', '旋律', '节奏', '和声', '声音', '演唱', '制作',
        '歌词', '词', '意思', '含义', '故事', '舞蹈', '编舞', '动作', '表演', '舞台',
        # 日文
        '音楽', '歌', 'メロディー', 'リズム', '声', '歌詞', 'ダンス', '踊り', '振り付け',
        # 韩文
        '음악', '노래', '멜로디', '리듬', '목소리', '가사', '춤', '댄스', '안무'
    ]
    
    # 计算各类别得分
    visual_score = sum(1 for keyword in visual_keywords if keyword in text)
    content_score = sum(1 for keyword in content_keywords if keyword in text)
    
    # 特殊规则处理
    
    # 1. 强visual信号
    if visual_score >= 2 or any(strong_visual in text for strong_visual in 
                               ['so handsome', 'so beautiful', 'so pretty', 'visual king', 'visual queen',
                                '太帅', '太美', '颜值爆表', '盛世美颜']):
        return "visual-focused"
    
    # 2. 强content信号  
    if content_score >= 2 or any(strong_content in text for strong_content in
                               ['love this song', 'amazing music', 'perfect dance', 'lyrics are', 'choreography',
                                '好听', '歌词', '编舞', '舞蹈', '제일 좋아', '음악이']):
        return "content-focused"
    
    # 3. 单一visual关键词
    if visual_score == 1:
        return "visual-focused"
    
    # 4. 单一content关键词
    if content_score == 1:
        return "content-focused"
    
    # 5. 长度和复杂度规则
    if len(text.strip()) < 10:  # 很短的评论
        return "general-praise"
    
    # 6. 纯情感表达（无具体内容）
    pure_emotion_patterns = [
        r'^(omg|wow|amazing|incredible|perfect|love|great|awesome|beautiful)+[!\s]*$',
        r'^[😍❤️💕🔥👏✨🥰😘💖]+$',  # 纯表情符号
        r'^(so good|too good|very good|really good)[!\s]*$',
    ]
    
    for pattern in pure_emotion_patterns:
        if re.match(pattern, text.strip()):
            return "general-praise"
    
    # 7. 默认分类：根据长度和内容判断
    if len(text.split()) < 5:  # 短评论倾向于general-praise
        return "general-praise"
    else:  # 长评论但没找到明确特征，也归为general-praise
        return "general-praise"

# ============================================
# 重新分类所有评论
# ============================================

print("\n🔄 RECLASSIFYING ALL COMMENTS WITH 3-CATEGORY SYSTEM...")
print("-" * 60)

# 重新分类
new_fan_types = []
for comment in tqdm(df['comment_text'], desc="Applying 3-category classifier"):
    new_type = classify_fan_type_simple(comment)
    new_fan_types.append(new_type)

# 更新数据框
df['fan_type_new'] = new_fan_types

# 统计新的分布
new_distribution = pd.Series(new_fan_types).value_counts()
print(f"\n📊 NEW 3-CATEGORY DISTRIBUTION:")
for category, count in new_distribution.items():
    percentage = count / len(df) * 100
    print(f"   {category}: {count} ({percentage:.1f}%)")

# 检查改进效果
specific_types = new_distribution.get('visual-focused', 0) + new_distribution.get('content-focused', 0)
specific_ratio = specific_types / len(df) * 100
general_ratio = new_distribution.get('general-praise', 0) / len(df) * 100

print(f"\n🎯 CLASSIFICATION EFFECTIVENESS:")
print(f"   • Specific categorization: {specific_ratio:.1f}% (visual + content)")
print(f"   • General praise: {general_ratio:.1f}%")
print(f"   • Success rate: {specific_ratio:.1f}% (vs original {100-50:.1f}% unclear)")

# ============================================
# 质量检查：抽样验证
# ============================================

print(f"\n🔍 QUALITY CHECK: SAMPLE CLASSIFICATIONS")
print("-" * 60)

# 显示每类的样本
for category in ['visual-focused', 'content-focused', 'general-praise']:
    if category in new_distribution.index:
        samples = df[df['fan_type_new'] == category].head(3)
        print(f"\n💎 {category.upper()} examples:")
        for idx, row in samples.iterrows():
            comment = str(row['comment_text'])[:70]
            emotion = row['emotion']
            print(f"   Comment: {comment}...")
            print(f"   Emotion: {emotion}")

# ============================================
# 保存新分类结果
# ============================================

# 保存完整结果
df_final = df[['comment_text', 'emotion', 'fan_type_new']].copy()
df_final.rename(columns={'fan_type_new': 'fan_type'}, inplace=True)
df_final.to_csv("labeled_comments_hylt_3category.csv", index=False, encoding="utf-8-sig")

print(f"\n💾 SAVED NEW CLASSIFICATION:")
print(f"   ✅ labeled_comments_hylt_3category.csv")

# ============================================
# 生成对比可视化数据
# ============================================

print(f"\n📊 CREATING COMPARISON DATA FOR VISUALIZATION...")

# 对比数据
comparison_data = {
    'Original_5category': df['fan_type'].value_counts().to_dict(),
    'New_3category': df_final['fan_type'].value_counts().to_dict()
}

print(f"\n📈 COMPARISON SUMMARY:")
print(f"   Original system: 5 categories, ~50% unclear")
print(f"   New system: 3 categories, {general_ratio:.1f}% general-praise")
print(f"   Improvement: {50 - general_ratio:.1f}% better specificity")

print(f"\n🚀 NEXT STEPS:")
print(f"   1. 📊 Generate new visualizations with 3-category data")
print(f"   2. 🎵 Proceed to Spotify audio features analysis") 
print(f"   3. 🔄 Build emotion-audio feature correlation analysis")
print(f"   4. 🎯 Design the final diagnosis interface")

print(f"\n✅ 3-CATEGORY CLASSIFIER READY!")
print(f"更realistic、更实用、更准确的分类系统完成！")
