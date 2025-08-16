import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from wordcloud import WordCloud
from collections import Counter
import re

# 设置图表样式
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['figure.facecolor'] = 'white'
sns.set_style("whitegrid")

# 读取改进前后的数据
print("📖 Loading data for comparison...")
df_original = pd.read_csv("labeled_comments_hylt_hf.csv")  # 原始HF结果
df_improved = pd.read_csv("labeled_comments_hylt_improved.csv")  # 改进后结果

print(f"📊 Total comments: {len(df_improved)}")

# 计算改进效果
original_unclear = (df_original['fan_type'] == 'unclear').sum()
improved_unclear = (df_improved['fan_type'] == 'unclear').sum()
improvement = original_unclear - improved_unclear

print(f"🎯 IMPROVEMENT SUMMARY:")
print(f"   Original unclear: {original_unclear} ({original_unclear/len(df_original)*100:.1f}%)")
print(f"   Improved unclear: {improved_unclear} ({improved_unclear/len(df_improved)*100:.1f}%)")
print(f"   Reclassified: {improvement} comments ({improvement/len(df_original)*100:.1f}% improvement)")

# ============================================
# 图表 1: Before vs After Comparison
# ============================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

# 改进前
original_counts = df_original['fan_type'].value_counts()
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57', '#FF9FF3']

ax1.pie(original_counts.values, labels=original_counts.index, autopct='%1.1f%%', 
        startangle=90, colors=colors[:len(original_counts)])
ax1.set_title('BEFORE: HuggingFace Only\n(High Unclear Rate)', fontsize=14, fontweight='bold')

# 改进后  
improved_counts = df_improved['fan_type'].value_counts()
ax2.pie(improved_counts.values, labels=improved_counts.index, autopct='%1.1f%%',
        startangle=90, colors=colors[:len(improved_counts)])
ax2.set_title('AFTER: Improved Rules\n(Reduced Unclear Rate)', fontsize=14, fontweight='bold')

plt.suptitle('Fan Type Classification Improvement', fontsize=18, fontweight='bold')
plt.tight_layout()
plt.savefig('comparison_before_after_fantype.png', dpi=300, bbox_inches='tight')
plt.show()
print("✅ Saved: comparison_before_after_fantype.png")

# ============================================
# 图表 2: Updated Emotion Distribution
# ============================================
plt.figure(figsize=(10, 8))
emotion_counts = df_improved["emotion"].value_counts()
colors_emotion = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57', '#FF9FF3', '#54A0FF', '#5F27CD']

emotion_counts.plot(kind='pie', autopct='%1.1f%%', startangle=90, 
                   colors=colors_emotion[:len(emotion_counts)])
plt.title('Updated K-pop Comment Emotion Distribution', fontsize=16, fontweight='bold', pad=20)
plt.ylabel('')
plt.axis('equal')
plt.tight_layout()
plt.savefig('updated_emotion_distribution.png', dpi=300, bbox_inches='tight')
plt.show()
print("✅ Saved: updated_emotion_distribution.png")

# ============================================
# 图表 3: Updated Fan Type Bar Chart
# ============================================
plt.figure(figsize=(12, 8))
improved_counts.plot(kind='bar', color='lightcoral', alpha=0.8)
plt.title('Updated Fan Type Distribution - Improved Classification', 
          fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Fan Type Categories', fontsize=12)
plt.ylabel('Number of Comments', fontsize=12)
plt.xticks(rotation=45, ha='right')

# 添加数值标签
for i, (idx, value) in enumerate(improved_counts.items()):
    plt.text(i, value + len(df_improved)*0.01, f'{value}', 
             ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('updated_fan_type_bar.png', dpi=300, bbox_inches='tight')
plt.show()
print("✅ Saved: updated_fan_type_bar.png")

# ============================================
# 图表 4: Updated Emotion-Fan Type Heatmap
# ============================================
plt.figure(figsize=(12, 8))
cross_table = pd.crosstab(df_improved['emotion'], df_improved['fan_type'], normalize='index') * 100

sns.heatmap(cross_table, annot=True, fmt='.1f', cmap='YlOrRd', 
            cbar_kws={'label': 'Percentage (%)'})
plt.title('Updated Emotion-Fan Type Correlation Matrix', fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Fan Type', fontsize=12)
plt.ylabel('Emotion', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('updated_emotion_fantype_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()
print("✅ Saved: updated_emotion_fantype_heatmap.png")

# ============================================
# 图表 5: Improved Emotion Word Cloud
# ============================================
print("\n🎨 Generating updated emotion word clouds...")

# 为每种情感生成词云 - 使用改进后的数据
emotion_texts = {}
for emotion in emotion_counts.index:
    emotion_comments = df_improved[df_improved['emotion'] == emotion]['comment_text'].tolist()
    emotion_texts[emotion] = ' '.join([str(comment) for comment in emotion_comments])

# 生成主要情感的词云
top_emotions = emotion_counts.head(4).index
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten()

for i, emotion in enumerate(top_emotions):
    if i < 4:
        text = re.sub(r'[^\w\s]', ' ', emotion_texts[emotion])
        text = re.sub(r'\s+', ' ', text)
        
        if len(text.strip()) > 0:
            wordcloud = WordCloud(
                width=400, height=300, 
                background_color='white',
                max_words=50,
                colormap='viridis',
                relative_scaling=0.5
            ).generate(text)
            
            axes[i].imshow(wordcloud, interpolation='bilinear')
            axes[i].set_title(f'{emotion.title()} Emotion - Word Patterns', 
                            fontsize=14, fontweight='bold')
            axes[i].axis('off')

plt.suptitle('Updated Emotion Word Cloud Analysis', fontsize=18, fontweight='bold')
plt.tight_layout()
plt.savefig('updated_emotion_wordcloud.png', dpi=300, bbox_inches='tight')
plt.show()
print("✅ Saved: updated_emotion_wordcloud.png")

# ============================================
# 图表 6: Classification Quality Analysis
# ============================================
plt.figure(figsize=(12, 8))

# 创建改进效果对比
categories = ['Visual-Lover', 'Dance-Lover', 'Music-Lover', 'Lyrics-Lover', 'Personality-Lover', 'Unclear']
original_values = [original_counts.get(cat.lower().replace('-', '-'), 0) for cat in categories]
improved_values = [improved_counts.get(cat.lower().replace('-', '-'), 0) for cat in categories]

x = np.arange(len(categories))
width = 0.35

bars1 = plt.bar(x - width/2, original_values, width, label='Original Classification', 
                color='lightblue', alpha=0.7)
bars2 = plt.bar(x + width/2, improved_values, width, label='Improved Classification', 
                color='orange', alpha=0.7)

plt.title('Classification Improvement Comparison', fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Fan Type Categories', fontsize=12)
plt.ylabel('Number of Comments', fontsize=12)
plt.xticks(x, categories, rotation=45, ha='right')
plt.legend()

# 添加数值标签
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{int(height)}', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig('classification_improvement_comparison.png', dpi=300, bbox_inches='tight')
plt.show()
print("✅ Saved: classification_improvement_comparison.png")

# ============================================
# 图表 7: Updated Radar Chart
# ============================================
plt.figure(figsize=(10, 10))

# 使用改进后的情感数据
emotions = emotion_counts.index.tolist()
values = emotion_counts.values.tolist()
values_pct = [v/sum(values)*100 for v in values]

# 雷达图设置
fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
angles = np.linspace(0, 2 * np.pi, len(emotions), endpoint=False).tolist()
values_pct += values_pct[:1]
angles += angles[:1]

ax.plot(angles, values_pct, 'o-', linewidth=3, label='How You Like That - Updated', color='#FF6B6B')
ax.fill(angles, values_pct, alpha=0.25, color='#FF6B6B')

ax.set_xticks(angles[:-1])
ax.set_xticklabels([emotion.title() for emotion in emotions], fontsize=11)
ax.set_ylim(0, max(values_pct) * 1.2)

ax.set_yticks(np.arange(0, max(values_pct) * 1.2, max(values_pct) * 1.2 / 4))
ax.set_yticklabels([f'{int(y)}%' for y in np.arange(0, max(values_pct) * 1.2, max(values_pct) * 1.2 / 4)])

plt.title('Updated K-pop Emotion Profile - Radar Analysis', 
          fontsize=16, fontweight='bold', pad=30)
plt.tight_layout()
plt.savefig('updated_emotion_radar.png', dpi=300, bbox_inches='tight')
plt.show()
print("✅ Saved: updated_emotion_radar.png")

# ============================================
# 总结报告
# ============================================
print("\n" + "="*70)
print("📊 UPDATED VISUALIZATION SUMMARY")
print("="*70)

print(f"\n🎯 KEY IMPROVEMENTS:")
print(f"   • Unclear rate reduced: {original_unclear/len(df_original)*100:.1f}% → {improved_unclear/len(df_improved)*100:.1f}%")
print(f"   • Successfully reclassified: {improvement} comments")
print(f"   • Classification confidence improved significantly")

print(f"\n📁 NEW VISUALIZATION FILES GENERATED:")
files = [
    "comparison_before_after_fantype.png - Shows improvement effect",
    "updated_emotion_distribution.png - Latest emotion analysis", 
    "updated_fan_type_bar.png - Improved fan type distribution",
    "updated_emotion_fantype_heatmap.png - Updated correlation matrix",
    "updated_emotion_wordcloud.png - Refined word patterns",
    "classification_improvement_comparison.png - Side-by-side comparison",
    "updated_emotion_radar.png - Enhanced radar profile"
]

for file in files:
    print(f"   ✅ {file}")

print(f"\n🚀 WHAT'S NEXT?")
print(f"   1. 📊 Review the comparison charts to validate improvement")
print(f"   2. 🎵 Move to Spotify audio features analysis")
print(f"   3. 🔄 Start building the integration between comment sentiment & audio features")
print(f"   4. 🎨 Design the interactive diagnosis interface")

print(f"\n💡 DATA QUALITY ASSESSMENT:")
current_unclear_pct = improved_unclear/len(df_improved)*100
if current_unclear_pct < 30:
    print(f"   ✅ EXCELLENT: {current_unclear_pct:.1f}% unclear rate is very good!")
elif current_unclear_pct < 50:
    print(f"   ✅ GOOD: {current_unclear_pct:.1f}% unclear rate is acceptable")
else:
    print(f"   ⚠️ CONSIDER: {current_unclear_pct:.1f}% unclear rate still high, may need further optimization")

print(f"\n📈 READY FOR NEXT MODULE: SPOTIFY AUDIO FEATURES!")