import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from wordcloud import WordCloud
from collections import Counter
import re
import warnings
warnings.filterwarnings("ignore")

# 设置图表样式
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['figure.facecolor'] = 'white'
sns.set_style("whitegrid")

print("🎨 COMPREHENSIVE K-POP ANALYSIS VISUALIZATION SUITE")
print("="*70)

# ============================================
# 数据加载和整合
# ============================================

print("📖 Loading all analysis results...")

# 主要数据：情感 + 无监督聚类发现
try:
    df_main = pd.read_csv("labeled_comments_discovered_types.csv")
    print("✅ Loaded: Discovered types data")
    has_discovered = True
except:
    df_main = pd.read_csv("labeled_comments_hylt_hf.csv")
    print("⚠️ Using: Original HF data (discovered types not found)")
    has_discovered = False

# 尝试加载Rule-based结果进行对比
try:
    df_rule = pd.read_csv("labeled_comments_hylt_3category.csv")
    print("✅ Loaded: Rule-based 3-category data")
    has_rule = True
except:
    print("⚠️ Rule-based data not found")
    has_rule = False

print(f"📊 Total comments: {len(df_main)}")

# ============================================
# 图表 1: Emotion Distribution Pie Chart
# ============================================
print("\n🎭 Generating Emotion Distribution Analysis...")

emotion_counts = df_main["emotion"].value_counts()
plt.figure(figsize=(12, 10))

# 创建更美观的饼图
colors_emotion = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57', '#FF9FF3', '#54A0FF', '#5F27CD', '#48CAE4']
explode = [0.05 if i == 0 else 0.02 for i in range(len(emotion_counts))]  # 突出最大的

wedges, texts, autotexts = plt.pie(emotion_counts.values, labels=emotion_counts.index, 
                                   autopct='%1.1f%%', startangle=90, 
                                   colors=colors_emotion[:len(emotion_counts)],
                                   explode=explode, shadow=True)

# 美化文字
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontweight('bold')
    autotext.set_fontsize(10)

plt.title('K-pop Comment Emotion Distribution Analysis', 
          fontsize=18, fontweight='bold', pad=30)
plt.axis('equal')
plt.tight_layout()
plt.savefig('01_emotion_distribution_pie.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()
print("✅ Saved: 01_emotion_distribution_pie.png")

# ============================================
# 图表 2: Fan Type Distribution Bar Chart
# ============================================
print("\n👥 Generating Fan Type Distribution Analysis...")

if has_discovered:
    fan_type_counts = df_main["fan_type_discovered"].value_counts()
    title_suffix = "Data-Driven Discovery"
else:
    fan_type_counts = df_main.get("fan_type", pd.Series(["general-praise"] * len(df_main))).value_counts()
    title_suffix = "Original Classification"

plt.figure(figsize=(14, 8))
bars = fan_type_counts.plot(kind='bar', color='lightcoral', alpha=0.8, width=0.7)
plt.title(f'Fan Type Distribution - {title_suffix}', 
          fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Fan Type Categories', fontsize=12)
plt.ylabel('Number of Comments', fontsize=12)
plt.xticks(rotation=45, ha='right')

# 添加数值标签和百分比
for i, (idx, value) in enumerate(fan_type_counts.items()):
    percentage = value/len(df_main)*100
    plt.text(i, value + len(df_main)*0.01, f'{value}\n({percentage:.1f}%)', 
             ha='center', va='bottom', fontweight='bold')

plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('02_fan_type_distribution_bar.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()
print("✅ Saved: 02_fan_type_distribution_bar.png")

# ============================================
# 图表 3: Emotion-Fan Type Correlation Heatmap
# ============================================
print("\n🔥 Generating Correlation Heatmap...")

plt.figure(figsize=(12, 8))
if has_discovered:
    cross_table = pd.crosstab(df_main['emotion'], df_main['fan_type_discovered'], normalize='index') * 100
else:
    cross_table = pd.crosstab(df_main['emotion'], df_main.get('fan_type', 'general'), normalize='index') * 100

sns.heatmap(cross_table, annot=True, fmt='.1f', cmap='RdYlBu_r', 
            cbar_kws={'label': 'Percentage (%)'}, linewidths=0.5)
plt.title('Emotion-Fan Type Correlation Matrix\n(Row-wise Percentage)', 
          fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Fan Type', fontsize=12)
plt.ylabel('Emotion', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('03_emotion_fantype_heatmap.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()
print("✅ Saved: 03_emotion_fantype_heatmap.png")

# ============================================
# 图表 4: Comment Length Distribution + Analysis
# ============================================
print("\n📏 Generating Comment Length Analysis...")

df_main['comment_length'] = df_main['comment_text'].str.len()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# 长度分布直方图
df_main['comment_length'].hist(bins=30, alpha=0.7, color='green', edgecolor='black', ax=ax1)
mean_length = df_main['comment_length'].mean()
median_length = df_main['comment_length'].median()

ax1.axvline(mean_length, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_length:.1f}')
ax1.axvline(median_length, color='orange', linestyle='--', linewidth=2, label=f'Median: {median_length:.1f}')
ax1.set_title('Comment Length Distribution', fontsize=14, fontweight='bold')
ax1.set_xlabel('Character Count')
ax1.set_ylabel('Frequency')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 长度 vs 情感箱线图
emotion_order = emotion_counts.index.tolist()
sns.boxplot(data=df_main, x='emotion', y='comment_length', order=emotion_order, ax=ax2)
ax2.set_title('Comment Length by Emotion Type', fontsize=14, fontweight='bold')
ax2.set_xlabel('Emotion')
ax2.set_ylabel('Character Count')
ax2.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('04_comment_length_analysis.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()
print("✅ Saved: 04_comment_length_analysis.png")

# ============================================
# 图表 5: Emotion Word Clouds (4 Main Emotions)
# ============================================
print("\n🎨 Generating Emotion Word Clouds...")

top_emotions = emotion_counts.head(4).index
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten()

for i, emotion in enumerate(top_emotions):
    emotion_comments = df_main[df_main['emotion'] == emotion]['comment_text'].tolist()
    text = ' '.join([str(comment) for comment in emotion_comments])
    
    # 清理文本
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    
    if len(text.strip()) > 0:
        # 为不同情感使用不同色彩主题
        colormaps = ['Reds', 'Blues', 'Greens', 'Purples']
        
        wordcloud = WordCloud(
            width=400, height=300, 
            background_color='white',
            max_words=60,
            colormap=colormaps[i],
            relative_scaling=0.6
        ).generate(text)
        
        axes[i].imshow(wordcloud, interpolation='bilinear')
        axes[i].set_title(f'{emotion.title()} Emotion Word Patterns\n({emotion_counts[emotion]} comments)', 
                        fontsize=12, fontweight='bold')
        axes[i].axis('off')

plt.suptitle('Emotion-Specific Word Cloud Analysis', fontsize=18, fontweight='bold')
plt.tight_layout()
plt.savefig('05_emotion_wordcloud_grid.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()
print("✅ Saved: 05_emotion_wordcloud_grid.png")

# ============================================
# 图表 6: Emotion Radar Chart (Enhanced)
# ============================================
print("\n📡 Generating Enhanced Emotion Radar Chart...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8), subplot_kw=dict(projection='polar'))

# 左侧：基础情感雷达图
emotions = emotion_counts.index.tolist()
values = emotion_counts.values.tolist()
values_pct = [v/sum(values)*100 for v in values]

angles = np.linspace(0, 2 * np.pi, len(emotions), endpoint=False).tolist()
values_pct += values_pct[:1]
angles += angles[:1]

ax1.plot(angles, values_pct, 'o-', linewidth=3, color='#FF6B6B', markersize=8)
ax1.fill(angles, values_pct, alpha=0.25, color='#FF6B6B')
ax1.set_xticks(angles[:-1])
ax1.set_xticklabels([emotion.title() for emotion in emotions], fontsize=10)
ax1.set_ylim(0, max(values_pct) * 1.2)
ax1.set_title('How You Like That\nEmotion Profile', fontsize=14, fontweight='bold', pad=20)

# 右侧：情感强度雷达图（基于评论长度加权）
weighted_emotions = {}
for emotion in emotions:
    emotion_df = df_main[df_main['emotion'] == emotion]
    avg_length = emotion_df['comment_length'].mean() if len(emotion_df) > 0 else 0
    weighted_emotions[emotion] = avg_length

weighted_values = list(weighted_emotions.values())
max_weight = max(weighted_values) if weighted_values else 1
weighted_values_norm = [v/max_weight*100 for v in weighted_values]
weighted_values_norm += weighted_values_norm[:1]

ax2.plot(angles, weighted_values_norm, 'o-', linewidth=3, color='#4ECDC4', markersize=8)
ax2.fill(angles, weighted_values_norm, alpha=0.25, color='#4ECDC4')
ax2.set_xticks(angles[:-1])
ax2.set_xticklabels([emotion.title() for emotion in emotions], fontsize=10)
ax2.set_ylim(0, max(weighted_values_norm) * 1.2)
ax2.set_title('Emotion Intensity Profile\n(Weighted by Comment Length)', fontsize=14, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig('06_emotion_radar_enhanced.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()
print("✅ Saved: 06_emotion_radar_enhanced.png")

# ============================================
# 图表 7: Fan Type Pie Chart (Enhanced)
# ============================================
print("\n🥧 Generating Enhanced Fan Type Pie Chart...")

plt.figure(figsize=(12, 10))

if has_discovered:
    fan_data = df_main["fan_type_discovered"].value_counts()
    title = "Data-Driven Fan Type Discovery"
else:
    fan_data = df_main.get("fan_type", pd.Series(["general-content"] * len(df_main))).value_counts()
    title = "Fan Type Distribution"

colors_fan = ['#8B5CF6', '#06D6A0', '#F72585', '#4CC9F0', '#FFD23F', '#FB5607']
explode = [0.1 if fan_data.iloc[i] < len(df_main)*0.1 else 0.05 for i in range(len(fan_data))]

wedges, texts, autotexts = plt.pie(fan_data.values, labels=fan_data.index, 
                                   autopct='%1.1f%%', startangle=90,
                                   colors=colors_fan[:len(fan_data)], 
                                   explode=explode, shadow=True,
                                   textprops={'fontsize': 11})

for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontweight('bold')

plt.title(f'{title}\nBased on Unsupervised Clustering Analysis', 
          fontsize=16, fontweight='bold', pad=20)
plt.axis('equal')
plt.tight_layout()
plt.savefig('07_fan_type_pie_enhanced.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()
print("✅ Saved: 07_fan_type_pie_enhanced.png")

# ============================================
# 图表 8: Emotion Frequency Bar Chart (Enhanced)
# ============================================
print("\n📊 Generating Enhanced Emotion Frequency Analysis...")

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))

# 上图：基础频次
bars1 = emotion_counts.plot(kind='bar', ax=ax1, color='coral', alpha=0.8)
ax1.set_title('Emotion Frequency Distribution', fontsize=14, fontweight='bold')
ax1.set_xlabel('')
ax1.set_ylabel('Number of Comments')
ax1.tick_params(axis='x', rotation=45)

for i, (emotion, count) in enumerate(emotion_counts.items()):
    percentage = count/len(df_main)*100
    ax1.text(i, count + len(df_main)*0.01, f'{count}\n({percentage:.1f}%)', 
             ha='center', va='bottom', fontweight='bold')

# 下图：情感的平均评论长度
emotion_avg_length = df_main.groupby('emotion')['comment_length'].mean().reindex(emotion_counts.index)
bars2 = emotion_avg_length.plot(kind='bar', ax=ax2, color='lightblue', alpha=0.8)
ax2.set_title('Average Comment Length by Emotion', fontsize=14, fontweight='bold')
ax2.set_xlabel('Emotion Type')
ax2.set_ylabel('Average Character Count')
ax2.tick_params(axis='x', rotation=45)

for i, (emotion, avg_len) in enumerate(emotion_avg_length.items()):
    ax2.text(i, avg_len + 2, f'{avg_len:.0f}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('08_emotion_frequency_enhanced.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()
print("✅ Saved: 08_emotion_frequency_enhanced.png")

# ============================================
# 图表 9: BONUS - Multi-Method Comparison
# ============================================
if has_rule and has_discovered:
    print("\n🔬 Generating Multi-Method Comparison Analysis...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Rule-based结果
    rule_counts = df_rule['fan_type'].value_counts()
    rule_counts.plot(kind='pie', ax=axes[0,0], autopct='%1.1f%%', startangle=90,
                     colors=['#FF6B6B', '#4ECDC4', '#45B7D1'])
    axes[0,0].set_title('Rule-Based Classification\n(Keyword Matching)', fontweight='bold')
    axes[0,0].set_ylabel('')
    
    # 无监督发现结果
    discovered_counts = df_main['fan_type_discovered'].value_counts()
    discovered_counts.plot(kind='pie', ax=axes[0,1], autopct='%1.1f%%', startangle=90,
                          colors=['#8B5CF6', '#06D6A0', '#F72585', '#4CC9F0'])
    axes[0,1].set_title('Data-Driven Discovery\n(Unsupervised Clustering)', fontweight='bold')
    axes[0,1].set_ylabel('')
    
    # 方法对比柱状图
    comparison_data = pd.DataFrame({
        'Rule-Based': rule_counts,
        'Data-Driven': discovered_counts
    }).fillna(0)
    
    comparison_data.plot(kind='bar', ax=axes[1,0], alpha=0.8)
    axes[1,0].set_title('Method Comparison\n(Side by Side)', fontweight='bold')
    axes[1,0].set_ylabel('Number of Comments')
    axes[1,0].tick_params(axis='x', rotation=45)
    axes[1,0].legend()
    
    # 一致性分析
    if len(df_rule) == len(df_main):
        agreement_matrix = pd.crosstab(df_rule['fan_type'], df_main['fan_type_discovered'])
        sns.heatmap(agreement_matrix, annot=True, fmt='d', cmap='Blues', ax=axes[1,1])
        axes[1,1].set_title('Method Agreement Matrix\n(Rule vs Data-Driven)', fontweight='bold')
        axes[1,1].set_xlabel('Data-Driven Classification')
        axes[1,1].set_ylabel('Rule-Based Classification')
    
    plt.suptitle('Multi-Method Classification Comparison', fontsize=18, fontweight='bold')
    plt.tight_layout()
    plt.savefig('09_multi_method_comparison.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    print("✅ Saved: 09_multi_method_comparison.png")

# ============================================
# 图表 10: BONUS - Language Pattern Analysis
# ============================================
print("\n🌐 Generating Language Pattern Analysis...")

def detect_language_simple(text):
    """简化的语言检测"""
    if pd.isna(text):
        return "unknown"
    
    text = str(text)
    korean_count = len(re.findall(r'[가-힣]', text))
    japanese_count = len(re.findall(r'[ひらがなカタカナ]', text))
    english_count = len(re.findall(r'[a-zA-Z]', text))
    
    total = korean_count + japanese_count + english_count
    if total == 0:
        return "emoji/other"
    
    korean_ratio = korean_count / total
    english_ratio = english_count / total
    
    if korean_ratio > 0.3:
        return "korean-dominant"
    elif english_ratio > 0.7:
        return "english-dominant"
    else:
        return "mixed-language"

df_main['language_pattern'] = df_main['comment_text'].apply(detect_language_simple)
lang_counts = df_main['language_pattern'].value_counts()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

# 语言分布
lang_counts.plot(kind='pie', ax=ax1, autopct='%1.1f%%', startangle=90,
                colors=['#FFB6C1', '#87CEEB', '#98FB98', '#DDA0DD'])
ax1.set_title('Language Pattern Distribution\nin Comments', fontweight='bold')
ax1.set_ylabel('')

# 语言 vs 情感
lang_emotion_cross = pd.crosstab(df_main['language_pattern'], df_main['emotion'], normalize='index') * 100
sns.heatmap(lang_emotion_cross, annot=True, fmt='.1f', cmap='YlOrRd', ax=ax2)
ax2.set_title('Language Pattern vs Emotion\n(Row-wise Percentage)', fontweight='bold')
ax2.set_xlabel('Emotion')
ax2.set_ylabel('Language Pattern')

plt.tight_layout()
plt.savefig('10_language_pattern_analysis.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()
print("✅ Saved: 10_language_pattern_analysis.png")

# ============================================
# 图表 11: BONUS - Engagement Pattern Analysis  
# ============================================
print("\n💬 Generating Engagement Pattern Analysis...")

# 创建参与度指标
df_main['engagement_score'] = (
    df_main['comment_length'] / df_main['comment_length'].max() * 50 +  # 长度权重
    (df_main['emotion'] != 'neutral').astype(int) * 30 +  # 情感表达权重
    (df_main.get('fan_type_discovered', 'general') != 'content-focused').astype(int) * 20  # 具体内容权重
)

# 参与度分布
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 参与度分布直方图
df_main['engagement_score'].hist(bins=20, alpha=0.7, color='purple', ax=axes[0,0])
axes[0,0].set_title('Comment Engagement Score Distribution', fontweight='bold')
axes[0,0].set_xlabel('Engagement Score')
axes[0,0].set_ylabel('Frequency')

# 参与度 vs 情感
engagement_by_emotion = df_main.groupby('emotion')['engagement_score'].mean().sort_values(ascending=False)
engagement_by_emotion.plot(kind='bar', ax=axes[0,1], color='orange', alpha=0.8)
axes[0,1].set_title('Average Engagement by Emotion', fontweight='bold')
axes[0,1].set_ylabel('Average Engagement Score')
axes[0,1].tick_params(axis='x', rotation=45)

# 评论长度分箱分析
df_main['length_category'] = pd.cut(df_main['comment_length'], 
                                   bins=[0, 20, 50, 100, float('inf')], 
                                   labels=['Very Short', 'Short', 'Medium', 'Long'])

length_emotion_cross = pd.crosstab(df_main['length_category'], df_main['emotion'], normalize='index') * 100
sns.heatmap(length_emotion_cross, annot=True, fmt='.1f', cmap='viridis', ax=axes[1,0])
axes[1,0].set_title('Comment Length vs Emotion\n(Row-wise %)', fontweight='bold')

# 复杂度分析（独特词汇比例）
def calculate_uniqueness(text):
    if pd.isna(text):
        return 0
    words = str(text).lower().split()
    if len(words) == 0:
        return 0
    return len(set(words)) / len(words)

df_main['uniqueness'] = df_main['comment_text'].apply(calculate_uniqueness)
uniqueness_by_emotion = df_main.groupby('emotion')['uniqueness'].mean().sort_values(ascending=False)
uniqueness_by_emotion.plot(kind='bar', ax=axes[1,1], color='green', alpha=0.8)
axes[1,1].set_title('Comment Uniqueness by Emotion\n(Unique Words Ratio)', fontweight='bold')
axes[1,1].set_ylabel('Uniqueness Score')
axes[1,1].tick_params(axis='x', rotation=45)

plt.suptitle('Advanced Engagement Pattern Analysis', fontsize=18, fontweight='bold')
plt.tight_layout()
plt.savefig('11_engagement_pattern_analysis.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()
print("✅ Saved: 11_engagement_pattern_analysis.png")

# ============================================
# 综合分析报告
# ============================================
print("\n" + "="*70)
print("📋 COMPREHENSIVE ANALYSIS REPORT")
print("="*70)

# 核心发现
total_comments = len(df_main)
positive_emotions = ['cool', 'cute', 'touching', 'inspiring', 'addictive']
positive_count = df_main[df_main['emotion'].isin(positive_emotions)].shape[0] if 'emotion' in df_main.columns else 0
positive_ratio = positive_count / total_comments * 100

print(f"\n🎯 KEY FINDINGS:")
print(f"   📊 Dataset: {total_comments} comments analyzed")
print(f"   😊 Positive sentiment: {positive_ratio:.1f}%")
print(f"   🏆 Dominant emotion: {emotion_counts.index[0]} ({emotion_counts.iloc[0]} comments)")

if has_discovered:
    discovered_specific = (df_main['fan_type_discovered'] != 'content-focused').sum()
    print(f"   🎭 Fan type diversity: {len(df_main['fan_type_discovered'].unique())} distinct patterns")
    print(f"   📈 Specific content focus: {(1-discovered_specific/total_comments)*100:.1f}%")

print(f"\n🔍 INTERESTING DISCOVERIES:")
print(f"   🎵 Song-reference comments: Found specific cluster discussing song titles/lyrics")
print(f"   💬 Lyric-quoting behavior: Fans actively quote memorable lines")  
print(f"   🎭 Meme format usage: 'No one: / Literally no one:' pattern identified")
print(f"   📝 General content dominance: Confirms K-pop as emotional entertainment")

print(f"\n📁 GENERATED VISUALIZATIONS:")
viz_files = [
    "01_emotion_distribution_pie.png - Core emotion analysis",
    "02_fan_type_distribution_bar.png - Fan behavior patterns", 
    "03_emotion_fantype_heatmap.png - Cross-correlation matrix",
    "04_comment_length_analysis.png - Engagement depth analysis",
    "05_emotion_wordcloud_grid.png - Emotion-specific language patterns",
    "06_emotion_radar_enhanced.png - Emotion profile radar",
    "07_fan_type_pie_enhanced.png - Discovered fan type distribution",
    "08_emotion_frequency_enhanced.png - Enhanced frequency analysis",
    "09_multi_method_comparison.png - Method validation (if available)",
    "10_language_pattern_analysis.png - Multilingual behavior analysis", 
    "11_engagement_pattern_analysis.png - Advanced engagement metrics"
]

for file in viz_files:
    print(f"   ✅ {file}")

print(f"\n🚀 PROJECT STATUS:")
print(f"   ✅ Phase 1 Complete: Comment sentiment & fan behavior analysis")
print(f"   🎵 Phase 2 Ready: Spotify audio features integration")
print(f"   🔗 Phase 3 Target: Audio features ↔ Fan emotion correlation")
print(f"   🎯 Phase 4 Goal: Interactive K-pop creation diagnosis platform")

print(f"\n💡 NEXT RECOMMENDED ACTION:")
print(f"   🎶 Begin Spotify Audio Features Analysis for 'How You Like That'")
print(f"   📊 Correlate audio characteristics with discovered emotion patterns")
print(f"   🔮 Build predictive model: Audio Features → Expected Fan Reactions")

print(f"\n🎉 COMPREHENSIVE ANALYSIS COMPLETE!")
print(f"Ready to proceed to the exciting Spotify integration phase! 🎵")
