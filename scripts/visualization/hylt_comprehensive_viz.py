import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from wordcloud import WordCloud
from collections import Counter
import re
import warnings
import os
warnings.filterwarnings("ignore")

# ============================================
# BLACKPINK 主题配色方案
# ============================================

BLACKPINK_COLORS = {
    'black': '#000000',
    'dark_gray': '#2d2d2d', 
    'medium_gray': '#4a4a4a',
    'light_gray': '#8a8a8a',
    'hot_pink': '#FF1493',        
    'bubblegum_pink': '#FF69B4',  
    'light_pink': '#FFB6C1',      
    'baby_pink': '#FFC0CB',       
    'rose_pink': '#FF91A4',       
    'coral_pink': '#FF6EB4',      
    'blush_pink': '#FFD1DC',      
    'magenta': '#FF0080',         
    'white': '#FFFFFF'
}

EMOTION_COLORS = [
    BLACKPINK_COLORS['hot_pink'], 
    BLACKPINK_COLORS['bubblegum_pink'],
    BLACKPINK_COLORS['light_pink'], 
    BLACKPINK_COLORS['baby_pink'],
    BLACKPINK_COLORS['rose_pink'],
    BLACKPINK_COLORS['coral_pink'],
    BLACKPINK_COLORS['blush_pink'],
    BLACKPINK_COLORS['magenta'],
    BLACKPINK_COLORS['dark_gray']
]

FAN_TYPE_COLORS = [
    BLACKPINK_COLORS['hot_pink'],
    BLACKPINK_COLORS['black'], 
    BLACKPINK_COLORS['bubblegum_pink'],
    BLACKPINK_COLORS['dark_gray'],
    BLACKPINK_COLORS['light_pink'],
    BLACKPINK_COLORS['magenta']
]

# 设置图表样式
plt.style.use('dark_background')
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'figure.facecolor': 'black',
    'axes.facecolor': 'black',
    'axes.edgecolor': BLACKPINK_COLORS['hot_pink'],
    'axes.linewidth': 1.5,
    'grid.color': BLACKPINK_COLORS['dark_gray'],
    'text.color': 'white',
    'axes.labelcolor': 'white',
    'xtick.color': BLACKPINK_COLORS['light_pink'],
    'ytick.color': BLACKPINK_COLORS['light_pink'],
    'axes.titlepad': 25,  # 增加标题间距
    'axes.labelpad': 15   # 增加轴标签间距
})

print("BLACKPINK THEMED K-POP ANALYSIS VISUALIZATION SUITE")
print("="*70)

# ============================================
# 数据加载 - 简化版
# ============================================

print("Loading analysis results...")

# 直接使用指定路径
data_path = "../../data/processed/"

try:
    df_main = pd.read_csv(data_path + "hylt_labeled_comments_discovered_types.csv")
    print("Loaded: Discovered types data")
    has_discovered = True
except:
    try:
        df_main = pd.read_csv(data_path + "labeled_comments_hylt_hf.csv")
        print("Loaded: HF emotion data")
        has_discovered = False
    except:
        print("Error: Could not find data files in", data_path)
        exit()

print(f"Total comments: {len(df_main)}")

# ============================================
# 创建输出目录
# ============================================

output_dir = "../../results/hylt"
os.makedirs(output_dir, exist_ok=True)
print(f"Output directory: {output_dir}")

# ============================================
# 图表 1: Emotion Distribution
# ============================================
print("\nGenerating Emotion Distribution...")

emotion_counts = df_main["emotion"].value_counts()
total_n = int(emotion_counts.sum())

fig, ax = plt.subplots(figsize=(12, 9), facecolor='black')

# 让第一名稍微突出，其余统一微微偏移；并设置“甜甜圈”宽度
explode = [0.10 if i == 0 else 0.04 for i in range(len(emotion_counts))]

def _autopct(pct):
    return f'{pct:.1f}%' if pct >= 4 else ''

# DONUT: 使用 wedgeprops width 形成内圈空白
wedges, texts, autotexts = ax.pie(
    emotion_counts.values,
    labels=None,
    autopct=_autopct,
    pctdistance=0.72,
    startangle=90,
    colors=EMOTION_COLORS[:len(emotion_counts)] if 'EMOTION_COLORS' in globals() else None,
    explode=explode,
    shadow=True,
    wedgeprops={'edgecolor': 'white', 'linewidth': 2, 'width': 0.42}
)

# 中心总数（两行：标题 + 数字）
ax.text(0, 0.10, 'Total', ha='center', va='center', color='white', fontsize=13, fontweight='bold')
ax.text(0, -0.12, f'{total_n}', ha='center', va='center', color='white', fontsize=22, fontweight='bold')

# 百分比文字样式
for t in ax.texts:
    t.set_color('white')
    t.set_fontweight('bold')
    t.set_fontsize(11)

# 右侧图例
legend = ax.legend(
    wedges,
    [str(x) for x in emotion_counts.index],
    title='Emotion',
    loc='center left',
    bbox_to_anchor=(1.02, 0.5),
    frameon=True
)
legend.get_frame().set_facecolor('black')
legend.get_frame().set_edgecolor('white')
legend.get_frame().set_alpha(0.6)
for text in legend.get_texts():
    text.set_color('white')
    text.set_fontweight('bold')

ax.set_title('BLACKPINK Fan Emotion Distribution How You Like That Analysis',
             fontsize=18, fontweight='bold', color='white', pad=30)
ax.set_aspect('equal')

plt.tight_layout()
plt.savefig(f'{output_dir}/BP_01_emotion_distribution_donut.png', dpi=300,
            bbox_inches='tight', facecolor='black', edgecolor='none')
plt.show()
print(f"Saved: {output_dir}/BP_01_emotion_distribution_donut.png")

# ============================================
# 图表 2: Fan Type Distribution
# ============================================
print("\nGenerating Fan Type Distribution...")

if has_discovered:
    fan_type_counts = df_main["fan_type_discovered"].value_counts()
    title_suffix = "Data-Driven Discovery"
else:
    fan_type_counts = df_main.get("fan_type", pd.Series(["general-praise"] * len(df_main))).value_counts()
    title_suffix = "Fan Analysis"

fig, ax = plt.subplots(figsize=(14, 9), facecolor='black')
ax.set_facecolor('black')

bars = ax.bar(range(len(fan_type_counts)), fan_type_counts.values, 
              color=FAN_TYPE_COLORS[:len(fan_type_counts)], 
              alpha=0.9, width=0.7,
              edgecolor='white', linewidth=2)

# 添加数值标签，调整位置避免重叠
for i, (idx, value) in enumerate(fan_type_counts.items()):
    percentage = value/len(df_main)*100
    ax.text(i, value + len(df_main)*0.03, f'{value}\n({percentage:.1f}%)', 
            ha='center', va='bottom', fontweight='bold', 
            color='white', fontsize=10)

ax.set_title(f'BLACKPINK Fan Type Distribution\n{title_suffix}', 
             fontsize=16, fontweight='bold', color='white', pad=35)
ax.set_xlabel('Fan Type Categories', fontsize=12, color='white', fontweight='bold')
ax.set_ylabel('Number of Comments', fontsize=12, color='white', fontweight='bold')
ax.set_xticks(range(len(fan_type_counts)))
ax.set_xticklabels(fan_type_counts.index, rotation=45, ha='right', color='white', fontweight='bold')
ax.grid(True, alpha=0.3, color=BLACKPINK_COLORS['dark_gray'])

# 调整布局避免重叠
plt.subplots_adjust(bottom=0.2, top=0.85)
plt.tight_layout()
plt.savefig(f'{output_dir}/BP_02_fan_type_distribution.png', dpi=300, bbox_inches='tight', 
            facecolor='black', edgecolor='none')
plt.show()
print(f"Saved: {output_dir}/BP_02_fan_type_distribution.png")

# ============================================
# 图表 3: Correlation Heatmap
# ============================================
print("\nGenerating Correlation Heatmap...")

fig, ax = plt.subplots(figsize=(12, 9), facecolor='black')

if has_discovered:
    cross_table = pd.crosstab(df_main['emotion'], df_main['fan_type_discovered'], normalize='index') * 100
else:
    cross_table = pd.crosstab(df_main['emotion'], df_main.get('fan_type', 'general'), normalize='index') * 100

# 自定义热力图色彩
colors_heatmap = ['#000000', '#2d2d2d', '#FF1493', '#FF69B4', '#FFB6C1']
from matplotlib.colors import LinearSegmentedColormap
blackpink_cmap = LinearSegmentedColormap.from_list("blackpink", colors_heatmap)

sns.heatmap(cross_table, annot=True, fmt='.1f', cmap=blackpink_cmap,
            cbar_kws={'label': 'Percentage (%)'}, linewidths=1,
            annot_kws={'color': 'white', 'fontweight': 'bold', 'fontsize': 9},
            ax=ax)

ax.set_title('Emotion vs Fan Type Correlation Matrix\nBLACKPINK Analysis', 
          fontsize=16, fontweight='bold', color='white', pad=30)
ax.set_xlabel('Fan Type', fontsize=12, color='white', fontweight='bold')
ax.set_ylabel('Emotion', fontsize=12, color='white', fontweight='bold')
plt.xticks(rotation=45, ha='right', color='white')
plt.yticks(rotation=0, color='white')

plt.subplots_adjust(bottom=0.25, top=0.85)
plt.tight_layout()
plt.savefig(f'{output_dir}/BP_03_emotion_correlation.png', dpi=300, bbox_inches='tight', 
            facecolor='black', edgecolor='none')
plt.show()
print(f"Saved: {output_dir}/BP_03_emotion_correlation.png")

# ============================================
# 图表 4: Comment Length Analysis
# ============================================
print("\nGenerating Comment Length Analysis...")

df_main['comment_length'] = df_main['comment_text'].str.len()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), facecolor='black')
for ax in [ax1, ax2]:
    ax.set_facecolor('black')

# 长度分布直方图
ax1.hist(df_main['comment_length'], bins=30, alpha=0.8, 
         color=BLACKPINK_COLORS['hot_pink'], edgecolor='white', linewidth=1)

mean_length = df_main['comment_length'].mean()
median_length = df_main['comment_length'].median()

ax1.axvline(mean_length, color=BLACKPINK_COLORS['bubblegum_pink'], 
           linestyle='--', linewidth=3, label=f'Mean: {mean_length:.1f}')
ax1.axvline(median_length, color=BLACKPINK_COLORS['light_pink'], 
           linestyle='--', linewidth=3, label=f'Median: {median_length:.1f}')

ax1.set_title('Comment Length Distribution', fontsize=14, fontweight='bold', color='white', pad=20)
ax1.set_xlabel('Character Count', color='white', fontweight='bold')
ax1.set_ylabel('Frequency', color='white', fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3, color=BLACKPINK_COLORS['dark_gray'])

# 情感vs长度箱线图
emotion_order = emotion_counts.index.tolist()
box_plot = ax2.boxplot([df_main[df_main['emotion'] == emotion]['comment_length'].values 
                       for emotion in emotion_order],
                      labels=emotion_order, patch_artist=True)

# BLACKPINK风格箱线图
for patch, color in zip(box_plot['boxes'], EMOTION_COLORS):
    patch.set_facecolor(color)
    patch.set_alpha(0.8)

ax2.set_title('Comment Length by Emotion', fontsize=14, fontweight='bold', color='white', pad=20)
ax2.set_xlabel('Emotion', color='white', fontweight='bold')
ax2.set_ylabel('Character Count', color='white', fontweight='bold')
ax2.tick_params(axis='x', rotation=45, colors='white')
ax2.grid(True, alpha=0.3, color=BLACKPINK_COLORS['dark_gray'])

plt.subplots_adjust(bottom=0.2, top=0.85)
plt.tight_layout()
plt.savefig(f'{output_dir}/BP_04_comment_length_analysis.png', dpi=300, bbox_inches='tight', 
            facecolor='black', edgecolor='none')
plt.show()
print(f"Saved: {output_dir}/BP_04_comment_length_analysis.png")

# ============================================
# 图表 5: Word Clouds
# ============================================
print("\nGenerating Word Clouds...")

top_emotions = emotion_counts.head(4).index
fig, axes = plt.subplots(2, 2, figsize=(16, 12), facecolor='black')
axes = axes.flatten()

for ax in axes:
    ax.set_facecolor('black')

def blackpink_color_func(word, font_size, position, orientation, random_state=None, **kwargs):
    colors = ['#FF1493', '#FF69B4', '#FFB6C1', '#FF91A4', '#FF6EB4', '#FFC0CB']
    return np.random.choice(colors)

for i, emotion in enumerate(top_emotions):
    emotion_comments = df_main[df_main['emotion'] == emotion]['comment_text'].tolist()
    text = ' '.join([str(comment) for comment in emotion_comments])
    
    # 清理文本
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    
    if len(text.strip()) > 0:
        wordcloud = WordCloud(
            width=400, height=300,
            background_color='black',
            max_words=60,
            color_func=blackpink_color_func,
            relative_scaling=0.6,
            prefer_horizontal=0.7
        ).generate(text)
        
        axes[i].imshow(wordcloud, interpolation='bilinear')
        axes[i].set_title(f'{emotion.title()} Words\n({emotion_counts[emotion]} comments)', 
                         fontsize=12, fontweight='bold', color='white', pad=15)
        axes[i].axis('off')

plt.suptitle('BLACKPINK Emotion Word Clouds', 
             fontsize=18, fontweight='bold', color='white', y=0.95)
plt.subplots_adjust(top=0.9, hspace=0.3)
plt.tight_layout()
plt.savefig(f'{output_dir}/BP_05_emotion_wordclouds.png', dpi=300, bbox_inches='tight', 
            facecolor='black', edgecolor='none')
plt.show()
print(f"Saved: {output_dir}/BP_05_emotion_wordclouds.png")

# ============================================
# 图表 6: Radar Chart
# ============================================
print("\nGenerating Radar Chart...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8), 
                              subplot_kw=dict(projection='polar'), facecolor='black')

for ax in [ax1, ax2]:
    ax.set_facecolor('black')

# 情感雷达图
emotions = emotion_counts.index.tolist()
values = emotion_counts.values.tolist()
values_pct = [v/sum(values)*100 for v in values]

angles = np.linspace(0, 2 * np.pi, len(emotions), endpoint=False).tolist()
values_pct += values_pct[:1]
angles += angles[:1]

ax1.plot(angles, values_pct, 'o-', linewidth=4, color=BLACKPINK_COLORS['hot_pink'], 
         markersize=10, markerfacecolor=BLACKPINK_COLORS['bubblegum_pink'],
         markeredgecolor='white', markeredgewidth=2)
ax1.fill(angles, values_pct, alpha=0.3, color=BLACKPINK_COLORS['hot_pink'])

ax1.set_xticks(angles[:-1])
ax1.set_xticklabels([emotion.title() for emotion in emotions], 
                    fontsize=10, color='white', fontweight='bold')
ax1.set_ylim(0, max(values_pct) * 1.2)
ax1.set_title('How You Like That Emotion Profile', 
              fontsize=14, fontweight='bold', color='white', pad=30)
ax1.grid(True, color=BLACKPINK_COLORS['dark_gray'], alpha=0.7)

# 情感强度雷达图
weighted_emotions = {}
for emotion in emotions:
    emotion_df = df_main[df_main['emotion'] == emotion]
    avg_length = emotion_df['comment_length'].mean() if len(emotion_df) > 0 else 0
    weighted_emotions[emotion] = avg_length

weighted_values = list(weighted_emotions.values())
max_weight = max(weighted_values) if weighted_values else 1
weighted_values_norm = [v/max_weight*100 for v in weighted_values]
weighted_values_norm += weighted_values_norm[:1]

ax2.plot(angles, weighted_values_norm, 'o-', linewidth=4, 
         color=BLACKPINK_COLORS['bubblegum_pink'], markersize=10,
         markerfacecolor=BLACKPINK_COLORS['light_pink'],
         markeredgecolor='white', markeredgewidth=2)
ax2.fill(angles, weighted_values_norm, alpha=0.3, color=BLACKPINK_COLORS['bubblegum_pink'])

ax2.set_xticks(angles[:-1])
ax2.set_xticklabels([emotion.title() for emotion in emotions], 
                    fontsize=10, color='white', fontweight='bold')
ax2.set_ylim(0, max(weighted_values_norm) * 1.2)
ax2.set_title('Emotion Intensity by Engagement', 
              fontsize=14, fontweight='bold', color='white', pad=30)
ax2.grid(True, color=BLACKPINK_COLORS['dark_gray'], alpha=0.7)

plt.subplots_adjust(top=0.85)
plt.tight_layout()
plt.savefig(f'{output_dir}/BP_06_emotion_radar.png', dpi=300, bbox_inches='tight', 
            facecolor='black', edgecolor='none')
plt.show()
print(f"Saved: {output_dir}/BP_06_emotion_radar.png")

# ============================================
# 图表 7: Fan Type Pie
# ============================================
print("\nGenerating Fan Type Pie...")

fig, ax = plt.subplots(figsize=(12, 10), facecolor='black')

if has_discovered:
    fan_data = df_main["fan_type_discovered"].value_counts()
    title = "BLACKPINK Fan Types"
else:
    fan_data = df_main.get("fan_type", pd.Series(["general-content"] * len(df_main))).value_counts()
    title = "Fan Type Analysis"

explode = [0.1 if fan_data.iloc[i] < len(df_main)*0.15 else 0.05 for i in range(len(fan_data))]

wedges, texts, autotexts = plt.pie(
    fan_data.values, 
    labels=fan_data.index,
    autopct='%1.1f%%',
    startangle=90,
    colors=FAN_TYPE_COLORS[:len(fan_data)],
    explode=explode,
    shadow=True,
    wedgeprops={'edgecolor': 'white', 'linewidth': 2}
)

for text in texts:
    text.set_color('white')
    text.set_fontweight('bold')
    text.set_fontsize(11)

for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontweight('bold')
    autotext.set_fontsize(10)

plt.title(f'{title}\nData-Driven Discovery', 
          fontsize=16, fontweight='bold', color='white', pad=40)
plt.axis('equal')
plt.subplots_adjust(top=0.85)
plt.tight_layout()
plt.savefig(f'{output_dir}/BP_07_fan_type_pie.png', dpi=300, bbox_inches='tight', 
            facecolor='black', edgecolor='none')
plt.show()
print(f"Saved: {output_dir}/BP_07_fan_type_pie.png")

# ============================================
# 图表 8: Enhanced Frequency Analysis
# ============================================
print("\nGenerating Enhanced Frequency Analysis...")

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12), facecolor='black')
for ax in [ax1, ax2]:
    ax.set_facecolor('black')

# 情感频次
bars1 = ax1.bar(range(len(emotion_counts)), emotion_counts.values,
                color=EMOTION_COLORS[:len(emotion_counts)], alpha=0.9,
                edgecolor='white', linewidth=2)

for i, (emotion, count) in enumerate(emotion_counts.items()):
    percentage = count/len(df_main)*100
    ax1.text(i, count + len(df_main)*0.03, f'{count}\n({percentage:.1f}%)', 
             ha='center', va='bottom', fontweight='bold', color='white', fontsize=10)

ax1.set_title('BLACKPINK Emotion Frequency Distribution', 
              fontsize=16, fontweight='bold', color='white', pad=25)
ax1.set_xticks(range(len(emotion_counts)))
ax1.set_xticklabels(emotion_counts.index, rotation=45, ha='right', color='white', fontweight='bold')
ax1.set_ylabel('Number of Comments', color='white', fontweight='bold')
ax1.grid(True, alpha=0.3, color=BLACKPINK_COLORS['dark_gray'])

# 平均长度
emotion_avg_length = df_main.groupby('emotion')['comment_length'].mean().reindex(emotion_counts.index)
bars2 = ax2.bar(range(len(emotion_avg_length)), emotion_avg_length.values,
                color=EMOTION_COLORS[:len(emotion_avg_length)], alpha=0.7,
                edgecolor='white', linewidth=2)

for i, (emotion, avg_len) in enumerate(emotion_avg_length.items()):
    ax2.text(i, avg_len + 4, f'{avg_len:.0f}', ha='center', va='bottom', 
             fontweight='bold', color='white', fontsize=10)

ax2.set_title('Average Comment Length by Emotion', 
              fontsize=16, fontweight='bold', color='white', pad=25)
ax2.set_xticks(range(len(emotion_avg_length)))
ax2.set_xticklabels(emotion_avg_length.index, rotation=45, ha='right', color='white', fontweight='bold')
ax2.set_ylabel('Average Character Count', color='white', fontweight='bold')
ax2.grid(True, alpha=0.3, color=BLACKPINK_COLORS['dark_gray'])

plt.subplots_adjust(bottom=0.15, top=0.9, hspace=0.4)
plt.tight_layout()
plt.savefig(f'{output_dir}/BP_08_emotion_frequency_enhanced.png', dpi=300, bbox_inches='tight', 
            facecolor='black', edgecolor='none')
plt.show()
print(f"Saved: {output_dir}/BP_08_emotion_frequency_enhanced.png")

# ============================================
# 最终报告
# ============================================
print("\n" + "="*70)
print("BLACKPINK THEMED ANALYSIS COMPLETE")
print("="*70)

total_comments = len(df_main)
positive_emotions = ['cool', 'cute', 'touching', 'inspiring', 'addictive']
positive_count = df_main[df_main['emotion'].isin(positive_emotions)].shape[0]
positive_ratio = positive_count / total_comments * 100

print(f"\nKEY FINDINGS:")
print(f"   Total Comments Analyzed: {total_comments}")
print(f"   Positive Sentiment Rate: {positive_ratio:.1f}%")
print(f"   Dominant Emotion: {emotion_counts.index[0]} ({emotion_counts.iloc[0]} comments)")

print(f"\nGENERATED VISUALIZATIONS:")
blackpink_files = [
    "BP_01_emotion_distribution.png - Emotion pie chart",
    "BP_02_fan_type_distribution.png - Fan type analysis", 
    "BP_03_emotion_correlation.png - Correlation heatmap",
    "BP_04_comment_length_analysis.png - Length distribution",
    "BP_05_emotion_wordclouds.png - Word cloud grid",
    "BP_06_emotion_radar.png - Dual radar charts",
    "BP_07_fan_type_pie.png - Enhanced fan type pie",
    "BP_08_emotion_frequency_enhanced.png - Frequency analysis"
]

for file in blackpink_files:
    print(f"   ✅ {output_dir}/{file}")

print(f"\nREADY FOR SPOTIFY INTEGRATION:")
print(f"   Next phase: Audio features analysis with BLACKPINK aesthetic")
print(f"   Correlate audio characteristics with fan emotional responses")
print(f"   Build 'How You Like That' audio-emotion prediction model")

print(f"\nHOW YOU LIKE THAT DATA ANALYSIS COMPLETE!")

