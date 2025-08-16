import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import Counter
import re
from wordcloud import WordCloud
import warnings
warnings.filterwarnings('ignore')

# 🎨 NewJeans风格配色设置
plt.style.use('default')  # 重置为默认样式

# NewJeans蓝白清爽配色方案
NJ_COLORS = {
    'primary_blue': '#4A90E2',
    'light_blue': '#74B9FF',
    'soft_blue': '#A8D8FF',
    'pale_blue': '#E3F2FD',
    'white': '#FFFFFF',
    'light_gray': '#F8FAFC',
    'text_dark': '#2C3E50',
    'text_light': '#7F8C8D'
}

# 专业配色板
NJ_PALETTE = [NJ_COLORS['primary_blue'], NJ_COLORS['light_blue'], NJ_COLORS['soft_blue'], 
              NJ_COLORS['pale_blue'], '#B8E6B8', '#FFD93D', '#FF6B9D', '#C44569']

# 设置全局字体和样式
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Segoe UI', 'Arial', 'DejaVu Sans'],
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 16,
    'axes.facecolor': NJ_COLORS['white'],
    'figure.facecolor': NJ_COLORS['white'],
    'axes.edgecolor': NJ_COLORS['light_gray'],
    'axes.linewidth': 0.8,
    'grid.alpha': 0.3,
    'grid.linewidth': 0.5
})

# 🎯 加载分析结果
df = pd.read_csv("../analysis/ditto_sentiment_analysis_results.csv")

print("🎨 Creating NewJeans-Style Professional Visualizations")
print("=" * 60)
print(f"📊 Total comments analyzed: {len(df)}")

def setup_clean_plot(figsize=(12, 8)):
    """设置清爽的图表基础样式"""
    fig, ax = plt.subplots(figsize=figsize, facecolor='white')
    ax.set_facecolor('white')
    ax.grid(True, alpha=0.2, color=NJ_COLORS['light_gray'])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(NJ_COLORS['text_light'])
    ax.spines['bottom'].set_color(NJ_COLORS['text_light'])
    return fig, ax

def add_clean_title(ax, title, subtitle=None):
    """添加清爽的标题样式"""
    ax.set_title(title, fontsize=16, fontweight='600', color=NJ_COLORS['text_dark'], 
                pad=30, loc='center')
    if subtitle:
        ax.text(0.5, 1.12, subtitle, transform=ax.transAxes, 
                fontsize=11, color=NJ_COLORS['text_light'], 
                ha='center', style='italic')

# ================================
# 📊 图表1：情感分布饼图 (改进版)
# ================================
def create_sentiment_pie_chart():
    fig, ax = plt.subplots(figsize=(12, 10), facecolor='white')
    
    sentiment_counts = df['sentiment'].value_counts()
    
    # 使用清爽的蓝色系配色
    colors = [NJ_COLORS['primary_blue'], NJ_COLORS['light_blue'], NJ_COLORS['soft_blue'], 
              NJ_COLORS['pale_blue'], '#E8F4FD']
    
    # 创建饼图
    wedges, texts, autotexts = ax.pie(sentiment_counts.values, 
                                     labels=sentiment_counts.index,
                                     autopct='%1.1f%%',
                                     colors=colors,
                                     startangle=90,
                                     explode=[0.08 if x == 'melancholy_positive' else 0.02 for x in sentiment_counts.index],
                                     shadow=True,
                                     textprops={'fontsize': 11, 'fontweight': '500'})
    
    # 美化文本
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(12)
    
    for text in texts:
        text.set_fontsize(12)
        text.set_fontweight('600')
        text.set_color(NJ_COLORS['text_dark'])
    
    # 添加标题
    add_clean_title(ax, 'Sentiment Distribution Analysis', 
                   'Overall emotional response patterns across 905 comments')
    
    # 添加统计信息框
    total_comments = len(df)
    stats_text = f"""Total Analyzed: {total_comments:,} comments
Dominant: {sentiment_counts.index[0]} ({sentiment_counts.iloc[0]/total_comments*100:.1f}%)
Unique Emotion: melancholy_positive ({sentiment_counts.get('melancholy_positive', 0)} comments)"""
    
    ax.text(1.15, 0.9, stats_text, transform=ax.transAxes, 
            fontsize=11, bbox=dict(boxstyle="round,pad=0.6", 
                                  facecolor=NJ_COLORS['pale_blue'], 
                                  alpha=0.8, edgecolor=NJ_COLORS['light_blue']),
            verticalalignment='top')
    
    plt.tight_layout(rect=[0, 0, 0.85, 1])  # 为右侧信息留出空间
    plt.savefig('ditto_sentiment_distribution.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none', pad_inches=0.2)
    plt.show()
    print("✅ Sentiment pie chart saved")

# ================================
# 📊 图表2：粉丝类型分布条形图 (改进版)
# ================================
def create_fan_type_bar_chart():
    fig, ax = setup_clean_plot(figsize=(14, 8))
    
    fan_type_counts = df['comment_type'].value_counts()
    
    # 创建水平条形图
    bars = ax.barh(range(len(fan_type_counts)), fan_type_counts.values, 
                   color=NJ_COLORS['primary_blue'], alpha=0.8, height=0.6)
    
    # 渐变效果
    for i, bar in enumerate(bars):
        bar.set_color(NJ_PALETTE[i % len(NJ_PALETTE)])
        bar.set_alpha(0.85)
    
    # 添加数值标签
    for i, (bar, value) in enumerate(zip(bars, fan_type_counts.values)):
        ax.text(value + max(fan_type_counts.values) * 0.01, bar.get_y() + bar.get_height()/2, 
                f'{value} ({value/len(df)*100:.1f}%)',
                va='center', ha='left', fontweight='600', 
                color=NJ_COLORS['text_dark'], fontsize=11)
    
    ax.set_yticks(range(len(fan_type_counts)))
    ax.set_yticklabels(fan_type_counts.index, fontweight='500')
    ax.set_xlabel('Number of Comments', fontweight='600', color=NJ_COLORS['text_dark'])
    ax.set_ylabel('Fan Engagement Type', fontweight='600', color=NJ_COLORS['text_dark'])
    
    add_clean_title(ax, 'Fan Engagement Types Distribution', 
                   'How fans interact with complex narrative content')
    
    # 添加洞察
    mv_analysis_pct = fan_type_counts.get('mv_analysis', 0) / len(df) * 100
    insight_text = f"🎬 MV Analysis: {mv_analysis_pct:.1f}% - Exceptionally high for K-pop!"
    ax.text(0.02, 0.98, insight_text, transform=ax.transAxes, 
            fontsize=12, fontweight='bold', color=NJ_COLORS['primary_blue'],
            bbox=dict(boxstyle="round,pad=0.5", facecolor=NJ_COLORS['pale_blue'], alpha=0.8),
            verticalalignment='top')
    
    ax.set_xlim(0, max(fan_type_counts.values) * 1.25)  # 增加右侧空间
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.savefig('ditto_fan_types_distribution.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none', pad_inches=0.2)
    plt.show()
    print("✅ Fan types bar chart saved")

# ================================
# 📊 图表3：情感-粉丝类型关联热力图 (改进版)
# ================================
def create_sentiment_fantype_heatmap():
    fig, ax = setup_clean_plot(figsize=(12, 8))
    
    # 创建交叉表
    crosstab = pd.crosstab(df['comment_type'], df['sentiment'], normalize='index') * 100
    
    # 使用蓝色系热力图
    sns.heatmap(crosstab, 
                annot=True, 
                fmt='.1f', 
                cmap='Blues',
                cbar_kws={'label': 'Percentage (%)', 'shrink': 0.8},
                square=False,
                linewidths=1,
                linecolor='white',
                ax=ax,
                annot_kws={'fontsize': 10, 'fontweight': '600'})
    
    add_clean_title(ax, 'Sentiment Patterns Across Fan Types', 
                   'Emotional response distribution by comment category')
    
    ax.set_xlabel('Sentiment Type', fontweight='600', color=NJ_COLORS['text_dark'])
    ax.set_ylabel('Fan Engagement Type', fontweight='600', color=NJ_COLORS['text_dark'])
    
    # 旋转标签
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    # 添加洞察
    mv_melancholy = crosstab.loc['mv_analysis', 'melancholy_positive'] if 'mv_analysis' in crosstab.index and 'melancholy_positive' in crosstab.columns else 0
    insight_text = f"💡 MV Analysis shows {mv_melancholy:.1f}% melancholy_positive"
    ax.text(1.05, 0.98, insight_text, transform=ax.transAxes, 
            fontsize=11, fontweight='bold', color=NJ_COLORS['primary_blue'],
            bbox=dict(boxstyle="round,pad=0.5", facecolor=NJ_COLORS['pale_blue'], alpha=0.8),
            verticalalignment='top', rotation=0)
    
    plt.tight_layout(rect=[0, 0, 0.9, 1])
    plt.savefig('ditto_sentiment_fantype_heatmap.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none', pad_inches=0.2)
    plt.show()
    print("✅ Sentiment-fantype heatmap saved")

# ================================
# 📊 图表4：评论长度分布图 (改进版)
# ================================
def create_comment_length_distribution():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), facecolor='white')
    fig.subplots_adjust(wspace=0.3)  # 增加水平间距
    
    # 左图：整体长度分布
    ax1.hist(df['comment_length'], bins=25, color=NJ_COLORS['primary_blue'], 
             alpha=0.7, edgecolor='white', linewidth=1.2)
    
    mean_length = df['comment_length'].mean()
    median_length = df['comment_length'].median()
    
    ax1.axvline(mean_length, color=NJ_COLORS['light_blue'], linestyle='--', linewidth=2.5, 
                label=f'Mean: {mean_length:.0f} chars')
    ax1.axvline(median_length, color='#FF6B9D', linestyle='--', linewidth=2.5, 
                label=f'Median: {median_length:.0f} chars')
    
    ax1.set_title('Comment Length Distribution', fontsize=14, fontweight='600', 
                  color=NJ_COLORS['text_dark'], pad=20)
    ax1.set_xlabel('Comment Length (characters)', fontweight='600', color=NJ_COLORS['text_dark'])
    ax1.set_ylabel('Frequency', fontweight='600', color=NJ_COLORS['text_dark'])
    ax1.legend(loc='upper right', frameon=True, fancybox=True, shadow=True,
               bbox_to_anchor=(0.98, 0.98))
    ax1.grid(True, alpha=0.2)
    
    # 右图：按类型的长度分布
    type_data = [df[df['comment_type'] == ctype]['comment_length'].values 
                 for ctype in df['comment_type'].unique()]
    type_labels = df['comment_type'].unique()
    
    bp = ax2.boxplot(type_data, labels=type_labels, patch_artist=True, 
                     boxprops=dict(facecolor=NJ_COLORS['light_blue'], alpha=0.7),
                     medianprops=dict(color=NJ_COLORS['primary_blue'], linewidth=2),
                     whiskerprops=dict(color=NJ_COLORS['text_dark']),
                     capprops=dict(color=NJ_COLORS['text_dark']))
    
    # 为每个箱子设置不同颜色
    for patch, color in zip(bp['boxes'], NJ_PALETTE):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax2.set_title('Length Distribution by Fan Type', fontsize=14, fontweight='600', 
                  color=NJ_COLORS['text_dark'], pad=20)
    ax2.set_xlabel('Fan Engagement Type', fontweight='600', color=NJ_COLORS['text_dark'])
    ax2.set_ylabel('Comment Length (characters)', fontweight='600', color=NJ_COLORS['text_dark'])
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.2)
    
    plt.tight_layout()
    plt.savefig('ditto_comment_length_analysis.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none', pad_inches=0.2)
    plt.show()
    print("✅ Comment length analysis saved")

# ================================
# 📊 图表5：情绪词云 (改进版)
# ================================
def create_emotion_wordcloud():
    # 提取所有评论文本
    all_text = ' '.join(df['comment_text'].astype(str))
    
    # 扩展的情感关键词
    emotion_keywords = [
        'love', 'amazing', 'beautiful', 'perfect', 'gorgeous', 'talent', 'queen',
        'nostalgia', 'nostalgic', 'memory', 'childhood', 'youth', 'miss',
        'winter', 'cozy', 'warm', 'comfort', 'peaceful', 'calm',
        'genius', 'masterpiece', 'art', 'cinematic', 'story', 'concept',
        'sad', 'crying', 'tears', 'heartbreak', 'lonely', 'melancholy',
        'bittersweet', 'emotional', 'deep', 'thoughtful', 'complex',
        'ditto', 'heesoo', 'side', 'theory', 'meaning', 'analysis',
        'dreamy', 'ethereal', 'aesthetic', 'vibe', 'mood'
    ]
    
    # 文本预处理
    words = re.findall(r'\b[a-zA-Z]{3,}\b', all_text.lower())
    emotion_words = [word for word in words if word in emotion_keywords]
    word_freq = Counter(emotion_words)
    
    if len(word_freq) > 0:
        fig, ax = plt.subplots(figsize=(16, 10), facecolor='white')
        
        # 创建蓝色系词云
        def nj_color_func(*args, **kwargs):
            colors = [NJ_COLORS['primary_blue'], NJ_COLORS['light_blue'], 
                     NJ_COLORS['soft_blue'], '#4169E1', '#1E90FF']
            return np.random.choice(colors)
        
        wordcloud = WordCloud(width=1200, height=700, 
                             background_color='white',
                             color_func=nj_color_func,
                             max_words=80,
                             relative_scaling=0.6,
                             random_state=42,
                             font_path=None,
                             prefer_horizontal=0.8).generate_from_frequencies(word_freq)
        
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        
        add_clean_title(ax, 'Emotional Keywords Cloud', 
                       'Most frequently used emotion-related terms in fan comments')
        
        # 添加统计
        top_5_words = word_freq.most_common(5)
        stats_text = "Top Emotional Terms:\n" + "\n".join([f"• {word}: {count}" for word, count in top_5_words])
        ax.text(1.05, 0.98, stats_text, transform=ax.transAxes, 
                fontsize=12, bbox=dict(boxstyle="round,pad=0.6", 
                                      facecolor=NJ_COLORS['pale_blue'], alpha=0.9),
                verticalalignment='top')
        
        plt.tight_layout(rect=[0, 0, 0.88, 1])
        plt.savefig('ditto_emotion_wordcloud.png', dpi=300, bbox_inches='tight',
                    facecolor='white', edgecolor='none', pad_inches=0.2)
        plt.show()
        print("✅ Emotion wordcloud saved")
    else:
        print("⚠️ Insufficient emotion keywords found")

# ================================
# 📊 图表6：情绪雷达图 (改进版)
# ================================
def create_emotion_radar_chart():
    sentiment_counts = df['sentiment'].value_counts(normalize=True) * 100
    
    categories = list(sentiment_counts.index)
    values = list(sentiment_counts.values)
    
    # 闭合雷达图
    categories += categories[:1]
    values += values[:1]
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'), 
                          facecolor='white')
    
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=True)
    
    # 绘制雷达图
    ax.plot(angles, values, 'o-', linewidth=3, color=NJ_COLORS['primary_blue'], 
            markersize=8, alpha=0.8)
    ax.fill(angles, values, alpha=0.2, color=NJ_COLORS['light_blue'])
    
    # 设置标签
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories[:-1], fontsize=12, fontweight='600',
                       color=NJ_COLORS['text_dark'])
    
    # 设置数值范围
    max_val = max(values)
    ax.set_ylim(0, max_val + 5)
    ax.set_yticks(np.arange(0, max_val + 5, max_val/4))
    ax.set_yticklabels([f'{int(y)}%' for y in ax.get_yticks()], 
                       fontsize=10, color=NJ_COLORS['text_light'])
    
    # 美化网格
    ax.grid(True, alpha=0.3, color=NJ_COLORS['light_blue'])
    ax.set_facecolor('white')
    
    plt.title('Emotional Profile Radar Chart\nOverall sentiment distribution pattern', 
              fontsize=16, fontweight='600', color=NJ_COLORS['text_dark'], 
              pad=40, loc='center')
    
    # 添加统计
    dominant_emotion = sentiment_counts.index[0]
    stats_text = f"""Dominant: {dominant_emotion} ({sentiment_counts.iloc[0]:.1f}%)
Unique: melancholy_positive ({sentiment_counts.get('melancholy_positive', 0):.1f}%)
Emotions: {len([x for x in sentiment_counts if x > 5])} main categories"""
    
    plt.figtext(0.02, 0.20, stats_text, fontsize=11, 
                bbox=dict(boxstyle="round,pad=0.6", facecolor=NJ_COLORS['pale_blue'], alpha=0.9))
    
    plt.tight_layout()
    plt.savefig('ditto_emotion_radar.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none', pad_inches=0.2)
    plt.show()
    print("✅ Emotion radar chart saved")

# ================================
# 📊 图表7：情绪频次条形图 (改进版)
# ================================
def create_emotion_frequency_chart():
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12), facecolor='white')
    fig.subplots_adjust(hspace=0.4)  # 增加垂直间距
    
    # 上图：情感频次
    sentiment_counts = df['sentiment'].value_counts()
    bars1 = ax1.bar(sentiment_counts.index, sentiment_counts.values, 
                    color=[NJ_PALETTE[i] for i in range(len(sentiment_counts))],
                    alpha=0.8, edgecolor='white', linewidth=1.5)
    
    # 添加数值标签
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + max(sentiment_counts.values) * 0.01,
                f'{int(height)}', ha='center', va='bottom', fontweight='bold',
                color=NJ_COLORS['text_dark'], fontsize=11)
    
    ax1.set_title('Sentiment Frequency Distribution', fontsize=14, fontweight='600', 
                  color=NJ_COLORS['text_dark'], pad=20)
    ax1.set_xlabel('Sentiment Type', fontweight='600', color=NJ_COLORS['text_dark'])
    ax1.set_ylabel('Number of Comments', fontweight='600', color=NJ_COLORS['text_dark'])
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.2)
    
    # 下图：按质量分组
    df['quality_group'] = pd.cut(df['quality_score'], 
                                bins=[0, 40, 60, 100], 
                                labels=['Low (0-40)', 'Medium (40-60)', 'High (60-100)'])
    
    quality_sentiment = df.groupby(['quality_group', 'sentiment']).size().unstack(fill_value=0)
    
    # 绘制堆叠条形图
    quality_sentiment.plot(kind='bar', ax=ax2, stacked=True, 
                          color=NJ_PALETTE[:len(quality_sentiment.columns)],
                          alpha=0.8, edgecolor='white', linewidth=1)
    
    ax2.set_title('Sentiment by Comment Quality', fontsize=14, fontweight='600', 
                  color=NJ_COLORS['text_dark'], pad=20)
    ax2.set_xlabel('Comment Quality Group', fontweight='600', color=NJ_COLORS['text_dark'])
    ax2.set_ylabel('Number of Comments', fontweight='600', color=NJ_COLORS['text_dark'])
    ax2.legend(title='Sentiment', bbox_to_anchor=(1.02, 1), loc='upper left',
               frameon=True, fancybox=True, shadow=True)
    ax2.tick_params(axis='x', rotation=0)
    ax2.grid(True, alpha=0.2)
    
    plt.tight_layout(rect=[0, 0, 0.92, 1])  # 为legend留出空间
    plt.savefig('ditto_emotion_frequency.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none', pad_inches=0.2)
    plt.show()
    print("✅ Emotion frequency chart saved")

# ================================
# 📊 图表8：综合统计面板 (改进版)
# ================================
def create_comprehensive_stats_panel():
    fig = plt.figure(figsize=(16, 12), facecolor='white')
    
    # 创建2x2布局，增加间距
    gs = fig.add_gridspec(2, 2, hspace=0.4, wspace=0.35)
    
    # 图1：质量分数分布
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(df['quality_score'], bins=20, color=NJ_COLORS['primary_blue'], 
             alpha=0.7, edgecolor='white', linewidth=1.2)
    ax1.axvline(df['quality_score'].mean(), color='#FF6B9D', linestyle='--', linewidth=2.5, 
                label=f'Mean: {df["quality_score"].mean():.1f}')
    ax1.set_title('Quality Score Distribution', fontweight='600', color=NJ_COLORS['text_dark'], pad=15)
    ax1.set_xlabel('Quality Score', fontweight='600')
    ax1.set_ylabel('Frequency', fontweight='600')
    ax1.legend(loc='upper right', bbox_to_anchor=(0.98, 0.98))
    ax1.grid(True, alpha=0.2)
    
    # 图2：高质量评论情感分布
    ax2 = fig.add_subplot(gs[0, 1])
    high_quality = df[df['quality_score'] > 60]
    if len(high_quality) > 0:
        hq_sentiment = high_quality['sentiment'].value_counts()
        colors = [NJ_PALETTE[i] for i in range(len(hq_sentiment))]
        wedges, texts, autotexts = ax2.pie(hq_sentiment.values, labels=hq_sentiment.index, 
                                          autopct='%1.1f%%', startangle=90, colors=colors,
                                          textprops={'fontsize': 10})
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        ax2.set_title(f'High Quality Comments Sentiment\n({len(high_quality)} comments)', 
                     fontweight='600', color=NJ_COLORS['text_dark'], pad=15)
    
    # 图3：Melancholy_positive分析
    ax3 = fig.add_subplot(gs[1, 0])
    melancholy_positive_comments = df[df['sentiment'] == 'melancholy_positive']
    if len(melancholy_positive_comments) > 0:
        mel_pos_types = melancholy_positive_comments['comment_type'].value_counts()
        bars = ax3.bar(mel_pos_types.index, mel_pos_types.values, 
                      color=NJ_COLORS['light_blue'], alpha=0.8, 
                      edgecolor='white', linewidth=1.5)
        ax3.set_title('Melancholy-Positive by Type', fontweight='600', color=NJ_COLORS['text_dark'], pad=15)
        ax3.set_xlabel('Comment Type', fontweight='600')
        ax3.set_ylabel('Count', fontweight='600')
        ax3.tick_params(axis='x', rotation=45)
        ax3.grid(True, alpha=0.2)
    else:
        ax3.text(0.5, 0.5, 'No Melancholy-Positive\nComments Found', 
                ha='center', va='center', transform=ax3.transAxes,
                fontsize=14, color=NJ_COLORS['text_light'])
    
    # 图4：综合洞察文本
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')
    
    total_comments = len(df)
    avg_quality = df['quality_score'].mean()
    top_sentiment = df['sentiment'].mode().iloc[0]
    mv_analysis_count = len(df[df['comment_type'] == 'mv_analysis'])
    
    summary_text = f"""📊 ANALYSIS SUMMARY

Total Comments: {total_comments:,}
Average Quality: {avg_quality:.1f}/100
Top Sentiment: {top_sentiment}

🎭 EMOTIONAL INSIGHTS:
• Unique Emotion: {len(df[df['sentiment'] == 'melancholy_positive'])} melancholy_positive
• Sentiment Types: {df['sentiment'].nunique()}
• High Quality: {len(df[df['quality_score'] > 60])} comments

👥 ENGAGEMENT PATTERNS:
• MV Analysis: {mv_analysis_count} ({mv_analysis_count/total_comments*100:.1f}%)
• Avg Length: {df['comment_length'].mean():.0f} chars

💡 KEY FINDINGS:
• Complex narrative drives analysis
• Balanced emotional responses
• High-quality discourse patterns
• Cultural depth resonates globally"""
    
    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, 
             fontsize=11, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.8", facecolor=NJ_COLORS['pale_blue'], alpha=0.9),
             color=NJ_COLORS['text_dark'])
    
    plt.suptitle('NewJeans "Ditto" - Comprehensive Analysis Overview', 
                 fontsize=18, fontweight='600', color=NJ_COLORS['text_dark'], y=0.95)
    
    plt.tight_layout()
    plt.savefig('ditto_comprehensive_stats.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none', pad_inches=0.2)
    plt.show()
    print("✅ Comprehensive stats panel saved")

# ================================
# 🚀 执行所有可视化
# ================================
if __name__ == "__main__":
    print("\n🎨 Generating NewJeans-Style Professional Visualizations...")
    print("=" * 60)
    
    create_sentiment_pie_chart()
    print()
    
    create_fan_type_bar_chart()
    print()
    
    create_sentiment_fantype_heatmap()
    print()
    
    create_comment_length_distribution()
    print()
    
    create_emotion_wordcloud()
    print()
    
    create_emotion_radar_chart()
    print()
    
    create_emotion_frequency_chart()
    print()
    
    create_comprehensive_stats_panel()
    print()
    
    print("🎉 ALL NEWJEANS-STYLE VISUALIZATIONS COMPLETE!")
    print("📁 Generated Files (Dashboard-Ready):")
    print("   • ditto_sentiment_distribution.png")
    print("   • ditto_fan_types_distribution.png") 
    print("   • ditto_sentiment_fantype_heatmap.png")
    print("   • ditto_comment_length_analysis.png")
    print("   • ditto_emotion_wordcloud.png")
    print("   • ditto_emotion_radar.png")
    print("   • ditto_emotion_frequency.png")
    print("   • ditto_comprehensive_stats.png")
    print("\n✨ Perfect harmony with dashboard design achieved! 🎵")
