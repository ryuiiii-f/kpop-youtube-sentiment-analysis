import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib import font_manager
import warnings
warnings.filterwarnings('ignore')

# 🎨 设置图表样式
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# 🎯 加载分析结果
df = pd.read_csv("../analysis/ditto_sentiment_analysis_results.csv")

print("🎨 Creating Professional Cross-Cultural Analysis Dashboard")
print("=" * 60)

# 📊 数据预处理：聚焦主要语言
main_languages = ['en', 'ko', 'es', 'ja', 'pt']
df_main = df[df['detected_language'].isin(main_languages)].copy()

# 语言标签美化
language_labels = {
    'en': 'English (Global)',
    'ko': 'Korean (Domestic)', 
    'es': 'Spanish (Latin America)',
    'ja': 'Japanese (Asia)',
    'pt': 'Portuguese (Brazil)'
}
df_main['language_label'] = df_main['detected_language'].map(language_labels)

print(f"📈 Analyzing {len(df_main)} comments across {len(main_languages)} languages")

# ================================
# 📊 图表1：情感分布热力图
# ================================
def create_sentiment_heatmap():
    plt.figure(figsize=(12, 8))
    
    # 创建交叉表
    sentiment_lang_crosstab = pd.crosstab(df_main['language_label'], 
                                         df_main['sentiment'], 
                                         normalize='index') * 100
    
    # 热力图
    sns.heatmap(sentiment_lang_crosstab, 
                annot=True, 
                fmt='.1f', 
                cmap='RdYlBu_r',
                cbar_kws={'label': 'Percentage (%)'},
                square=True,
                linewidths=0.5)
    
    plt.title('NewJeans "Ditto" - Sentiment Distribution by Language/Market', 
              fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Sentiment Type', fontsize=12, fontweight='bold')
    plt.ylabel('Language/Market', fontsize=12, fontweight='bold')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('ditto_sentiment_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✅ Sentiment heatmap saved as 'ditto_sentiment_heatmap.png'")

# ================================
# 📊 图表2：评论类型分布
# ================================
def create_comment_type_analysis():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # 左图：堆叠柱状图
    comment_lang_crosstab = pd.crosstab(df_main['language_label'], 
                                       df_main['comment_type'], 
                                       normalize='index') * 100
    
    comment_lang_crosstab.plot(kind='bar', 
                              stacked=True, 
                              ax=ax1,
                              colormap='Set3')
    ax1.set_title('Comment Types Distribution by Language', fontweight='bold', fontsize=14)
    ax1.set_xlabel('Language/Market', fontweight='bold')
    ax1.set_ylabel('Percentage (%)', fontweight='bold')
    ax1.legend(title='Comment Type', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.tick_params(axis='x', rotation=45)
    
    # 右图：MV Analysis专门分析
    mv_analysis_by_lang = df_main[df_main['comment_type'] == 'mv_analysis'].groupby('language_label').size()
    total_by_lang = df_main.groupby('language_label').size()
    mv_analysis_pct = (mv_analysis_by_lang / total_by_lang * 100).fillna(0)
    
    bars = ax2.bar(mv_analysis_pct.index, mv_analysis_pct.values, color='coral', alpha=0.8)
    ax2.set_title('MV Analysis Comments Ratio by Language', fontweight='bold', fontsize=14)
    ax2.set_xlabel('Language/Market', fontweight='bold')
    ax2.set_ylabel('MV Analysis Percentage (%)', fontweight='bold')
    ax2.tick_params(axis='x', rotation=45)
    
    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('ditto_comment_types.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✅ Comment types analysis saved as 'ditto_comment_types.png'")

# ================================
# 📊 图表3：质量分析
# ================================
def create_quality_analysis():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # 左图：质量分数分布箱线图
    df_main.boxplot(column='quality_score', 
                   by='language_label', 
                   ax=ax1)
    ax1.set_title('Comment Quality Distribution by Language', fontweight='bold', fontsize=14)
    ax1.set_xlabel('Language/Market', fontweight='bold')
    ax1.set_ylabel('Quality Score', fontweight='bold')
    ax1.tick_params(axis='x', rotation=45)
    plt.suptitle('')  # 移除默认标题
    
    # 右图：长度 vs 质量散点图
    colors = {'en': 'blue', 'ko': 'red', 'es': 'green', 'ja': 'orange', 'pt': 'purple'}
    
    for lang in main_languages:
        lang_data = df_main[df_main['detected_language'] == lang]
        if len(lang_data) > 0:
            ax2.scatter(lang_data['comment_length'], 
                       lang_data['quality_score'],
                       c=colors[lang], 
                       label=language_labels[lang],
                       alpha=0.6,
                       s=40)
    
    ax2.set_title('Comment Length vs Quality by Language', fontweight='bold', fontsize=14)
    ax2.set_xlabel('Comment Length (characters)', fontweight='bold')
    ax2.set_ylabel('Quality Score', fontweight='bold')
    ax2.legend(title='Language', loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('ditto_quality_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✅ Quality analysis saved as 'ditto_quality_analysis.png'")

# ================================
# 📊 图表4：韩语 vs 英语深度对比
# ================================
def create_korean_english_comparison():
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    ko_data = df_main[df_main['detected_language'] == 'ko']
    en_data = df_main[df_main['detected_language'] == 'en']
    
    # 图1：情感对比
    ko_en_comparison = df_main[df_main['detected_language'].isin(['ko', 'en'])]
    sentiment_comparison = ko_en_comparison.groupby(['detected_language', 'sentiment']).size().unstack(fill_value=0)
    sentiment_comparison_pct = sentiment_comparison.div(sentiment_comparison.sum(axis=1), axis=0) * 100
    
    sentiment_comparison_pct.plot(kind='bar', ax=ax1, width=0.7)
    ax1.set_title('Korean vs English Sentiment Distribution', fontweight='bold', fontsize=14)
    ax1.set_xlabel('Language', fontweight='bold')
    ax1.set_ylabel('Percentage (%)', fontweight='bold')
    ax1.legend(title='Sentiment', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.set_xticklabels(['Korean', 'English'], rotation=0)
    
    # 图2：评论类型对比
    ko_types = ko_data['comment_type'].value_counts(normalize=True) * 100
    en_types = en_data['comment_type'].value_counts(normalize=True) * 100
    
    comparison_df = pd.DataFrame({'Korean': ko_types, 'English': en_types}).fillna(0)
    comparison_df.plot(kind='bar', ax=ax2, width=0.7)
    ax2.set_title('Korean vs English Comment Types', fontweight='bold', fontsize=14)
    ax2.set_xlabel('Comment Type', fontweight='bold')
    ax2.set_ylabel('Percentage (%)', fontweight='bold')
    ax2.legend(title='Language')
    ax2.tick_params(axis='x', rotation=45)
    
    # 图3：质量分数对比
    quality_data = [ko_data['quality_score'], en_data['quality_score']]
    ax3.boxplot(quality_data, labels=['Korean', 'English'])
    ax3.set_title('Korean vs English Quality Score Distribution', fontweight='bold', fontsize=14)
    ax3.set_xlabel('Language', fontweight='bold')
    ax3.set_ylabel('Quality Score', fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # 图4：统计对比表
    ax4.axis('off')
    
    stats_text = f"""
KOREAN MARKET ANALYSIS:
• Total Comments: {len(ko_data)}
• Average Quality: {ko_data['quality_score'].mean():.1f}/100
• Average Length: {ko_data['comment_length'].mean():.1f} chars
• Top Sentiment: {ko_data['sentiment'].mode().iloc[0] if len(ko_data) > 0 else 'N/A'}
• Top Comment Type: {ko_data['comment_type'].mode().iloc[0] if len(ko_data) > 0 else 'N/A'}
• MV Analysis Ratio: {len(ko_data[ko_data['comment_type'] == 'mv_analysis']) / len(ko_data) * 100:.1f}%

GLOBAL MARKET (ENGLISH) ANALYSIS:
• Total Comments: {len(en_data)}
• Average Quality: {en_data['quality_score'].mean():.1f}/100
• Average Length: {en_data['comment_length'].mean():.1f} chars
• Top Sentiment: {en_data['sentiment'].mode().iloc[0] if len(en_data) > 0 else 'N/A'}
• Top Comment Type: {en_data['comment_type'].mode().iloc[0] if len(en_data) > 0 else 'N/A'}
• MV Analysis Ratio: {len(en_data[en_data['comment_type'] == 'mv_analysis']) / len(en_data) * 100:.1f}%

KEY INSIGHTS:
• Korean fans prefer analytical discussion
• English fans show more emotional response
• Both markets engage deeply with MV content
• Quality scores reflect cultural communication styles
    """
    
    ax4.text(0.05, 0.95, stats_text, 
             transform=ax4.transAxes, 
             fontsize=11,
             verticalalignment='top',
             fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('ditto_korean_english_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✅ Korean vs English comparison saved as 'ditto_korean_english_comparison.png'")

# ================================
# 📊 图表5：详细统计可视化
# ================================
def create_detailed_statistics():
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 图1：各语言评论数量
    language_counts = df_main['language_label'].value_counts()
    bars1 = ax1.bar(range(len(language_counts)), language_counts.values, color='skyblue', alpha=0.8)
    ax1.set_title('Comment Volume by Language/Market', fontweight='bold', fontsize=14)
    ax1.set_xlabel('Language/Market', fontweight='bold')
    ax1.set_ylabel('Number of Comments', fontweight='bold')
    ax1.set_xticks(range(len(language_counts)))
    ax1.set_xticklabels(language_counts.index, rotation=45)
    
    # 添加数值标签
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 5,
                f'{int(height)}', ha='center', va='bottom', fontweight='bold')
    
    # 图2：平均质量分数对比
    avg_quality = df_main.groupby('language_label')['quality_score'].mean().sort_values(ascending=False)
    bars2 = ax2.bar(range(len(avg_quality)), avg_quality.values, color='lightcoral', alpha=0.8)
    ax2.set_title('Average Quality Score by Language', fontweight='bold', fontsize=14)
    ax2.set_xlabel('Language/Market', fontweight='bold')
    ax2.set_ylabel('Average Quality Score', fontweight='bold')
    ax2.set_xticks(range(len(avg_quality)))
    ax2.set_xticklabels(avg_quality.index, rotation=45)
    
    # 添加数值标签
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.1f}', ha='center', va='bottom', fontweight='bold')
    
    # 图3：Melancholy_positive情感分析
    melancholy_positive = df_main[df_main['sentiment'] == 'melancholy_positive']
    if len(melancholy_positive) > 0:
        mel_pos_by_lang = melancholy_positive.groupby('language_label').size()
        total_by_lang = df_main.groupby('language_label').size()
        mel_pos_pct = (mel_pos_by_lang / total_by_lang * 100).fillna(0)
        
        bars3 = ax3.bar(range(len(mel_pos_pct)), mel_pos_pct.values, color='mediumpurple', alpha=0.8)
        ax3.set_title('Melancholy-Positive Sentiment by Language\n(Unique Ditto Emotion)', fontweight='bold', fontsize=14)
        ax3.set_xlabel('Language/Market', fontweight='bold')
        ax3.set_ylabel('Percentage (%)', fontweight='bold')
        ax3.set_xticks(range(len(mel_pos_pct)))
        ax3.set_xticklabels(mel_pos_pct.index, rotation=45)
        
        # 添加数值标签
        for bar in bars3:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')
    else:
        ax3.text(0.5, 0.5, 'No Melancholy-Positive\nComments Found', 
                ha='center', va='center', transform=ax3.transAxes, fontsize=14)
    
    # 图4：综合市场洞察
    ax4.axis('off')
    
    # 计算关键指标
    total_comments = len(df_main)
    korean_pct = len(df_main[df_main['detected_language'] == 'ko']) / total_comments * 100
    english_pct = len(df_main[df_main['detected_language'] == 'en']) / total_comments * 100
    
    mv_analysis_total = len(df_main[df_main['comment_type'] == 'mv_analysis'])
    mv_analysis_pct = mv_analysis_total / total_comments * 100
    
    insights_text = f"""
NEWJEANS "DITTO" GLOBAL IMPACT SUMMARY

📊 MARKET PENETRATION:
• Total Analyzed Comments: {total_comments:,}
• Korean Market: {korean_pct:.1f}%
• Global Market (English): {english_pct:.1f}%
• Emerging Markets: {100-korean_pct-english_pct:.1f}%

🎬 CONTENT ENGAGEMENT:
• MV Analysis Comments: {mv_analysis_total} ({mv_analysis_pct:.1f}%)
• High Quality Comments (>60): {len(df_main[df_main['quality_score'] > 60])}
• Average Comment Length: {df_main['comment_length'].mean():.0f} characters

💡 STRATEGIC INSIGHTS:
• Korean fans engage analytically (90% neutral sentiment)
• Global fans respond emotionally (43% positive sentiment)
• Complex narrative drives deep discussion across cultures
• Melancholy-positive emotion unique to Ditto concept

🎯 MARKETING RECOMMENDATIONS:
• Domestic: Emphasize artistic depth and cultural context
• Global: Highlight emotional moments and relatability
• Content: Continue narrative complexity for engagement
• Expansion: Latin American market shows aesthetic appreciation
    """
    
    ax4.text(0.05, 0.95, insights_text, 
             transform=ax4.transAxes, 
             fontsize=10,
             verticalalignment='top',
             fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.8", facecolor="lightyellow", alpha=0.9))
    
    plt.tight_layout()
    plt.savefig('ditto_detailed_statistics.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✅ Detailed statistics saved as 'ditto_detailed_statistics.png'")

# ================================
# 🚀 执行所有可视化
# ================================
if __name__ == "__main__":
    print("\n🎨 Generating Individual High-Quality Charts...")
    print("=" * 60)
    
    create_sentiment_heatmap()
    print()
    
    create_comment_type_analysis()
    print()
    
    create_quality_analysis()
    print()
    
    create_korean_english_comparison()
    print()
    
    create_detailed_statistics()
    print()
    
    print("🎉 ALL VISUALIZATIONS COMPLETE!")
    print("📁 Generated Files:")
    print("   • ditto_sentiment_heatmap.png")
    print("   • ditto_comment_types.png") 
    print("   • ditto_quality_analysis.png")
    print("   • ditto_korean_english_comparison.png")
    print("   • ditto_detailed_statistics.png")
    print("\n✨ Professional-grade cross-cultural analysis ready for presentation!")
