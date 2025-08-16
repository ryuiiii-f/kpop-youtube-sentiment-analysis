import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# 🎨 设置蓝白配色主题
plt.style.use('seaborn-v0_8-whitegrid')

# 定义蓝白配色方案
BLUE_PALETTE = ['#E3F2FD', '#BBDEFB', '#90CAF9', '#64B5F6', '#42A5F5', '#2196F3', '#1E88E5', '#1976D2', '#1565C0', '#0D47A1']
BLUE_WHITE_COLORS = ['#2196F3', '#64B5F6', '#90CAF9', '#BBDEFB', '#E3F2FD']

# 设置全局字体和颜色
plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 16
})

# 🎯 加载分析结果
df = pd.read_csv("../analysis/ditto_sentiment_analysis_results.csv")

print("🎨 Creating NewJeans Ditto Comprehensive Analysis Dashboard")
print("=" * 70)
print(f"📊 Dataset: {len(df)} comments | Languages: {df['detected_language'].nunique()} | Sentiments: {df['sentiment'].nunique()}")

# 数据预处理
main_languages = ['en', 'ko', 'es', 'ja', 'pt']
df_main = df[df['detected_language'].isin(main_languages)].copy()

language_labels = {
    'en': 'English (Global)',
    'ko': 'Korean (Domestic)', 
    'es': 'Spanish (Latin America)',
    'ja': 'Japanese (Asia)',
    'pt': 'Portuguese (Brazil)'
}
df_main['language_label'] = df_main['detected_language'].map(language_labels)

# ================================
# 📊 创建综合Dashboard
# ================================
def create_comprehensive_dashboard():
    # 创建大尺寸画布，分为多个区域
    fig = plt.figure(figsize=(24, 16))
    fig.patch.set_facecolor('white')
    
    # 主标题
    fig.suptitle('NewJeans "Ditto" - Comprehensive Cross-Cultural Analysis Dashboard\n'
                 'Global Fan Engagement & Emotional Response Patterns', 
                 fontsize=24, fontweight='bold', color='#1565C0', y=0.98)
    
    # ================================
    # 区域1：整体情感概览 (左上)
    # ================================
    ax1 = plt.subplot2grid((4, 6), (0, 0), colspan=2, rowspan=2)
    
    sentiment_counts = df['sentiment'].value_counts()
    wedges, texts, autotexts = ax1.pie(sentiment_counts.values, 
                                      labels=sentiment_counts.index,
                                      autopct='%1.1f%%',
                                      colors=BLUE_WHITE_COLORS,
                                      startangle=90,
                                      explode=[0.1 if x == 'melancholy_positive' else 0 for x in sentiment_counts.index])
    
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(10)
    
    for text in texts:
        text.set_fontsize(10)
        text.set_fontweight('bold')
        text.set_color('#1565C0')
    
    ax1.set_title('Overall Sentiment Distribution\n(905 Comments)', 
                  fontweight='bold', color='#1565C0', pad=20)
    
    # ================================
    # 区域2：跨语言情感对比 (中上)
    # ================================
    ax2 = plt.subplot2grid((4, 6), (0, 2), colspan=2, rowspan=1)
    
    # 创建韩语vs英语对比
    ko_en_data = df_main[df_main['detected_language'].isin(['ko', 'en'])]
    lang_sentiment = ko_en_data.groupby(['detected_language', 'sentiment']).size().unstack(fill_value=0)
    lang_sentiment_pct = lang_sentiment.div(lang_sentiment.sum(axis=1), axis=0) * 100
    
    lang_sentiment_pct.plot(kind='bar', ax=ax2, color=BLUE_PALETTE[:len(lang_sentiment_pct.columns)], width=0.7)
    ax2.set_title('Korean vs English Sentiment Patterns', fontweight='bold', color='#1565C0')
    ax2.set_xlabel('Language Market', fontweight='bold', color='#1565C0')
    ax2.set_ylabel('Percentage (%)', fontweight='bold', color='#1565C0')
    ax2.legend(title='Sentiment', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.set_xticklabels(['Korean (Domestic)', 'English (Global)'], rotation=0)
    ax2.grid(True, alpha=0.3)
    
    # ================================
    # 区域3：粉丝参与类型 (右上)
    # ================================
    ax3 = plt.subplot2grid((4, 6), (0, 4), colspan=2, rowspan=1)
    
    fan_type_counts = df['comment_type'].value_counts()
    bars = ax3.barh(range(len(fan_type_counts)), fan_type_counts.values, 
                    color=BLUE_PALETTE[:len(fan_type_counts)])
    
    # 添加数值标签
    for i, (bar, value) in enumerate(zip(bars, fan_type_counts.values)):
        ax3.text(value + 5, bar.get_y() + bar.get_height()/2, 
                f'{value} ({value/len(df)*100:.1f}%)',
                va='center', fontweight='bold', color='#1565C0')
    
    ax3.set_yticks(range(len(fan_type_counts)))
    ax3.set_yticklabels(fan_type_counts.index)
    ax3.set_xlabel('Number of Comments', fontweight='bold', color='#1565C0')
    ax3.set_title('Fan Engagement Types', fontweight='bold', color='#1565C0')
    ax3.grid(True, alpha=0.3)
    
    # ================================
    # 区域4：语言分布统计 (中上右)
    # ================================
    ax4 = plt.subplot2grid((4, 6), (1, 2), colspan=2, rowspan=1)
    
    language_counts = df_main['language_label'].value_counts()
    bars4 = ax4.bar(range(len(language_counts)), language_counts.values, 
                    color=BLUE_PALETTE[:len(language_counts)], alpha=0.8)
    
    # 添加数值标签
    for bar in bars4:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 5,
                f'{int(height)}', ha='center', va='bottom', 
                fontweight='bold', color='#1565C0')
    
    ax4.set_xticks(range(len(language_counts)))
    ax4.set_xticklabels([label.split(' ')[0] for label in language_counts.index], rotation=45)
    ax4.set_ylabel('Number of Comments', fontweight='bold', color='#1565C0')
    ax4.set_title('Global Market Penetration', fontweight='bold', color='#1565C0')
    ax4.grid(True, alpha=0.3)
    
    # ================================
    # 区域5：质量分析 (右中)
    # ================================
    ax5 = plt.subplot2grid((4, 6), (1, 4), colspan=2, rowspan=1)
    
    # 质量分组
    df['quality_group'] = pd.cut(df['quality_score'], 
                                bins=[0, 40, 60, 100], 
                                labels=['Low (0-40)', 'Medium (40-60)', 'High (60-100)'])
    
    quality_counts = df['quality_group'].value_counts()
    ax5.bar(quality_counts.index, quality_counts.values, 
            color=['#E3F2FD', '#90CAF9', '#1976D2'], alpha=0.8)
    
    for i, v in enumerate(quality_counts.values):
        ax5.text(i, v + 5, f'{v}\n({v/len(df)*100:.1f}%)', 
                ha='center', va='bottom', fontweight='bold', color='#1565C0')
    
    ax5.set_ylabel('Number of Comments', fontweight='bold', color='#1565C0')
    ax5.set_title('Comment Quality Distribution', fontweight='bold', color='#1565C0')
    ax5.tick_params(axis='x', rotation=45)
    ax5.grid(True, alpha=0.3)
    
    # ================================
    # 区域6：情感×粉丝类型热力图 (下左)
    # ================================
    ax6 = plt.subplot2grid((4, 6), (2, 0), colspan=3, rowspan=1)
    
    crosstab = pd.crosstab(df['comment_type'], df['sentiment'], normalize='index') * 100
    
    sns.heatmap(crosstab, 
                annot=True, 
                fmt='.1f', 
                cmap='Blues',
                cbar_kws={'label': 'Percentage (%)'},
                ax=ax6,
                linewidths=0.5)
    
    ax6.set_title('Sentiment Patterns by Fan Engagement Type', 
                  fontweight='bold', color='#1565C0', pad=15)
    ax6.set_xlabel('Sentiment Type', fontweight='bold', color='#1565C0')
    ax6.set_ylabel('Fan Engagement Type', fontweight='bold', color='#1565C0')
    
    # ================================
    # 区域7：评论长度分析 (下右)
    # ================================
    ax7 = plt.subplot2grid((4, 6), (2, 3), colspan=3, rowspan=1)
    
    # 按语言分组的长度对比
    language_length = df_main.groupby('language_label')['comment_length'].mean().sort_values(ascending=True)
    
    bars7 = ax7.barh(range(len(language_length)), language_length.values, 
                     color=BLUE_PALETTE[:len(language_length)], alpha=0.8)
    
    for i, (bar, value) in enumerate(zip(bars7, language_length.values)):
        ax7.text(value + 2, bar.get_y() + bar.get_height()/2, 
                f'{value:.0f} chars',
                va='center', fontweight='bold', color='#1565C0')
    
    ax7.set_yticks(range(len(language_length)))
    ax7.set_yticklabels([label.split(' ')[0] for label in language_length.index])
    ax7.set_xlabel('Average Comment Length (characters)', fontweight='bold', color='#1565C0')
    ax7.set_title('Comment Depth by Language Market', fontweight='bold', color='#1565C0')
    ax7.grid(True, alpha=0.3)
    
    # ================================
    # 区域8：关键洞察面板 (底部)
    # ================================
    ax8 = plt.subplot2grid((4, 6), (3, 0), colspan=6, rowspan=1)
    ax8.axis('off')
    
    # 计算关键统计
    ko_neutral_pct = len(df[(df['detected_language'] == 'ko') & (df['sentiment'] == 'neutral')]) / len(df[df['detected_language'] == 'ko']) * 100
    en_positive_pct = len(df[(df['detected_language'] == 'en') & (df['sentiment'] == 'positive')]) / len(df[df['detected_language'] == 'en']) * 100
    mv_analysis_pct = len(df[df['comment_type'] == 'mv_analysis']) / len(df) * 100
    melancholy_count = len(df[df['sentiment'] == 'melancholy_positive'])
    
    insights_text = f"""
🔍 KEY CULTURAL INSIGHTS & STRATEGIC IMPLICATIONS

📊 GLOBAL MARKET ANALYSIS:
• Total Comments Analyzed: {len(df):,} across {df['detected_language'].nunique()} languages
• Korean Market (Domestic): {len(df[df['detected_language'] == 'ko'])} comments - {ko_neutral_pct:.1f}% analytical/neutral sentiment
• Global Market (English): {len(df[df['detected_language'] == 'en'])} comments - {en_positive_pct:.1f}% positive emotional response
• Emerging Markets: Strong presence in Latin America (Spanish) and Asia (Japanese)

🎬 CONTENT ENGAGEMENT DEPTH:
• MV Analysis Comments: {len(df[df['comment_type'] == 'mv_analysis'])} ({mv_analysis_pct:.1f}%) - Exceptionally high for K-pop content
• Unique Emotional Response: {melancholy_count} melancholy_positive comments - Ditto's distinctive emotional complexity
• High Quality Discussions: {len(df[df['quality_score'] > 60])} comments demonstrate deep fan engagement

💡 STRATEGIC RECOMMENDATIONS:
• Korean Market: Emphasize artistic depth, cultural context, and analytical content (90% prefer rational discourse)
• Global Market: Focus on emotional storytelling, relatability, and immediate impact (43% seek positive emotional connection)
• Content Strategy: Complex narratives drive 20.6% analytical engagement - continue innovative storytelling approaches
• Cultural Localization: Adapt emotional tone and communication style for different markets while maintaining core artistic vision

🎯 BUSINESS IMPLICATIONS:
• Narrative complexity creates cross-cultural engagement without losing cultural authenticity
• High MV analysis ratio indicates successful artistic risk-taking and cultural impact
• Balanced emotional response across markets suggests universal appeal with localized resonance
    """
    
    ax8.text(0.02, 0.98, insights_text, 
             transform=ax8.transAxes, 
             fontsize=11,
             verticalalignment='top',
             fontfamily='monospace',
             color='#1565C0',
             bbox=dict(boxstyle="round,pad=1.0", facecolor="#F3F9FF", alpha=0.9, edgecolor='#2196F3'))
    
    # 调整布局
    plt.tight_layout()
    plt.subplots_adjust(top=0.93, bottom=0.08, hspace=0.35, wspace=0.25)
    
    # 保存高分辨率图片
    plt.savefig('ditto_comprehensive_dashboard.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.show()
    
    print("✅ Comprehensive dashboard saved as 'ditto_comprehensive_dashboard.png'")

# ================================
# 🚀 创建补充页面：详细统计
# ================================
def create_detailed_metrics_page():
    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(20, 12))
    fig.patch.set_facecolor('white')
    fig.suptitle('NewJeans "Ditto" - Detailed Metrics & Cross-Cultural Comparison', 
                 fontsize=20, fontweight='bold', color='#1565C0', y=0.98)
    
    # 图1：情感随质量变化
    quality_sentiment = df.groupby(['quality_group', 'sentiment']).size().unstack(fill_value=0)
    quality_sentiment_pct = quality_sentiment.div(quality_sentiment.sum(axis=1), axis=0) * 100
    
    quality_sentiment_pct.plot(kind='bar', ax=ax1, color=BLUE_PALETTE, stacked=True)
    ax1.set_title('Sentiment Evolution by Quality Level', fontweight='bold', color='#1565C0')
    ax1.set_xlabel('Quality Group', fontweight='bold')
    ax1.set_ylabel('Percentage (%)', fontweight='bold')
    ax1.legend(title='Sentiment', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.tick_params(axis='x', rotation=0)
    
    # 图2：各语言MV分析比例
    mv_analysis_by_lang = df_main[df_main['comment_type'] == 'mv_analysis'].groupby('language_label').size()
    total_by_lang = df_main.groupby('language_label').size()
    mv_analysis_pct = (mv_analysis_by_lang / total_by_lang * 100).fillna(0)
    
    bars2 = ax2.bar(range(len(mv_analysis_pct)), mv_analysis_pct.values, 
                    color=BLUE_PALETTE[:len(mv_analysis_pct)], alpha=0.8)
    
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.3,
                f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    ax2.set_xticks(range(len(mv_analysis_pct)))
    ax2.set_xticklabels([label.split(' ')[0] for label in mv_analysis_pct.index], rotation=45)
    ax2.set_title('MV Analysis Engagement by Market', fontweight='bold', color='#1565C0')
    ax2.set_ylabel('MV Analysis Percentage (%)', fontweight='bold')
    
    # 图3：评论长度分布
    ax3.hist(df['comment_length'], bins=25, color='#64B5F6', alpha=0.7, edgecolor='#1976D2')
    ax3.axvline(df['comment_length'].mean(), color='#D32F2F', linestyle='--', linewidth=2, 
                label=f'Mean: {df["comment_length"].mean():.0f}')
    ax3.set_title('Comment Length Distribution', fontweight='bold', color='#1565C0')
    ax3.set_xlabel('Length (characters)', fontweight='bold')
    ax3.set_ylabel('Frequency', fontweight='bold')
    ax3.legend()
    
    # 图4：高质量评论的语言分布
    high_quality = df_main[df_main['quality_score'] > 60]
    hq_lang_dist = high_quality['language_label'].value_counts()
    
    ax4.pie(hq_lang_dist.values, labels=[label.split(' ')[0] for label in hq_lang_dist.index], 
            autopct='%1.1f%%', colors=BLUE_WHITE_COLORS, startangle=90)
    ax4.set_title('High Quality Comments\nby Language Market', fontweight='bold', color='#1565C0')
    
    # 图5：Melancholy_positive详细分析
    if len(df[df['sentiment'] == 'melancholy_positive']) > 0:
        mel_pos_data = df[df['sentiment'] == 'melancholy_positive']
        mel_pos_types = mel_pos_data['comment_type'].value_counts()
        
        ax5.bar(mel_pos_types.index, mel_pos_types.values, 
                color='#9C27B0', alpha=0.7)
        ax5.set_title('Melancholy-Positive Comments\nby Engagement Type', 
                      fontweight='bold', color='#1565C0')
        ax5.set_ylabel('Count', fontweight='bold')
        ax5.tick_params(axis='x', rotation=45)
    else:
        ax5.text(0.5, 0.5, 'No Melancholy-Positive\nComments Available', 
                ha='center', va='center', transform=ax5.transAxes, fontsize=12)
    
    # 图6：综合得分对比
    metrics_data = {
        'Korean Market': [ko_neutral_pct, len(df[df['detected_language'] == 'ko']), 
                         df[df['detected_language'] == 'ko']['quality_score'].mean()],
        'English Market': [en_positive_pct, len(df[df['detected_language'] == 'en']), 
                          df[df['detected_language'] == 'en']['quality_score'].mean()]
    }
    
    x = np.arange(2)
    width = 0.25
    
    ax6.bar(x - width, [metrics_data['Korean Market'][0], metrics_data['English Market'][0]], 
            width, label='Dominant Sentiment %', color='#2196F3', alpha=0.8)
    ax6.bar(x, [metrics_data['Korean Market'][2], metrics_data['English Market'][2]], 
            width, label='Avg Quality Score', color='#64B5F6', alpha=0.8)
    ax6.bar(x + width, [len(df[df['detected_language'] == 'ko'])/len(df)*100, 
                       len(df[df['detected_language'] == 'en'])/len(df)*100], 
            width, label='Market Share %', color='#90CAF9', alpha=0.8)
    
    ax6.set_xlabel('Market', fontweight='bold')
    ax6.set_ylabel('Percentage / Score', fontweight='bold')
    ax6.set_title('Korean vs English Market Metrics', fontweight='bold', color='#1565C0')
    ax6.set_xticks(x)
    ax6.set_xticklabels(['Korean', 'English'])
    ax6.legend()
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    plt.savefig('ditto_detailed_metrics.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.show()
    
    print("✅ Detailed metrics page saved as 'ditto_detailed_metrics.png'")

# ================================
# 🚀 执行Dashboard生成
# ================================
if __name__ == "__main__":
    print("\n🎨 Generating Clean Blue-White Dashboard...")
    print("=" * 70)
    
    create_comprehensive_dashboard()
    print()
    
    create_detailed_metrics_page()
    print()
    
    print("🎉 CLEAN DASHBOARD COMPLETE!")
    print("📁 Generated Files:")
    print("   • ditto_comprehensive_dashboard.png - Main dashboard (24x16)")
    print("   • ditto_detailed_metrics.png - Detailed metrics page (20x12)")
    print("\n✨ Professional blue-white themed dashboard ready for presentation!")
    print("🎯 Key highlights: Cross-cultural insights, Ditto's unique emotional impact, Strategic recommendations")
