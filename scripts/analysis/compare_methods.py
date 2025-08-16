import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 读取两种方法的结果
print("📊 COMPARING TWO 3-CATEGORY CLASSIFICATION METHODS")
print("="*60)

# 假设你已经运行了两种方法
try:
    df_rule = pd.read_csv("labeled_comments_hylt_3category.csv")  # 规则方法
    print("✅ Rule-based results loaded")
except:
    print("❌ Rule-based results not found - run the rule classifier first")
    df_rule = None

try:
    df_hf = pd.read_csv("labeled_comments_hylt_hf3category.csv")  # HF方法
    print("✅ HuggingFace results loaded")
except:
    print("❌ HuggingFace results not found - run the HF classifier first") 
    df_hf = None

if df_rule is not None and df_hf is not None:
    # ============================================
    # 性能对比分析
    # ============================================
    
    rule_counts = df_rule['fan_type'].value_counts()
    hf_counts = df_hf['fan_type'].value_counts()
    
    print(f"\n📈 PERFORMANCE COMPARISON:")
    print("-" * 40)
    
    # 规则方法统计
    rule_specific = rule_counts.get('visual-focused', 0) + rule_counts.get('content-focused', 0)
    rule_specific_pct = rule_specific / len(df_rule) * 100
    rule_general_pct = rule_counts.get('general-praise', 0) / len(df_rule) * 100
    
    print(f"🔧 RULE-BASED METHOD:")
    print(f"   Specific classification: {rule_specific_pct:.1f}%")
    print(f"   General praise: {rule_general_pct:.1f}%")
    
    # HF方法统计
    hf_specific = hf_counts.get('visual-focused', 0) + hf_counts.get('content-focused', 0)
    hf_specific_pct = hf_specific / len(df_hf) * 100
    hf_general_pct = hf_counts.get('general-praise', 0) / len(df_hf) * 100
    
    print(f"🤖 HUGGINGFACE METHOD:")
    print(f"   Specific classification: {hf_specific_pct:.1f}%")
    print(f"   General praise: {hf_general_pct:.1f}%")
    
    # 判断哪种方法更好
    print(f"\n🏆 WINNER ANALYSIS:")
    if hf_specific_pct > rule_specific_pct + 5:
        print(f"   🥇 HuggingFace wins! {hf_specific_pct:.1f}% vs {rule_specific_pct:.1f}%")
        print(f"   💡 Recommendation: Use HuggingFace method")
        winner = "huggingface"
    elif rule_specific_pct > hf_specific_pct + 5:
        print(f"   🥇 Rule-based wins! {rule_specific_pct:.1f}% vs {hf_specific_pct:.1f}%")
        print(f"   💡 Recommendation: Use rule-based method")
        winner = "rule"
    else:
        print(f"   🤝 Similar performance: HF {hf_specific_pct:.1f}% vs Rule {rule_specific_pct:.1f}%")
        print(f"   💡 Recommendation: Choose based on other factors")
        winner = "tie"
    
    # ============================================
    # 可视化对比
    # ============================================
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 方法1：规则分类分布
    rule_counts.plot(kind='pie', ax=axes[0,0], autopct='%1.1f%%', startangle=90,
                     colors=['#FF6B6B', '#4ECDC4', '#45B7D1'])
    axes[0,0].set_title('Rule-Based Classification\n3-Category System', fontweight='bold')
    axes[0,0].set_ylabel('')
    
    # 方法2：HF分类分布  
    hf_counts.plot(kind='pie', ax=axes[0,1], autopct='%1.1f%%', startangle=90,
                   colors=['#FF6B6B', '#4ECDC4', '#45B7D1'])
    axes[0,1].set_title('HuggingFace Classification\n3-Category System', fontweight='bold')
    axes[0,1].set_ylabel('')
    
    # 性能对比柱状图
    methods = ['Rule-Based', 'HuggingFace']
    specific_ratios = [rule_specific_pct, hf_specific_pct]
    general_ratios = [rule_general_pct, hf_general_pct]
    
    x = np.arange(len(methods))
    width = 0.35
    
    axes[1,0].bar(x - width/2, specific_ratios, width, label='Specific Classification', color='lightcoral')
    axes[1,0].bar(x + width/2, general_ratios, width, label='General Praise', color='lightblue')
    axes[1,0].set_title('Performance Comparison\nSpecific vs General Classification', fontweight='bold')
    axes[1,0].set_ylabel('Percentage (%)')
    axes[1,0].set_xticks(x)
    axes[1,0].set_xticklabels(methods)
    axes[1,0].legend()
    
    # 语言效果分析（如果有语言数据）
    if 'language_pattern' in df_hf.columns:
        lang_performance = df_hf.groupby('language_pattern')['fan_type'].apply(
            lambda x: (x != 'general-praise').sum() / len(x) * 100
        )
        
        lang_performance.plot(kind='bar', ax=axes[1,1], color='green', alpha=0.7)
        axes[1,1].set_title('HF Performance by Language\n(% Specific Classification)', fontweight='bold')
        axes[1,1].set_ylabel('Specific Classification %')
        axes[1,1].tick_params(axis='x', rotation=45)
    else:
        axes[1,1].text(0.5, 0.5, 'Language Analysis\nNot Available', 
                      ha='center', va='center', fontsize=14, alpha=0.5)
        axes[1,1].set_title('Language Performance Analysis', fontweight='bold')
    
    plt.suptitle('3-Category Classification Method Comparison', fontsize=18, fontweight='bold')
    plt.tight_layout()
    plt.savefig('method_comparison_3category.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✅ Saved: method_comparison_3category.png")
    
    # ============================================
    # 保存最优结果
    # ============================================
    
    if winner == "huggingface":
        best_df = df_hf.copy()
        method_name = "HuggingFace"
    elif winner == "rule":
        best_df = df_rule.copy()  
        method_name = "Rule-Based"
    else:
        # 如果平局，选择HF（更现代）
        best_df = df_hf.copy()
        method_name = "HuggingFace"
    
    # 保存最终最优结果
    final_df = best_df[['comment_text', 'emotion', 'fan_type']].copy()
    final_df.to_csv("final_labeled_comments_3category.csv", index=False, encoding="utf-8-sig")
    
    print(f"\n💾 FINAL RESULTS SAVED:")
    print(f"   ✅ final_labeled_comments_3category.csv ({method_name} method)")
    print(f"   📊 Specific classification rate: {max(rule_specific_pct, hf_specific_pct):.1f}%")
    
    print(f"\n🚀 READY FOR NEXT MODULE:")
    print(f"   🎵 Spotify Audio Features Analysis")
    print(f"   🔗 Audio Features ↔ Fan Emotion Correlation")
    print(f"   🎯 Build K-pop Creation Diagnosis Engine")

else:
    print(f"\n❌ Please run both classification methods first:")
    print(f"   1. Rule-based 3-category classifier")
    print(f"   2. HuggingFace 3-category classifier") 
    print(f"   3. Then run this comparison analysis")
