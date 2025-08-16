import pandas as pd
import re

# 加载原始评论数据
df = pd.read_csv("how_you_like_that_comments.csv")

# 去除空值和仅含空格或表情的评论
def is_valid_comment(text):
    if pd.isna(text):
        return False
    # 去掉空白和非文本字符后是否仍有字母/汉字/韩文等
    clean = re.sub(r"[\W_]", "", str(text))  # 移除符号
    return len(clean.strip()) >= 3

df["comment_text"] = df["comment_text"].astype(str)
df = df[df["comment_text"].apply(is_valid_comment)]

# 去重
df = df.drop_duplicates(subset=["comment_text"])

# 重置索引
df = df.reset_index(drop=True)

# 输出清洗后的文件
df.to_csv("clean_comments_how_you_like_that.csv", index=False, encoding="utf-8-sig")

print(f"✅ Cleaned data saved to 'clean_comments_how_you_like_that.csv' with {len(df)} rows.")
