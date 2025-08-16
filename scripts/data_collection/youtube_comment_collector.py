import os
import pandas as pd
from googleapiclient.discovery import build
from dotenv import load_dotenv
from tqdm import tqdm

# ✅ 载入 API 密钥（从 .env 文件）
load_dotenv()
API_KEY = os.getenv("YOUTUBE_API_KEY")

# ✅ 视频 ID（BLACKPINK - How You Like That）
VIDEO_ID = "ioNng23DkIM"
MAX_COMMENTS = 500  # 你可以改成 1000 或更多

# ✅ 初始化 YouTube API 客户端
youtube = build("youtube", "v3", developerKey=API_KEY)

# ✅ 评论抓取函数
def get_comments(video_id, max_comments):
    comments = []
    next_page_token = None
    pbar = tqdm(total=max_comments, desc="Fetching Comments")

    while len(comments) < max_comments:
        request = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=100,
            pageToken=next_page_token,
            order="relevance",  # 可改为 "time" 查看最新评论
            textFormat="plainText"
        )
        response = request.execute()

        for item in response["items"]:
            snippet = item["snippet"]["topLevelComment"]["snippet"]
            comment = {
                "comment_text": snippet["textDisplay"],
                "like_count": snippet["likeCount"],
                "published_at": snippet["publishedAt"]
            }
            comments.append(comment)
            pbar.update(1)
            if len(comments) >= max_comments:
                break

        next_page_token = response.get("nextPageToken")
        if not next_page_token:
            break

    pbar.close()
    return comments

# ✅ 执行抓取并保存
if __name__ == "__main__":
    print("Starting comment collection...")
    data = get_comments(VIDEO_ID, MAX_COMMENTS)
    df = pd.DataFrame(data)
    df.to_csv("how_you_like_that_comments.csv", index=False, encoding="utf-8-sig")
    print("✅ Done! Comments saved to 'how_you_like_that_comments.csv'")
