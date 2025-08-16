import os
import pandas as pd
from googleapiclient.discovery import build
from dotenv import load_dotenv
from tqdm import tqdm
import time
from datetime import datetime

# ✅ 载入 API 密钥（从 .env 文件）
load_dotenv()
API_KEY = os.getenv("YOUTUBE_API_KEY")

# 🎯 NewJeans Ditto 视频信息
VIDEOS = {
    "ditto_side_a": {
        "id": "pSUydWEqKwE", 
        "title": "NewJeans 'Ditto' Side A"
    },
    "ditto_side_b": {
        "id": "V37TaRdVUQY", 
        "title": "NewJeans 'Ditto' Side B"
    },
    "ditto_performance": {
        "id": "Km71Rr9K-Bw", 
        "title": "NewJeans 'Ditto' Performance"
    }
}

MAX_COMMENTS = 1000  # 建议增加到1000，Ditto评论质量高
SELECTED_VIDEO = "ditto_side_a"  # 主要分析Side A

# ✅ 初始化 YouTube API 客户端
youtube = build("youtube", "v3", developerKey=API_KEY)

# ✅ 增强版评论抓取函数
def get_comments_enhanced(video_id, max_comments, video_title):
    """
    增强版评论收集，包含更多metadata
    """
    comments = []
    next_page_token = None
    pbar = tqdm(total=max_comments, desc=f"Fetching {video_title}")

    # 先获取视频基础信息
    video_info = get_video_stats(video_id)
    
    while len(comments) < max_comments:
        try:
            request = youtube.commentThreads().list(
                part="snippet,replies",  # 增加replies获取回复数
                videoId=video_id,
                maxResults=100,
                pageToken=next_page_token,
                order="relevance",  # Ditto适合用relevance，能获取高质量讨论
                textFormat="plainText"
            )
            response = request.execute()

            for item in response["items"]:
                snippet = item["snippet"]["topLevelComment"]["snippet"]
                
                # 🔍 增强数据收集
                comment = {
                    "video_id": video_id,
                    "video_title": video_title,
                    "comment_id": item["snippet"]["topLevelComment"]["id"],
                    "comment_text": snippet["textDisplay"],
                    "author_name": snippet["authorDisplayName"],
                    "like_count": snippet["likeCount"],
                    "reply_count": item["snippet"]["totalReplyCount"],
                    "published_at": snippet["publishedAt"],
                    "updated_at": snippet.get("updatedAt", snippet["publishedAt"]),
                    "comment_length": len(snippet["textDisplay"]),
                    "is_reply": False  # 顶级评论标记
                }
                
                # 计算发布后的天数
                pub_date = datetime.fromisoformat(snippet["publishedAt"].replace('Z', '+00:00'))
                days_since_upload = (datetime.now().astimezone() - pub_date).days
                comment["days_since_upload"] = days_since_upload
                
                comments.append(comment)
                pbar.update(1)
                
                if len(comments) >= max_comments:
                    break
                
                # 🔥 可选：收集高质量回复
                if item["snippet"]["totalReplyCount"] > 5:  # 回复多的评论往往讨论度高
                    replies = get_comment_replies(item["snippet"]["topLevelComment"]["id"], max_replies=3)
                    comments.extend(replies)

            next_page_token = response.get("nextPageToken")
            if not next_page_token:
                break
                
            # API友好延迟
            time.sleep(0.1)
            
        except Exception as e:
            print(f"Error fetching comments: {e}")
            break

    pbar.close()
    
    # 添加视频统计信息到每条评论
    for comment in comments:
        comment.update(video_info)
    
    return comments

def get_video_stats(video_id):
    """获取视频基础统计信息"""
    try:
        request = youtube.videos().list(
            part="statistics,snippet",
            id=video_id
        )
        response = request.execute()
        
        if response["items"]:
            stats = response["items"][0]["statistics"]
            snippet = response["items"][0]["snippet"]
            return {
                "video_view_count": int(stats.get("viewCount", 0)),
                "video_like_count": int(stats.get("likeCount", 0)),
                "video_comment_count": int(stats.get("commentCount", 0)),
                "video_upload_date": snippet["publishedAt"],
                "video_description": snippet["description"][:200] + "..."  # 截取描述
            }
    except Exception as e:
        print(f"Error fetching video stats: {e}")
        return {}

def get_comment_replies(comment_id, max_replies=3):
    """获取高质量评论的回复"""
    replies = []
    try:
        request = youtube.comments().list(
            part="snippet",
            parentId=comment_id,
            maxResults=max_replies,
            textFormat="plainText"
        )
        response = request.execute()
        
        for item in response["items"]:
            snippet = item["snippet"]
            reply = {
                "comment_id": item["id"],
                "parent_id": comment_id,
                "comment_text": snippet["textDisplay"],
                "author_name": snippet["authorDisplayName"],
                "like_count": snippet["likeCount"],
                "reply_count": 0,  # 回复的回复不考虑
                "published_at": snippet["publishedAt"],
                "comment_length": len(snippet["textDisplay"]),
                "is_reply": True
            }
            replies.append(reply)
    except Exception as e:
        print(f"Error fetching replies: {e}")
    
    return replies

# ✅ 执行抓取并保存
if __name__ == "__main__":
    print("🎵 NewJeans 'Ditto' Comment Analysis Project")
    print(f"Target: {VIDEOS[SELECTED_VIDEO]['title']}")
    print(f"Expected comments: {MAX_COMMENTS}")
    print("-" * 50)
    
    video_info = VIDEOS[SELECTED_VIDEO]
    data = get_comments_enhanced(
        video_info["id"], 
        MAX_COMMENTS, 
        video_info["title"]
    )
    
    # 保存为CSV
    df = pd.DataFrame(data)
    output_filename = f"newjeans_ditto_{SELECTED_VIDEO}_comments.csv"
    df.to_csv(output_filename, index=False, encoding="utf-8-sig")
    
    # 📊 快速统计
    print("\n✅ Collection Summary:")
    print(f"📝 Total comments collected: {len(df)}")
    print(f"💬 Average comment length: {df['comment_length'].mean():.1f} chars")
    print(f"👍 Average likes per comment: {df['like_count'].mean():.1f}")
    print(f"🔄 Comments with replies: {len(df[df['reply_count'] > 0])}")
    
    # 语言分布简单统计
    korean_comments = len(df[df['comment_text'].str.contains(r'[ㄱ-ㅎ가-힣]', na=False)])
    print(f"🇰🇷 Comments containing Korean: {korean_comments} ({korean_comments/len(df)*100:.1f}%)")
    
    print(f"\n💾 Data saved to: {output_filename}")
    print("🎯 Ready for sentiment analysis!")
    
    # 🔍 预览高点赞评论
    print("\n🔥 Top 3 Most Liked Comments:")
    top_comments = df.nlargest(3, 'like_count')[['comment_text', 'like_count']]
    for i, row in top_comments.iterrows():
        print(f"{row['like_count']} likes: {row['comment_text'][:100]}...")
