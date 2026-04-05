from __future__ import annotations

from typing import Any

import requests


YOUTUBE_API_BASE = "https://www.googleapis.com/youtube/v3"


def _request(endpoint: str, params: dict[str, Any]) -> dict[str, Any]:
    response = requests.get(f"{YOUTUBE_API_BASE}/{endpoint}", params=params, timeout=30)
    response.raise_for_status()
    return response.json()


def fetch_video_metadata(video_id: str, api_key: str) -> dict[str, Any]:
    payload = _request(
        "videos",
        {
            "part": "snippet,statistics",
            "id": video_id,
            "key": api_key,
        },
    )
    items = payload.get("items", [])
    if not items:
        raise ValueError("No video found for the provided YouTube video ID.")

    item = items[0]
    snippet = item.get("snippet", {})
    statistics = item.get("statistics", {})
    return {
        "video_id": video_id,
        "title": snippet.get("title", ""),
        "channel_title": snippet.get("channelTitle", ""),
        "published_at": snippet.get("publishedAt", ""),
        "view_count": int(statistics.get("viewCount", 0)),
        "comment_count": int(statistics.get("commentCount", 0)),
        "like_count": int(statistics.get("likeCount", 0)),
    }


def fetch_video_comments(video_id: str, api_key: str, max_comments: int = 100) -> list[dict[str, Any]]:
    comments: list[dict[str, Any]] = []
    next_page_token: str | None = None

    while len(comments) < max_comments:
        payload = _request(
            "commentThreads",
            {
                "part": "snippet",
                "videoId": video_id,
                "maxResults": min(100, max_comments - len(comments)),
                "order": "relevance",
                "textFormat": "plainText",
                "pageToken": next_page_token,
                "key": api_key,
            },
        )

        for item in payload.get("items", []):
            snippet = item.get("snippet", {}).get("topLevelComment", {}).get("snippet", {})
            comments.append(
                {
                    "comment_id": item.get("id"),
                    "author": snippet.get("authorDisplayName", "Unknown"),
                    "text": snippet.get("textDisplay", ""),
                    "like_count": int(snippet.get("likeCount", 0)),
                    "published_at": snippet.get("publishedAt", ""),
                    "updated_at": snippet.get("updatedAt", ""),
                }
            )
            if len(comments) >= max_comments:
                break

        next_page_token = payload.get("nextPageToken")
        if not next_page_token:
            break

    return comments
