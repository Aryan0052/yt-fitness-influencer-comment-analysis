const backendUrlInput = document.getElementById("backendUrl");
const apiKeyInput = document.getElementById("apiKey");
const videoIdInput = document.getElementById("videoId");
const detectedPageInput = document.getElementById("detectedPage");
const maxCommentsInput = document.getElementById("maxComments");
const analyzeButton = document.getElementById("analyzeButton");
const exportCsvButton = document.getElementById("exportCsvButton");
const exportReportButton = document.getElementById("exportReportButton");
const activeModelBanner = document.getElementById("activeModelBanner");
const activeModelName = document.getElementById("activeModelName");
const activeModelF1 = document.getElementById("activeModelF1");
const activeModelRecall = document.getElementById("activeModelRecall");
const statusText = document.getElementById("status");
const summaryPanel = document.getElementById("summaryPanel");
const sentimentPanel = document.getElementById("sentimentPanel");
const leaderboardPanel = document.getElementById("leaderboardPanel");
const wordCloudPanel = document.getElementById("wordCloudPanel");
const commentsPanel = document.getElementById("commentsPanel");
const videoMeta = document.getElementById("videoMeta");
const countCards = document.getElementById("countCards");
const chartBars = document.getElementById("chartBars");
const sentimentDonut = document.getElementById("sentimentDonut");
const donutTotal = document.getElementById("donutTotal");
const analysisBadge = document.getElementById("analysisBadge");
const commentList = document.getElementById("commentList");
const wordCloud = document.getElementById("wordCloud");
const cloudFilter = document.getElementById("cloudFilter");
const commentCountBadge = document.getElementById("commentCountBadge");
const modelSummary = document.getElementById("modelSummary");
const modelBadge = document.getElementById("modelBadge");
const leaderboardList = document.getElementById("leaderboardList");
const insightPanel = document.getElementById("insightPanel");
const insightCards = document.getElementById("insightCards");
const confidenceBars = document.getElementById("confidenceBars");
const engagementPanel = document.getElementById("engagementPanel");
const engagementBars = document.getElementById("engagementBars");
const lengthBars = document.getElementById("lengthBars");
const audiencePanel = document.getElementById("audiencePanel");
const authorList = document.getElementById("authorList");
const termList = document.getElementById("termList");
const timelinePanel = document.getElementById("timelinePanel");
const timelineChart = document.getElementById("timelineChart");
const commentFilter = document.getElementById("commentFilter");
const commentSearch = document.getElementById("commentSearch");

const sentimentColors = {
  positive: "#22c55e",
  neutral: "#f59e0b",
  negative: "#ef4444"
};

let latestResult = null;
let latestLeaderboard = [];
let autoTriggeredForVideoId = "";
let autoAnalyzeTimer = null;

function setStatus(message, isError = false) {
  statusText.textContent = message;
  statusText.style.color = isError ? "#f87171" : "#cbd5e1";
}

function formatNumber(value) {
  return new Intl.NumberFormat().format(value);
}

function formatPercent(value) {
  return `${value.toFixed(1)}%`;
}

function createCard(label, value) {
  const wrapper = document.createElement("div");
  wrapper.className = "meta-card";
  wrapper.innerHTML = `<div class="mini-label">${label}</div><div class="mini-value">${value}</div>`;
  return wrapper;
}

function createInsightCard(title, value, copy) {
  const wrapper = document.createElement("div");
  wrapper.className = "insight-card";
  wrapper.innerHTML = `
    <div class="insight-title">${title}</div>
    <div class="insight-value">${value}</div>
    <div class="insight-copy">${copy}</div>
  `;
  return wrapper;
}

function createStackItem(title, subtitle, value) {
  const wrapper = document.createElement("div");
  wrapper.className = "stack-item";
  wrapper.innerHTML = `
    <div class="stack-item-main">
      <div class="stack-item-title">${title}</div>
      <div class="stack-item-subtitle">${subtitle}</div>
    </div>
    <div class="stack-item-value">${value}</div>
  `;
  return wrapper;
}

function downloadTextFile(filename, content, type = "text/plain") {
  const blob = new Blob([content], { type });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  anchor.click();
  URL.revokeObjectURL(url);
}

function getTermsFromComments(comments, topN = 6) {
  const stopwords = new Set([
    "the", "and", "for", "this", "that", "with", "you", "your", "was", "are", "but", "have",
    "just", "from", "they", "them", "very", "really", "about", "would", "there", "their",
    "what", "when", "where", "which", "into", "also", "than", "then", "because", "like"
  ]);
  const counts = new Map();

  comments.forEach((comment) => {
    String(comment.text || "")
      .toLowerCase()
      .replace(/[^a-z0-9\s]/g, " ")
      .split(/\s+/)
      .filter((token) => token.length >= 4 && !stopwords.has(token))
      .forEach((token) => {
        counts.set(token, (counts.get(token) || 0) + 1);
      });
  });

  return [...counts.entries()]
    .sort((a, b) => b[1] - a[1])
    .slice(0, topN)
    .map(([term, count]) => ({ term, count }));
}

function renderVideoMeta(video) {
  videoMeta.innerHTML = "";
  videoMeta.appendChild(createCard("Title", video.title || "Unknown"));
  videoMeta.appendChild(createCard("Channel", video.channel_title || "Unknown"));
  videoMeta.appendChild(createCard("Views", formatNumber(video.view_count || 0)));
  videoMeta.appendChild(createCard("API Comment Count", formatNumber(video.comment_count || 0)));
}

function renderCounts(summary) {
  countCards.innerHTML = "";
  chartBars.innerHTML = "";
  donutTotal.textContent = String(Object.values(summary.sentiment_counts || {}).reduce((acc, value) => acc + value, 0));

  const positive = summary.sentiment_percentages?.positive || 0;
  const neutral = summary.sentiment_percentages?.neutral || 0;
  const negative = summary.sentiment_percentages?.negative || 0;
  sentimentDonut.style.background = `conic-gradient(
    ${sentimentColors.positive} 0 ${positive}%,
    ${sentimentColors.neutral} ${positive}% ${positive + neutral}%,
    ${sentimentColors.negative} ${positive + neutral}% 100%
  )`;

  ["positive", "neutral", "negative"].forEach((label) => {
    const count = summary.sentiment_counts?.[label] || 0;
    const percentage = summary.sentiment_percentages?.[label] || 0;

    const card = document.createElement("div");
    card.className = `count-card ${label}`;
    card.innerHTML = `
      <div class="mini-label">${label}</div>
      <div class="mini-value" style="color:${sentimentColors[label]}">${count}</div>
      <div style="margin-top:4px; font-size:12px; color:#94a3b8;">${formatPercent(percentage)}</div>
    `;
    countCards.appendChild(card);

    const bar = document.createElement("div");
    bar.className = "bar-row";
    bar.innerHTML = `
      <div class="bar-label">
        <span>${label}</span>
        <span>${formatPercent(percentage)}</span>
      </div>
      <div class="bar-track">
        <div class="bar-fill" style="width:${percentage}%; background:${sentimentColors[label]}"></div>
      </div>
    `;
    chartBars.appendChild(bar);
  });
}

function renderInsights(payload) {
  const comments = payload.comments || [];
  const counts = payload.sentiment_counts || {};
  const dominantSentiment = Object.entries(counts).sort((a, b) => b[1] - a[1])[0]?.[0] || "unknown";
  const avgConfidence = comments.length
    ? comments.reduce((sum, comment) => sum + (comment.confidence || 0), 0) / comments.length
    : 0;
  const mostLikedComment = [...comments].sort((a, b) => (b.like_count || 0) - (a.like_count || 0))[0];
  const engagedSentiment = ["positive", "neutral", "negative"]
    .map((label) => {
      const group = comments.filter((comment) => comment.predicted_sentiment === label);
      const avgLikes = group.length
        ? group.reduce((sum, comment) => sum + (comment.like_count || 0), 0) / group.length
        : 0;
      return { label, avgLikes };
    })
    .sort((a, b) => b.avgLikes - a.avgLikes)[0];

  insightCards.innerHTML = "";
  insightCards.appendChild(createInsightCard("Dominant Sentiment", dominantSentiment[0].toUpperCase() + dominantSentiment.slice(1), "The strongest overall audience mood across the analyzed comments."));
  insightCards.appendChild(createInsightCard("Average Confidence", `${(avgConfidence * 100).toFixed(1)}%`, "How confident the best-trained model feels across this video's comment set."));
  insightCards.appendChild(createInsightCard("Highest Engagement Mood", engagedSentiment?.label ? engagedSentiment.label[0].toUpperCase() + engagedSentiment.label.slice(1) : "N/A", "Sentiment group with the strongest average like count."));
  insightCards.appendChild(createInsightCard("Top Comment Signal", mostLikedComment ? `${formatNumber(mostLikedComment.like_count || 0)} likes` : "N/A", mostLikedComment ? `"${mostLikedComment.text.slice(0, 55)}${mostLikedComment.text.length > 55 ? "..." : ""}"` : "No comments were returned."));

  const buckets = [
    { label: "High confidence", count: comments.filter((comment) => (comment.confidence || 0) >= 0.8).length, color: sentimentColors.positive },
    { label: "Medium confidence", count: comments.filter((comment) => (comment.confidence || 0) >= 0.6 && (comment.confidence || 0) < 0.8).length, color: sentimentColors.neutral },
    { label: "Low confidence", count: comments.filter((comment) => (comment.confidence || 0) < 0.6).length, color: sentimentColors.negative }
  ];

  confidenceBars.innerHTML = "";
  const total = comments.length || 1;
  buckets.forEach((bucket) => {
    const percentage = (bucket.count / total) * 100;
    const row = document.createElement("div");
    row.className = "bar-row";
    row.innerHTML = `
      <div class="bar-label">
        <span>${bucket.label}</span>
        <span>${bucket.count} comments</span>
      </div>
      <div class="bar-track">
        <div class="bar-fill" style="width:${percentage}%; background:${bucket.color}"></div>
      </div>
    `;
    confidenceBars.appendChild(row);
  });

  const dominantPercent = payload.sentiment_percentages?.[dominantSentiment] || 0;
  analysisBadge.textContent = `${dominantSentiment.toUpperCase()} ${formatPercent(dominantPercent)}`;
  insightPanel.classList.remove("hidden");
}

function renderWordCloud(filterKey = "overall") {
  wordCloud.innerHTML = "";
  const terms = latestResult?.top_terms?.[filterKey] || [];

  if (!terms.length) {
    wordCloud.textContent = "No terms available for this selection.";
    return;
  }

  terms.forEach((entry) => {
    const chip = document.createElement("span");
    chip.className = "word-chip";
    chip.style.fontSize = `${entry.font_size}px`;
    chip.textContent = entry.term;
    wordCloud.appendChild(chip);
  });
}

function renderComments(comments) {
  commentList.innerHTML = "";
  commentCountBadge.textContent = `${comments.length} comments`;

  comments.forEach((comment) => {
    const card = document.createElement("article");
    card.className = "comment-card";
    card.innerHTML = `
      <div class="comment-meta">
        <span>${comment.author || "Unknown author"}</span>
        <span class="pill" style="background:${sentimentColors[comment.predicted_sentiment] || "#64748b"}">${comment.predicted_sentiment}</span>
      </div>
      <div style="font-size:13px; line-height:1.5; margin-bottom:8px; color:#e2e8f0;">${comment.text}</div>
      <div class="comment-meta">
        <span>Likes: ${formatNumber(comment.like_count || 0)}</span>
        <span>Confidence: ${Math.round((comment.confidence || 0) * 100)}%</span>
      </div>
    `;
    commentList.appendChild(card);
  });

  if (!comments.length) {
    commentList.innerHTML = `<div class="stack-item"><div class="stack-item-main"><div class="stack-item-title">No comments match this filter</div><div class="stack-item-subtitle">Try another sentiment or clear the search.</div></div><div class="stack-item-value">0</div></div>`;
  }
}

function renderFilteredComments() {
  if (!latestResult?.comments) {
    return;
  }

  const filterValue = commentFilter.value;
  const searchValue = commentSearch.value.trim().toLowerCase();
  const filtered = latestResult.comments.filter((comment) => {
    const matchesSentiment = filterValue === "all" || comment.predicted_sentiment === filterValue;
    const haystack = `${comment.author || ""} ${comment.text || ""}`.toLowerCase();
    const matchesSearch = !searchValue || haystack.includes(searchValue);
    return matchesSentiment && matchesSearch;
  });

  renderComments(filtered);
}

function renderEngagementAnalytics(payload) {
  const comments = payload.comments || [];
  const sentimentBuckets = ["positive", "neutral", "negative"].map((label) => {
    const group = comments.filter((comment) => comment.predicted_sentiment === label);
    const avgLikes = group.length
      ? group.reduce((sum, comment) => sum + (comment.like_count || 0), 0) / group.length
      : 0;
    return { label, avgLikes };
  });

  const maxLikes = Math.max(...sentimentBuckets.map((item) => item.avgLikes), 1);
  engagementBars.innerHTML = "";
  sentimentBuckets.forEach((item) => {
    const row = document.createElement("div");
    row.className = "bar-row";
    row.innerHTML = `
      <div class="bar-label">
        <span>${item.label}</span>
        <span>${item.avgLikes.toFixed(1)} avg likes</span>
      </div>
      <div class="bar-track">
        <div class="bar-fill" style="width:${(item.avgLikes / maxLikes) * 100}%; background:${sentimentColors[item.label]}"></div>
      </div>
    `;
    engagementBars.appendChild(row);
  });

  const lengthBuckets = [
    { label: "Short", count: comments.filter((comment) => String(comment.text || "").trim().split(/\s+/).length <= 8).length, color: "#38bdf8" },
    { label: "Medium", count: comments.filter((comment) => { const words = String(comment.text || "").trim().split(/\s+/).length; return words > 8 && words <= 18; }).length, color: "#f59e0b" },
    { label: "Long", count: comments.filter((comment) => String(comment.text || "").trim().split(/\s+/).length > 18).length, color: "#a855f7" }
  ];

  const total = comments.length || 1;
  lengthBars.innerHTML = "";
  lengthBuckets.forEach((item) => {
    const percentage = (item.count / total) * 100;
    const row = document.createElement("div");
    row.className = "bar-row";
    row.innerHTML = `
      <div class="bar-label">
        <span>${item.label}</span>
        <span>${item.count} comments</span>
      </div>
      <div class="bar-track">
        <div class="bar-fill" style="width:${percentage}%; background:${item.color}"></div>
      </div>
    `;
    lengthBars.appendChild(row);
  });

  engagementPanel.classList.remove("hidden");
}

function renderAudienceSignals(payload) {
  const comments = payload.comments || [];
  const authorCounts = new Map();

  comments.forEach((comment) => {
    const author = comment.author || "Unknown author";
    authorCounts.set(author, (authorCounts.get(author) || 0) + 1);
  });

  authorList.innerHTML = "";
  [...authorCounts.entries()].sort((a, b) => b[1] - a[1]).slice(0, 5).forEach(([author, count]) => {
    authorList.appendChild(createStackItem(author, "comments contributed", count));
  });
  if (!authorList.children.length) {
    authorList.appendChild(createStackItem("No author data", "No comments were returned", "0"));
  }

  termList.innerHTML = "";
  getTermsFromComments(comments, 6).forEach((item) => {
    termList.appendChild(createStackItem(item.term, "repeat frequency", item.count));
  });
  if (!termList.children.length) {
    termList.appendChild(createStackItem("No repeated terms", "Try another video", "0"));
  }

  audiencePanel.classList.remove("hidden");
}

function renderTimeline(payload) {
  const comments = payload.comments || [];
  timelineChart.innerHTML = "";

  comments.slice(0, 40).forEach((comment) => {
    const bar = document.createElement("div");
    bar.className = "timeline-bar";
    bar.style.height = `${Math.max(18, (comment.confidence || 0) * 120)}px`;
    bar.style.background = sentimentColors[comment.predicted_sentiment] || "#64748b";
    bar.title = `${comment.predicted_sentiment} | ${Math.round((comment.confidence || 0) * 100)}% | ${comment.author || "Unknown"}`;
    timelineChart.appendChild(bar);
  });

  timelinePanel.classList.remove("hidden");
}

function exportCommentsAsCsv() {
  if (!latestResult?.comments?.length) {
    setStatus("Run an analysis first, then export the CSV.", true);
    return;
  }

  const lines = [
    "author,text,predicted_sentiment,confidence,like_count,published_at",
    ...latestResult.comments.map((comment) => {
      const safeText = `"${String(comment.text || "").replace(/"/g, '""')}"`;
      return [
        comment.author || "",
        safeText,
        comment.predicted_sentiment || "",
        comment.confidence || 0,
        comment.like_count || 0,
        comment.published_at || ""
      ].join(",");
    })
  ];

  downloadTextFile(`${latestResult.video?.video_id || "youtube-video"}-sentiment-analysis.csv`, lines.join("\n"), "text/csv");
  setStatus("CSV exported successfully.");
}

function exportSummaryReport() {
  if (!latestResult?.comments?.length) {
    setStatus("Run an analysis first, then export the report.", true);
    return;
  }

  const video = latestResult.video || {};
  const counts = latestResult.sentiment_counts || {};
  const percentages = latestResult.sentiment_percentages || {};
  const report = [
    "FitScope Sentiment Report",
    "========================",
    "",
    `Video Title: ${video.title || "Unknown"}`,
    `Channel: ${video.channel_title || "Unknown"}`,
    `Video ID: ${video.video_id || "Unknown"}`,
    `Comments Analyzed: ${latestResult.total_comments_analyzed || 0}`,
    `Best Model: ${latestLeaderboard[0]?.model || modelBadge.textContent || "Unknown"}`,
    "",
    "Sentiment Summary",
    `Positive: ${counts.positive || 0} (${(percentages.positive || 0).toFixed(2)}%)`,
    `Neutral: ${counts.neutral || 0} (${(percentages.neutral || 0).toFixed(2)}%)`,
    `Negative: ${counts.negative || 0} (${(percentages.negative || 0).toFixed(2)}%)`,
    "",
    "Top Terms",
    ...(latestResult.top_terms?.overall || []).slice(0, 10).map((item) => `- ${item.term}: ${item.count}`),
    "",
    "Top Comments by Likes",
    ...[...latestResult.comments].sort((a, b) => (b.like_count || 0) - (a.like_count || 0)).slice(0, 5).map((comment) => `- [${comment.predicted_sentiment}] ${comment.author || "Unknown"} (${comment.like_count || 0} likes): ${comment.text}`)
  ].join("\n");

  downloadTextFile(`${latestResult.video?.video_id || "youtube-video"}-sentiment-report.txt`, report, "text/plain");
  setStatus("Report exported successfully.");
}

async function loadSavedSettings() {
  const saved = await chrome.storage.local.get(["backendUrl", "apiKey", "maxComments"]);
  if (saved.backendUrl) backendUrlInput.value = saved.backendUrl;
  if (saved.apiKey) apiKeyInput.value = saved.apiKey;
  if (saved.maxComments) maxCommentsInput.value = saved.maxComments;
}

async function detectActiveYouTubeVideo() {
  try {
    const context = await chrome.runtime.sendMessage({ type: "GET_ACTIVE_VIDEO_CONTEXT" });
    const videoId = context?.videoId || "";
    detectedPageInput.value = context?.url || "";

    if (videoId) {
      videoIdInput.value = videoId;
      return videoId;
    }

    videoIdInput.value = "";
    return "";
  } catch (error) {
    videoIdInput.value = "";
    detectedPageInput.value = "";
    return "";
  }
}

async function fetchModelSummary() {
  const backendUrl = backendUrlInput.value.trim().replace(/\/$/, "");
  if (!backendUrl) return;

  try {
    const response = await fetch(`${backendUrl}/model/summary`);
    const payload = await response.json();
    if (!response.ok) return;

    const bestModel = payload.best_model || {};
    if (!bestModel.model) return;

    latestLeaderboard = payload.leaderboard || [];
    activeModelBanner.classList.remove("hidden");
    activeModelName.textContent = payload.active_model || bestModel.model;
    activeModelF1.textContent = `${((bestModel.f1_macro || 0) * 100).toFixed(2)}%`;
    activeModelRecall.textContent = `${((bestModel.recall_macro || 0) * 100).toFixed(2)}%`;
    leaderboardPanel.classList.remove("hidden");
    modelBadge.textContent = bestModel.model;
    modelSummary.innerHTML = "";
    modelSummary.appendChild(createCard("Accuracy", `${((bestModel.accuracy || 0) * 100).toFixed(2)}%`));
    modelSummary.appendChild(createCard("Macro F1", `${((bestModel.f1_macro || 0) * 100).toFixed(2)}%`));
    modelSummary.appendChild(createCard("Precision", `${((bestModel.precision_macro || 0) * 100).toFixed(2)}%`));
    modelSummary.appendChild(createCard("Recall", `${((bestModel.recall_macro || 0) * 100).toFixed(2)}%`));

    leaderboardList.innerHTML = "";
    latestLeaderboard.slice(0, 5).forEach((model, index) => {
      leaderboardList.appendChild(createStackItem(`#${index + 1} ${model.model}`, `Accuracy ${(Number(model.accuracy || 0) * 100).toFixed(2)}%`, `${(Number(model.f1_macro || 0) * 100).toFixed(2)} F1`));
    });
  } catch (error) {
    // Keep hidden if backend unavailable.
  }
}

async function saveSettings() {
  await chrome.storage.local.set({
    backendUrl: backendUrlInput.value.trim(),
    apiKey: apiKeyInput.value.trim(),
    maxComments: maxCommentsInput.value
  });
}

async function scheduleAutoAnalyze() {
  if (autoAnalyzeTimer) {
    clearTimeout(autoAnalyzeTimer);
  }

  autoAnalyzeTimer = setTimeout(async () => {
    const videoId = await detectActiveYouTubeVideo();
    const apiKey = apiKeyInput.value.trim();
    if (!videoId || !apiKey) {
      return;
    }
    if (autoTriggeredForVideoId === videoId) {
      return;
    }
    autoTriggeredForVideoId = videoId;
    analyzeVideo();
  }, 250);
}

async function analyzeVideo() {
  const backendUrl = backendUrlInput.value.trim().replace(/\/$/, "");
  const apiKey = apiKeyInput.value.trim();
  const detectedVideoId = await detectActiveYouTubeVideo();
  const videoId = detectedVideoId || videoIdInput.value.trim();
  const maxComments = Number(maxCommentsInput.value);

  if (!backendUrl || !apiKey || !videoId) {
    setStatus("Open a YouTube video tab, then reopen the popup and try again.", true);
    return;
  }

  analyzeButton.disabled = true;
  setStatus("Fetching YouTube comments and scoring sentiment...");

  try {
    await saveSettings();
    const response = await fetch(`${backendUrl}/youtube/analyze`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        api_key: apiKey,
        video_id: videoId,
        max_comments: maxComments
      })
    });

    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.detail || "Request failed.");
    }

    latestResult = payload;
    renderVideoMeta(payload.video);
    renderCounts(payload);
    renderInsights(payload);
    renderEngagementAnalytics(payload);
    renderAudienceSignals(payload);
    renderTimeline(payload);
    renderWordCloud(cloudFilter.value);
    renderFilteredComments();

    summaryPanel.classList.remove("hidden");
    sentimentPanel.classList.remove("hidden");
    wordCloudPanel.classList.remove("hidden");
    commentsPanel.classList.remove("hidden");
    setStatus(`Analyzed ${payload.total_comments_analyzed} comments successfully.`);
  } catch (error) {
    setStatus(error.message || "Something went wrong.", true);
  } finally {
    analyzeButton.disabled = false;
  }
}

cloudFilter.addEventListener("change", () => renderWordCloud(cloudFilter.value));
analyzeButton.addEventListener("click", analyzeVideo);
exportCsvButton.addEventListener("click", exportCommentsAsCsv);
exportReportButton.addEventListener("click", exportSummaryReport);
commentFilter.addEventListener("change", renderFilteredComments);
commentSearch.addEventListener("input", renderFilteredComments);
apiKeyInput.addEventListener("input", async () => {
  await saveSettings();
  scheduleAutoAnalyze();
});
backendUrlInput.addEventListener("input", async () => {
  await saveSettings();
  fetchModelSummary();
});
maxCommentsInput.addEventListener("input", async () => {
  await saveSettings();
});

async function initializePopup() {
  await loadSavedSettings();
  await fetchModelSummary();
  const videoId = await detectActiveYouTubeVideo();

  if (videoId) {
    if (apiKeyInput.value.trim()) {
      setStatus("YouTube video detected. Auto-analyzing with the saved API key...");
      scheduleAutoAnalyze();
    } else {
      setStatus("YouTube video detected. Paste your API key once to enable auto-analysis.");
    }
  } else {
    setStatus("Open a YouTube video tab to detect the video ID automatically.", true);
  }
}

initializePopup();
