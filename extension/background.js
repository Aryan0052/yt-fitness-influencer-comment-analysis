function extractVideoIdFromUrl(url) {
  if (!url) return "";

  try {
    const parsed = new URL(url);

    if (parsed.hostname.includes("youtu.be")) {
      return parsed.pathname.replace("/", "").trim();
    }

    if (parsed.pathname === "/watch") {
      return parsed.searchParams.get("v") || "";
    }

    if (parsed.pathname.startsWith("/shorts/")) {
      return parsed.pathname.split("/")[2] || "";
    }

    if (parsed.pathname.startsWith("/live/")) {
      return parsed.pathname.split("/")[2] || "";
    }
  } catch (error) {
    return "";
  }

  return "";
}

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message?.type !== "GET_ACTIVE_VIDEO_CONTEXT") {
    return false;
  }

  chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
    const activeTab = tabs?.[0];
    const url = activeTab?.url || "";
    const videoId = extractVideoIdFromUrl(url);

    sendResponse({
      videoId,
      url,
      title: activeTab?.title || ""
    });
  });

  return true;
});
