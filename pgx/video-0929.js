"use strict";
// Native playback remains available without JavaScript.
const videos = Array.from(document.querySelectorAll("video"));
videos.forEach((video) => {
  const chapters = Array.from(document.querySelectorAll("[data-video-target]"))
    .filter((button) => button.dataset.videoTarget === video.id);
  let pendingSeek = null;
  function applySeek() {
    if (pendingSeek === null || video.readyState < 1) return;
    video.currentTime = Math.min(pendingSeek, video.duration || pendingSeek);
    pendingSeek = null;
  }
  video.addEventListener("loadedmetadata", applySeek);
  chapters.forEach((button) => {
    button.disabled = false;
    button.addEventListener("click", () => {
      pendingSeek = Number(button.dataset.seek);
      if (video.readyState < 1) video.load();
      else applySeek();
      video.play().catch(() => {});
    });
  });
  video.addEventListener("timeupdate", () => {
    const current = chapters.filter((b) => Number(b.dataset.seek) <= video.currentTime).pop();
    chapters.forEach((b) => {
      if (b === current) b.setAttribute("aria-current", "true");
      else b.removeAttribute("aria-current");
    });
  });
});
videos.forEach((video) => {
  video.addEventListener("play", () => {
    videos.forEach((other) => { if (other !== video) other.pause(); });
  });
  const showError = () => {
    const frame = video.closest(".video-frame");
    if (frame.querySelector(".video-error")) return;
    const message = document.createElement("p");
    message.className = "video-error";
    message.setAttribute("role", "status");
    message.append("Video unavailable. ");
    const link = document.createElement("a");
    link.href = video.querySelector("source").getAttribute("src");
    link.textContent = "Open the MP4 directly";
    message.append(link);
    frame.append(message);
  };
  video.addEventListener("error", showError);
  video.querySelector("source").addEventListener("error", showError);
});
