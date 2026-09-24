"use strict";
// No analytics, cookies, remote requests, or external libraries.
function localAssetPath(value) {
  return typeof value === "string" && /^assets\/videos\/[a-zA-Z0-9_./-]+$/.test(value) && !value.includes("..");
}
document.querySelectorAll("[data-demo]").forEach((card) => {
  const config = (window.KVLM_MEDIA || {})[card.dataset.demo];
  if (!config || !localAssetPath(config.src)) return;
  const container = card.querySelector(".demo-media");
  const poster = container.querySelector("img");
  const video = document.createElement("video");
  video.controls = true;
  video.playsInline = true;
  video.preload = "none";
  video.poster = poster.getAttribute("src");
  video.setAttribute("aria-label", card.querySelector("h3").textContent);
  video.src = config.src;
  if (localAssetPath(config.captions)) {
    const track = document.createElement("track");
    track.kind = "captions";
    track.label = "English";
    track.srclang = "en";
    track.src = config.captions;
    video.append(track);
  }
  video.addEventListener("error", () => {
    if (container.querySelector(".video-error")) return;
    const message = document.createElement("p");
    message.className = "video-error";
    message.textContent = "This video is currently unavailable.";
    container.append(message);
  });
  container.replaceChildren(video);
});
