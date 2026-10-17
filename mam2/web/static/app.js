"use strict";
const $ = (id) => document.getElementById(id);
const canvas = $("canvas");
const ctx = canvas.getContext("2d");
// Canvas prompts share the palette defined in style.css.
const theme = getComputedStyle(document.documentElement);
const selectionColors = {
  mask: theme.getPropertyValue("--selection-rgb").split(",").map(Number),
  box: theme.getPropertyValue("--selection-outline").trim(),
  positive: theme.getPropertyValue("--selection-positive").trim(),
  negative: theme.getPropertyValue("--selection-negative").trim(),
  markPositive: theme.getPropertyValue("--selection-mark-positive").trim(),
  markNegative: theme.getPropertyValue("--selection-mark-negative").trim(),
  border: theme.getPropertyValue("--surface").trim(),
};
let session = null,
  frame = null,
  overlay = null;
let actions = [],
  drag = null,
  running = false,
  previewBusy = false,
  needsPreview = false;
let approved = false,
  foregroundPixels = 0,
  view = "foreground",
  pollTimer = null,
  uploading = false,
  resetting = false;
let resultVersion = 0,
  selectionRevision = 0,
  frameIndex = 0,
  frameBusy = false,
  seekRevision = 0,
  seekTimer = null,
  options = null,
  promptError = false,
  loadingExample = null;
const promptedFrames = new Map();
const playbackFormat = document
  .createElement("video")
  .canPlayType('video/mp4;codecs="avc1.42E01E"')
  ? "mp4"
  : "webm";
function notice(message, error = false) {
  $("notice").textContent = message;
  $("notice").classList.toggle("error", error);
}
async function api(path, options = {}) {
  const response = await fetch(path, options);
  if (!response.ok) {
    const data = await response.json().catch(() => ({}));
    const detail = data.detail;
    const error = new Error(
      typeof detail === "string"
        ? detail
        : Array.isArray(detail)
          ? detail.map((x) => x.msg).join("; ")
          : "Request failed. Please try again.",
    );
    error.status = response.status;
    throw error;
  }
  return response.json();
}
function loadImage(url) {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => resolve(image);
    image.onerror = () => reject(new Error("Could not load the preview."));
    image.src = url;
  });
}
function updateControls() {
  const locked = running || uploading || resetting;
  $("start").disabled =
    !session ||
    locked ||
    frameBusy ||
    previewBusy ||
    needsPreview ||
    !options ||
    promptError ||
    ![...promptedFrames.values()].some((pixels) => pixels > 0);
  $("clear-video").disabled = !session || locked || previewBusy || frameBusy;
  $("clear-prompt").disabled =
    !session || locked || frameBusy || (!actions.length && !promptError);
  $("frame-slider").disabled = !session || locked || previewBusy || promptError;
  $("resolution").disabled = locked || !options;
  for (const button of $("example-list").children) {
    const selected = session?.example_id === button.dataset.example;
    button.disabled = locked || previewBusy || frameBusy || needsPreview;
    button.classList.toggle("selected", selected);
    button.setAttribute("aria-pressed", String(selected));
    button.setAttribute(
      "aria-busy",
      String(loadingExample === button.dataset.example),
    );
  }
}
function updateTimeline(index = frameIndex) {
  if (!session) return;
  $("frame-slider").value = index;
  $("frame-position").textContent = `Frame ${index + 1} / ${session.frames}`;
  const ms = Math.round((index / session.fps) * 1000);
  $("time-position").textContent =
    `${String(Math.floor(ms / 60000)).padStart(2, "0")}:${String(Math.floor(ms / 1000) % 60).padStart(2, "0")}.${String(ms % 1000).padStart(3, "0")}`;
  $("prompt-count").textContent = promptedFrames.size
    ? `${promptedFrames.size} prompted frame${promptedFrames.size === 1 ? "" : "s"}`
    : "No prompted frames";
  $("prompt-markers").replaceChildren(
    ...[...promptedFrames.keys()].map((index) => {
      const marker = document.createElement("i");
      marker.style.left = `${(100 * index) / Math.max(1, session.frames - 1)}%`;
      return marker;
    }),
  );
}
function prompt() {
  const points = actions.filter((a) => a.type === "point");
  const box = actions.findLast((a) => a.type === "box");
  return {
    frame_index: frameIndex,
    points: points.map((a) => a.xy),
    labels: points.map((a) => a.label),
    box: box ? box.xy : null,
  };
}
function restoreActions(prompt) {
  if (!prompt) return [];
  const points = (prompt.points || []).map((xy, i) => ({
    type: "point",
    xy,
    label: prompt.labels[i],
  }));
  return prompt.box ? [{ type: "box", xy: prompt.box }, ...points] : points;
}
function loading(label = null) {
  $("selection-loading").hidden = !label;
  if (label) $("loading-label").textContent = label;
}
async function seek(index, revision) {
  const currentSession = session;
  try {
    const data = await api(`/api/session/${currentSession.id}/frame/${index}`);
    const [image, mask] = await Promise.all([
      loadImage(data.image),
      data.mask ? renderMask(data.mask) : null,
    ]);
    // Rapid slider movements can finish out of order. Only the latest wins.
    if (revision !== seekRevision || session !== currentSession) return;
    frameIndex = index;
    frame = image;
    overlay = mask;
    actions = restoreActions(data.prompt);
    approved = !!data.prompt;
    foregroundPixels = data.foreground_pixels;
    selectionRevision++;
    updateTimeline();
    draw();
    notice(
      data.prompt
        ? "Saved selection restored. Add points or redraw the box to refine it."
        : "Click or drag to select your subject on this frame.",
    );
  } catch (error) {
    if (revision === seekRevision) {
      updateTimeline();
      notice(error.message, true);
    }
  } finally {
    if (revision === seekRevision) {
      frameBusy = false;
      loading();
      updateControls();
    }
  }
}
$("frame-slider").addEventListener("input", (event) => {
  if (!session || running || previewBusy || uploading || resetting) return;
  const index = Number(event.target.value),
    revision = ++seekRevision;
  clearTimeout(seekTimer);
  drag = null;
  frameBusy = true;
  updateTimeline(index);
  updateControls();
  loading("Loading frame…");
  seekTimer = setTimeout(() => seek(index, revision), 60);
});
function draw() {
  if (!frame) return;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(frame, 0, 0);
  if (overlay) {
    ctx.globalAlpha = approved ? 0.4 : 0.18;
    ctx.drawImage(overlay, 0, 0);
    ctx.globalAlpha = 1;
  }
  const scale =
    canvas.width / Math.max(1, canvas.getBoundingClientRect().width);
  function box(xy) {
    ctx.strokeStyle = selectionColors.box;
    ctx.lineWidth = 2 * scale;
    ctx.setLineDash([]);
    ctx.strokeRect(xy[0], xy[1], xy[2] - xy[0], xy[3] - xy[1]);
  }
  const selectedBox = actions.findLast((a) => a.type === "box");
  if (selectedBox) box(selectedBox.xy);
  actions
    .filter((a) => a.type === "point")
    .forEach((a) => {
      const [x, y] = a.xy,
        radius = 6 * scale;
      ctx.fillStyle = a.label
        ? selectionColors.positive
        : selectionColors.negative;
      ctx.strokeStyle = a.label
        ? selectionColors.border
        : selectionColors.markNegative;
      ctx.lineWidth = 1.5 * scale;
      ctx.beginPath();
      ctx.arc(x, y, radius, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();
      ctx.strokeStyle = a.label
        ? selectionColors.markPositive
        : selectionColors.markNegative;
      ctx.lineWidth = scale;
      ctx.beginPath();
      ctx.moveTo(x - radius / 2, y);
      ctx.lineTo(x + radius / 2, y);
      if (a.label) {
        ctx.moveTo(x, y - radius / 2);
        ctx.lineTo(x, y + radius / 2);
      }
      ctx.stroke();
    });
  if (drag && isBox(drag))
    box([drag.start[0], drag.start[1], drag.end[0], drag.end[1]]);
}
async function renderMask(url) {
  const image = await loadImage(url);
  const layer = document.createElement("canvas");
  layer.width = canvas.width;
  layer.height = canvas.height;
  const context = layer.getContext("2d", { willReadFrequently: true });
  context.drawImage(image, 0, 0);
  const pixels = context.getImageData(0, 0, layer.width, layer.height);
  for (let i = 0; i < pixels.data.length; i += 4) {
    pixels.data[i + 3] = pixels.data[i];
    pixels.data[i] = selectionColors.mask[0];
    pixels.data[i + 1] = selectionColors.mask[1];
    pixels.data[i + 2] = selectionColors.mask[2];
  }
  context.putImageData(pixels, 0, 0);
  return layer;
}
async function requestPreview() {
  needsPreview = true;
  approved = false;
  updateControls();
  draw();
  if (previewBusy) return;
  previewBusy = true;
  updateControls();
  loading("Updating mask…");
  try {
    while (needsPreview) {
      needsPreview = false;
      const snapshot = JSON.stringify(prompt()),
        revision = selectionRevision;
      const hasActions = actions.length > 0;
      const path = `/api/session/${session.id}/prompt`;
      const data = hasActions
        ? await api(path, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: snapshot,
          })
        : await api(`${path}/${frameIndex}`, { method: "DELETE" });
      if (hasActions) promptedFrames.set(frameIndex, data.foreground_pixels);
      else promptedFrames.delete(frameIndex);
      updateTimeline();
      if (revision !== selectionRevision) continue;
      const mask = hasActions ? await renderMask(data.mask) : null;
      if (revision !== selectionRevision) continue;
      overlay = mask;
      foregroundPixels = hasActions ? data.foreground_pixels : 0;
      approved = hasActions;
      promptError = false;
      draw();
      notice(
        !hasActions
          ? "Prompt cleared on this frame. Prompts on other frames are retained."
          : foregroundPixels
            ? "Selection ready. Refine here, prompt another frame, or start matting."
            : "No subject selected. Add an include point or draw a larger box.",
      );
    }
  } catch (error) {
    // Do not run with a stale server prompt after an unsuccessful edit.
    promptError = true;
    approved = false;
    notice(error.message, true);
  } finally {
    previewBusy = false;
    needsPreview = false;
    loading();
    updateTimeline();
    updateControls();
    draw();
  }
}
function changed() {
  selectionRevision++;
  approved = false;
  if (!actions.length) overlay = null;
  requestPreview();
}
function coordinate(event) {
  const bounds = canvas.getBoundingClientRect();
  return [
    Math.max(
      0,
      Math.min(
        canvas.width,
        ((event.clientX - bounds.left) / bounds.width) * canvas.width,
      ),
    ),
    Math.max(
      0,
      Math.min(
        canvas.height,
        ((event.clientY - bounds.top) / bounds.height) * canvas.height,
      ),
    ),
  ];
}
// Like MattePro: a click adds a point; a left-button drag adds a box.
// There are only foreground/background points; the middle mouse button is ignored.
function isBox(gesture) {
  const scale = canvas.getBoundingClientRect().width / canvas.width;
  return (
    gesture.button === 0 &&
    Math.abs(gesture.end[0] - gesture.start[0]) * scale >= 4 &&
    Math.abs(gesture.end[1] - gesture.start[1]) * scale >= 4
  );
}
canvas.addEventListener("contextmenu", (event) => event.preventDefault());
canvas.addEventListener("pointerdown", (event) => {
  if (
    !session ||
    running ||
    uploading ||
    resetting ||
    frameBusy ||
    ![0, 2].includes(event.button)
  )
    return;
  event.preventDefault();
  const xy = coordinate(event);
  drag = {
    start: xy,
    end: xy,
    button: event.button,
    label: event.button === 2 || event.shiftKey ? 0 : 1,
  };
  canvas.setPointerCapture(event.pointerId);
});
canvas.addEventListener("pointermove", (event) => {
  if (drag) {
    drag.end = coordinate(event);
    draw();
  }
});
canvas.addEventListener("pointerup", (event) => {
  if (!drag) return;
  const gesture = drag;
  gesture.end = coordinate(event);
  drag = null;
  if (isBox(gesture)) {
    const start = gesture.start,
      end = gesture.end;
    const xy = [
      Math.min(start[0], end[0]),
      Math.min(start[1], end[1]),
      Math.max(start[0], end[0]),
      Math.max(start[1], end[1]),
    ];
    // SAM2 accepts one box per object/frame. Redrawing replaces only the box.
    actions = actions.filter((a) => a.type !== "box");
    actions.push({ type: "box", xy });
  } else actions.push({ type: "point", xy: gesture.end, label: gesture.label });
  changed();
});
canvas.addEventListener("pointercancel", () => {
  drag = null;
  draw();
});
$("clear-prompt").onclick = () => {
  if ($("clear-prompt").disabled) return;
  actions = [];
  changed();
};
document.addEventListener("keydown", (event) => {
  if (
    !session ||
    running ||
    uploading ||
    resetting ||
    frameBusy ||
    ["INPUT", "SELECT", "TEXTAREA", "BUTTON"].includes(event.target.tagName)
  )
    return;
  if (
    (event.key === "Backspace" ||
      (event.key === "z" && (event.ctrlKey || event.metaKey))) &&
    actions.length
  ) {
    event.preventDefault();
    actions.pop();
    changed();
  }
});
function updateViewButtons() {
  ["foreground", "alpha"].forEach((name) => {
    const active = name === view;
    $(`${name}-view`).classList.toggle("active", active);
    $(`${name}-view`).setAttribute("aria-pressed", active);
  });
}
function resetResult() {
  clearTimeout(pollTimer);
  pollTimer = null;
  const player = $("result");
  player.onloadedmetadata = null;
  player.pause();
  player.removeAttribute("src");
  player.load();
  player.hidden = true;
  $("result-empty").hidden = false;
  $("processing").hidden = true;
  $("download").hidden = true;
  $("download").removeAttribute("href");
  $("download").removeAttribute("download");
  $("foreground-view").disabled = true;
  $("alpha-view").disabled = true;
  view = "foreground";
  updateViewButtons();
  $("result-badge").textContent = "Awaiting selection";
  $("output-meta").textContent = "Full video · original frame rate";
  $("progress-bar").style.width = "0%";
}
function resetWorkspace() {
  session = null;
  frame = null;
  overlay = null;
  actions = [];
  drag = null;
  approved = false;
  needsPreview = false;
  foregroundPixels = 0;
  selectionRevision++;
  seekRevision++;
  clearTimeout(seekTimer);
  frameIndex = 0;
  frameBusy = false;
  promptError = false;
  promptedFrames.clear();
  $("timeline").hidden = true;
  $("prompt-count").textContent = "No prompted frames";
  $("prompt-markers").replaceChildren();
  canvas.width = canvas.height = 0;
  canvas.hidden = true;
  $("dropzone").hidden = false;
  $("dropzone").classList.remove("dragging");
  $("selection-loading").hidden = true;
  $("file").value = "";
  resetResult();
}
$("clear-video").onclick = async () => {
  if ($("clear-video").disabled) return;
  resetting = true;
  updateControls();
  notice("Clearing your video and results…");
  try {
    await api(`/api/session/${session.id}`, { method: "DELETE" }).catch(
      (error) => {
        if (error.status !== 404) throw error;
      },
    );
    resetWorkspace();
    notice(
      "Workspace cleared. Upload a video or choose an example to start again.",
    );
  } catch (error) {
    notice(error.message, true);
  } finally {
    resetting = false;
    updateControls();
  }
};
async function loadVideoSession(createSession, message, exampleId = null) {
  if (
    running ||
    previewBusy ||
    frameBusy ||
    uploading ||
    resetting ||
    needsPreview
  )
    return;
  uploading = true;
  loadingExample = exampleId;
  updateControls();
  loading("Preparing video…");
  notice(message);
  let next = null;
  try {
    next = await createSession();
    const firstFrame = await loadImage(next.first_frame);
    const previous = session;
    session = next;
    frame = firstFrame;
    frameIndex = 0;
    promptError = false;
    seekRevision++;
    selectionRevision++;
    clearTimeout(seekTimer);
    promptedFrames.clear();
    $("timeline").hidden = false;
    $("frame-slider").max = next.frames - 1;
    updateTimeline();
    if (previous)
      api(`/api/session/${previous.id}`, { method: "DELETE" }).catch(() => {});
    resetResult();
    actions = [];
    overlay = null;
    approved = false;
    foregroundPixels = 0;
    drag = null;
    canvas.width = next.width;
    canvas.height = next.height;
    canvas.hidden = false;
    $("dropzone").hidden = true;
    draw();
    notice(
      "Video ready. Click for a point or drag a box. Use the timeline to prompt any frame.",
    );
  } catch (error) {
    if (next && next !== session)
      api(`/api/session/${next.id}`, { method: "DELETE" }).catch(() => {});
    notice(error.message, true);
  } finally {
    uploading = false;
    loadingExample = null;
    loading();
    $("file").value = "";
    updateControls();
  }
}
async function upload(file) {
  if (!file) return;
  return loadVideoSession(() => {
    const body = new FormData();
    body.append("video", file);
    return api("/api/upload", { method: "POST", body });
  }, "Uploading and preparing the video timeline…");
}
function renderExamples(examples) {
  $("example-list").replaceChildren(
    ...examples.map((example) => {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "example-card";
      button.dataset.example = example.id;
      button.setAttribute("aria-label", `Load ${example.name}`);
      const thumbnail = document.createElement("img");
      thumbnail.src = example.thumbnail;
      thumbnail.alt = "";
      thumbnail.loading = "lazy";
      const label = document.createElement("span");
      label.textContent = example.name;
      button.append(thumbnail, label);
      button.onclick = () =>
        loadVideoSession(
          () =>
            api(`/api/examples/${encodeURIComponent(example.id)}/select`, {
              method: "POST",
            }),
          `Loading ${example.name} and preparing the video timeline…`,
          example.id,
        );
      return button;
    }),
  );
  $("examples").hidden = examples.length === 0;
  updateControls();
}
$("file").onchange = (event) => upload(event.target.files[0]);
$("dropzone").onclick = () => {
  if (!uploading && !resetting) $("file").click();
};
$("dropzone").onkeydown = (event) => {
  if (event.key === "Enter" || event.key === " ") {
    event.preventDefault();
    if (!uploading && !resetting) $("file").click();
  }
};
const dropzone = $("dropzone");
["dragenter", "dragover"].forEach((type) =>
  dropzone.addEventListener(type, (event) => {
    event.preventDefault();
    dropzone.classList.add("dragging");
  }),
);
["dragleave", "drop"].forEach((type) =>
  dropzone.addEventListener(type, (event) => {
    event.preventDefault();
    dropzone.classList.remove("dragging");
  }),
);
dropzone.addEventListener("drop", (event) =>
  upload(event.dataTransfer.files[0]),
);
async function poll() {
  try {
    const data = await api(`/api/session/${session.id}`);
    if (data.status === "error")
      throw new Error(data.error || "Matting failed.");
    if (data.status === "complete") {
      running = false;
      resultVersion++;
      $("processing").hidden = true;
      $("result").hidden = false;
      $("foreground-view").disabled = false;
      $("alpha-view").disabled = false;
      $("result-badge").textContent = "Complete";
      $("output-meta").textContent =
        `${data.total} frames · ${session.fps.toFixed(1)} fps`;
      setView(view, true);
      notice(
        "Your matte is ready. Switch between the foreground and alpha views.",
      );
      updateControls();
      return;
    }
    const stages = {
      queued: "Waiting for the GPU",
      preparing: "Preparing your video",
      matting: "Matting every frame",
      encoding: "Preparing your previews",
    };
    $("stage").textContent = stages[data.status] || "Processing";
    $("frame-progress").textContent = ["matting", "encoding"].includes(
      data.status,
    )
      ? `${data.completed} / ${data.total} frames`
      : "Your full video will be processed.";
    $("progress-bar").style.width =
      `${data.status === "encoding" ? 80 + (20 * data.completed) / data.total : data.status === "matting" ? (80 * data.completed) / data.total : 0}%`;
    pollTimer = setTimeout(poll, options.poll_interval_ms);
  } catch (error) {
    running = false;
    $("processing").hidden = true;
    $("result-empty").hidden = false;
    $("result-badge").textContent = "Needs attention";
    notice(error.message, true);
    updateControls();
  }
}
$("start").onclick = async () => {
  if ($("start").disabled) return;
  running = true;
  updateControls();
  $("result").pause();
  $("result").hidden = true;
  $("result-empty").hidden = true;
  $("processing").hidden = false;
  $("download").hidden = true;
  $("foreground-view").disabled = true;
  $("alpha-view").disabled = true;
  $("result-badge").textContent = "Processing";
  $("stage").textContent = "Preparing your video";
  $("progress-bar").style.width = "0%";
  try {
    await api(`/api/session/${session.id}/start`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ resolution: $("resolution").value }),
    });
    notice("Matting started. You can follow the frame progress on the right.");
    poll();
  } catch (error) {
    running = false;
    $("processing").hidden = true;
    $("result-empty").hidden = false;
    notice(error.message, true);
    updateControls();
  }
};
function setView(next, fresh = false) {
  view = next;
  updateViewButtons();
  const player = $("result"),
    time = fresh ? 0 : player.currentTime,
    paused = fresh ? false : player.paused;
  const url = `/api/session/${session.id}/media/${view}?format=${playbackFormat}&v=${resultVersion}`;
  player.onloadedmetadata = () => {
    player.currentTime = Math.min(time, player.duration || 0);
    if (!paused) player.play().catch(() => {});
  };
  player.src = url;
  player.load();
  $("download").href =
    `/api/session/${session.id}/media/${view}?format=mp4&v=${resultVersion}`;
  $("download").download = `mam2_${view}.mp4`;
  $("download").hidden = false;
}
$("foreground-view").onclick = () => {
  if (view !== "foreground") setView("foreground");
};
$("alpha-view").onclick = () => {
  if (view !== "alpha") setView("alpha");
};
new ResizeObserver(draw).observe(canvas);
updateControls();

api("/api/options")
  .then((data) => {
    options = data;
    $("resolution").replaceChildren(
      ...Object.keys(data.resolution_presets).map((label) => {
        const option = document.createElement("option");
        option.value = option.textContent = label;
        return option;
      }),
    );
    $("resolution").value = data.default_resolution;
    updateControls();
  })
  .catch((error) => notice(error.message, true));

api("/api/examples")
  .then((data) => renderExamples(data.examples))
  .catch((error) => {
    if (error.status !== 404)
      notice(
        "Could not load the examples. You can still upload a video.",
        true,
      );
  });
