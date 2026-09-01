(() => {
  "use strict";

  const API = "/h3studio/api";
  const STORAGE_KEY = "forge-h3-studio-project-v2";
  const LEGACY_STORAGE_KEY = "forge-h3-studio-project-v1";
  const MODE_META = {
    t2v: { label: "文生视频", short: "T2V", icon: "✦", hint: "仅使用提示词生成带音频视频" },
    i2v: { label: "首帧图生", short: "I2V", icon: "◫", hint: "以一张首帧图片作为构图与角色起点" },
    fl2v: { label: "首尾帧", short: "FL2V", icon: "⇥", hint: "分别指定首帧与尾帧，控制镜头始末" },
    ref: { label: "多模态参考", short: "REF", icon: "⌘", hint: "最多 9 图、3 视频、3 音频参考" },
  };
  const KIND_META = {
    image: { label: "图片", icon: "▧", accept: ["png", "jpg", "jpeg", "webp", "gif", "bmp"] },
    video: { label: "视频", icon: "▶", accept: ["mp4", "mov", "webm", "mkv", "avi", "m4v"] },
    audio: { label: "音频", icon: "♫", accept: ["mp3", "wav", "flac", "m4a", "aac", "ogg"] },
  };
  const ASPECT_RATIOS = {
    "1:1": [1, 1],
    "2:3": [2, 3],
    "3:2": [3, 2],
    "3:4": [3, 4],
    "4:3": [4, 3],
    "9:16": [9, 16],
    "16:9": [16, 9],
    "21:9": [21, 9],
  };
  const MEGAPIXELS = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.98, 1.0, 1.2, 1.5, 1.8, 2.0];
  const H3_16_9_32 = {
    "0.2": [608, 352], "0.3": [736, 416], "0.4": [864, 480], "0.5": [960, 544],
    "0.6": [1056, 608], "0.7": [1152, 640], "0.8": [1216, 672], "0.9": [1280, 736],
    "0.98": [1344, 768], "1": [1376, 768], "1.2": [1504, 832], "1.5": [1664, 928],
    "1.8": [1824, 1024], "2": [1920, 1088],
  };

  const state = {
    mounted: false,
    entered: false,
    boot: null,
    config: {},
    backend: { state: "stopped", ready: false },
    catalog: { models: [], text_encoders: [], vaes: [], loras: [], samplers: [], schedulers: [] },
    mode: "t2v",
    assets: [],
    selectedAssetId: null,
    assetFilter: "all",
    assetSource: "all",
    assetSearch: "",
    firstFrame: null,
    lastFrame: null,
    frameCrops: { first: null, last: null },
    references: { image: [], video: [], audio: [] },
    refVideoAudio: [],
    loras: [],
    loraPresets: [],
    jobs: [],
    activeJobId: null,
    result: null,
    params: {
      model: "",
      text_encoder: "",
      video_vae: "",
      audio_vae: "",
      auto_model: true,
      weight_dtype: "default",
      clip_device: "default",
      prompt: "",
      width: 1344,
      height: 768,
      aspect_ratio: "16:9",
      megapixels: 0.98,
      rounding_multiple: 32,
      resolution_linked: true,
      frames: 124,
      steps: 30,
      seed: 0,
      random_seed: true,
      sampler: "euler",
      scheduler: "simple",
      shift_video: 12,
      shift_audio: 3,
      denoise: 1,
      ref_image_size: "match",
      output_fps: 24,
      output_format: "auto",
      output_codec: "auto",
      output_crf: 23,
      bit_depth: 8,
      filename_prefix: "video/Forge_H3_Studio",
    },
    sidebarTab: "assets",
    inspectorTab: "project",
    taskFilter: "all",
    taskSort: "newest",
    uploadBusy: false,
    backendStarting: false,
    catalogLoading: false,
    trackers: new Map(),
    cropEditor: null,
    thumbnailLoads: new Set(),
    taskRenderQueued: false,
    saveTimer: null,
  };

  const $ = (selector, root = document) => root.querySelector(selector);
  const $$ = (selector, root = document) => Array.from(root.querySelectorAll(selector));
  const root = () => document.getElementById("forge-h3-studio-root");
  const esc = (value) => String(value ?? "").replace(/[&<>'"]/g, (char) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;" })[char]);
  const clamp = (number, min, max) => Math.min(max, Math.max(min, number));
  const uid = (prefix = "id") => `${prefix}_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 7)}`;

  async function request(path, options = {}) {
    const init = { ...options, headers: { ...(options.headers || {}) } };
    if (init.body && !(init.body instanceof FormData) && typeof init.body !== "string") {
      init.headers["Content-Type"] = "application/json";
      init.body = JSON.stringify(init.body);
    }
    const response = await fetch(API + path, init);
    let payload = null;
    try { payload = await response.json(); } catch (_) { payload = null; }
    if (!response.ok) {
      const detail = payload?.detail || payload?.error || `${response.status} ${response.statusText}`;
      throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
    }
    return payload;
  }

  function toast(message, type = "info", timeout = 4200) {
    const host = $(".h3s-toasts", root());
    if (!host) return;
    const item = document.createElement("div");
    item.className = `h3s-toast h3s-toast-${type}`;
    item.innerHTML = `<span class="h3s-toast-dot"></span><span>${esc(message)}</span>`;
    host.appendChild(item);
    requestAnimationFrame(() => item.classList.add("show"));
    setTimeout(() => {
      item.classList.remove("show");
      setTimeout(() => item.remove(), 240);
    }, timeout);
  }

  function debounceSave() {
    clearTimeout(state.saveTimer);
    state.saveTimer = setTimeout(saveProject, 350);
  }

  function projectPayload() {
    return {
      version: 2,
      mode: state.mode,
      assets: state.assets.map((asset) => ({
        id: asset.id, name: asset.name, kind: asset.kind, file: asset.file, url: asset.url,
        size: asset.size, source: asset.source || "imported", generatedKey: asset.generatedKey,
        jobId: asset.jobId, createdAt: asset.createdAt, summary: asset.summary,
        width: asset.width, height: asset.height, duration: asset.duration,
      })),
      firstFrame: state.firstFrame,
      lastFrame: state.lastFrame,
      frameCrops: state.frameCrops,
      references: state.references,
      refVideoAudio: state.refVideoAudio,
      loras: state.loras,
      params: state.params,
    };
  }

  function saveProject() {
    try { localStorage.setItem(STORAGE_KEY, JSON.stringify(projectPayload())); } catch (_) { /* quota/private mode */ }
  }

  function restoreProject() {
    try {
      const saved = JSON.parse(localStorage.getItem(STORAGE_KEY) || localStorage.getItem(LEGACY_STORAGE_KEY) || "null");
      if (!saved || ![1, 2].includes(saved.version)) return;
      if (MODE_META[saved.mode]) state.mode = saved.mode;
      if (Array.isArray(saved.assets)) state.assets = saved.assets.map((asset) => ({ source: "imported", ...asset }));
      state.firstFrame = saved.firstFrame || null;
      state.lastFrame = saved.lastFrame || null;
      if (saved.frameCrops && typeof saved.frameCrops === "object") {
        state.frameCrops.first = saved.frameCrops.first || null;
        state.frameCrops.last = saved.frameCrops.last || null;
      }
      if (saved.references && typeof saved.references === "object") {
        for (const kind of Object.keys(KIND_META)) {
          if (Array.isArray(saved.references[kind])) state.references[kind] = saved.references[kind];
        }
      }
      if (Array.isArray(saved.refVideoAudio)) state.refVideoAudio = saved.refVideoAudio.map((value) => value !== false);
      while (state.refVideoAudio.length < state.references.video.length) state.refVideoAudio.push(true);
      if (Array.isArray(saved.loras)) state.loras = saved.loras;
      if (saved.params && typeof saved.params === "object") Object.assign(state.params, saved.params);
      if (!ASPECT_RATIOS[state.params.aspect_ratio]) state.params.aspect_ratio = "16:9";
      state.params.rounding_multiple = clamp(Math.round(Number(state.params.rounding_multiple) || 32), 1, 512);
    } catch (_) { /* ignore malformed local state */ }
  }

  function kindFromName(name, mime = "") {
    if (mime.startsWith("image/")) return "image";
    if (mime.startsWith("video/")) return "video";
    if (mime.startsWith("audio/")) return "audio";
    const ext = String(name).split(".").pop().toLowerCase();
    return Object.keys(KIND_META).find((kind) => KIND_META[kind].accept.includes(ext)) || null;
  }

  function formatBytes(value) {
    const bytes = Number(value || 0);
    if (!bytes) return "";
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 ** 2) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / 1024 ** 2).toFixed(1)} MB`;
  }

  function alignFrames(value) {
    let frames = Math.max(5, Math.round(Number(value) || 5));
    while (frames % 17 !== 5) frames += 1;
    return frames;
  }

  function alignDimension(value, multiple = state.params.rounding_multiple) {
    const step = clamp(Math.round(Number(multiple) || 32), 1, 512);
    const maximum = Math.floor(4096 / step) * step;
    return Math.min(maximum, Math.max(step, Math.round(Number(value || step) / step) * step));
  }

  function calculatedResolution(aspect = state.params.aspect_ratio, megapixels = state.params.megapixels, multiple = state.params.rounding_multiple) {
    const step = clamp(Math.round(Number(multiple) || 32), 1, 512);
    const exact = aspect === "16:9" && step === 32 ? H3_16_9_32[String(Number(megapixels))] : null;
    if (exact) return exact.slice();
    const [ratioWidth, ratioHeight] = ASPECT_RATIOS[aspect] || [state.params.width, state.params.height];
    const area = Math.max(0.01, Number(megapixels) || 0.98) * 1_000_000;
    const ratio = ratioWidth / ratioHeight;
    const width = Math.sqrt(area * ratio);
    const height = width / ratio;
    return [alignDimension(width, step), alignDimension(height, step)];
  }

  function applyResolutionSelection() {
    const [width, height] = calculatedResolution();
    state.params.width = width;
    state.params.height = height;
    state.params.resolution_linked = true;
    reframeAssignedCrops();
  }

  function formatClock(timestamp) {
    if (!timestamp) return "—";
    return new Date(Number(timestamp) * 1000).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
  }

  function formatElapsed(seconds) {
    const value = Math.max(0, Math.round(Number(seconds) || 0));
    if (value < 60) return `${value}s`;
    const minutes = Math.floor(value / 60);
    const rest = value % 60;
    if (minutes < 60) return `${minutes}m ${rest}s`;
    return `${Math.floor(minutes / 60)}h ${minutes % 60}m`;
  }

  function jobElapsed(job) {
    const start = job.started_at || job.live?.startedAt || job.created_at;
    const end = job.completed_at || Date.now() / 1000;
    return Math.max(0, end - start);
  }

  function jobProgress(job) {
    if (job.state === "completed") return { overall: 100, node: 100 };
    const live = job.live || {};
    const count = Math.max(1, Number(job.summary?.node_count || 1));
    const completed = Array.isArray(live.completedNodes) ? live.completedNodes.length : 0;
    const node = clamp(Number(live.nodePercent || 0), 0, 100);
    return { overall: clamp(Number(live.overallPercent ?? ((completed + node / 100) / count * 100)), 0, 99.9), node };
  }

  function jobEta(job) {
    const progress = jobProgress(job).overall;
    if (progress < 1 || job.state !== "running") return null;
    return Math.max(0, jobElapsed(job) * (100 - progress) / progress);
  }

  function outputKind(output) {
    return kindFromName(output?.filename || output?.name || "", output?.content_type || "");
  }

  function preferredOutput(job) {
    return (job?.outputs || []).find((item) => outputKind(item) === "video")
      || (job?.outputs || []).find((item) => outputKind(item) === "image")
      || (job?.outputs || []).find((item) => outputKind(item) === "audio")
      || job?.outputs?.[0] || null;
  }

  function captureVideoThumbnail(url) {
    return new Promise((resolve, reject) => {
      const video = document.createElement("video");
      const timeout = setTimeout(() => finish(new Error("视频缩略图读取超时")), 15000);
      let settled = false;
      const finish = (error, value = null) => {
        if (settled) return;
        settled = true; clearTimeout(timeout);
        video.removeAttribute("src"); video.load();
        if (error) reject(error); else resolve(value);
      };
      const draw = () => {
        try {
          const width = Math.min(480, video.videoWidth || 480);
          const height = Math.max(1, Math.round(width * (video.videoHeight || 270) / (video.videoWidth || 480)));
          const canvas = document.createElement("canvas"); canvas.width = width; canvas.height = height;
          canvas.getContext("2d").drawImage(video, 0, 0, width, height);
          finish(null, canvas.toDataURL("image/jpeg", 0.82));
        } catch (error) { finish(error); }
      };
      video.muted = true; video.playsInline = true; video.preload = "auto";
      video.onerror = () => finish(new Error("浏览器无法解码输出视频"));
      video.onloadedmetadata = () => {
        const target = Number.isFinite(video.duration) && video.duration > 0 ? Math.min(0.12, video.duration / 3) : 0;
        if (target > 0) { video.onseeked = draw; video.currentTime = target; }
        else video.onloadeddata = draw;
      };
      video.src = url;
    });
  }

  function ensureJobThumbnail(job) {
    if (!job) return;
    const output = preferredOutput(job);
    const live = job.live || (job.live = {});
    if (job.state !== "completed" || !output?.url || outputKind(output) !== "video" || live.previewUrl || live.outputThumbnailUrl || state.thumbnailLoads.has(job.id)) return;
    state.thumbnailLoads.add(job.id);
    captureVideoThumbnail(output.url).then((thumbnail) => {
      live.outputThumbnailUrl = thumbnail;
      state.assets.filter((asset) => asset.jobId === job.id).forEach((asset) => { asset.thumbnailUrl = thumbnail; });
      scheduleTaskRender();
      if (state.sidebarTab === "assets") renderSidebar();
    }).catch(() => { /* The video element remains as the final fallback. */ })
      .finally(() => state.thumbnailLoads.delete(job.id));
  }

  function syncGeneratedAssets(jobs, persist = true) {
    let changed = false;
    for (const job of Array.isArray(jobs) ? jobs : [jobs]) {
      if (!job || job.state !== "completed") continue;
      for (const output of job.outputs || []) {
        const kind = outputKind(output);
        if (!kind || !output.url) continue;
        if (output.type === "temp" && (job.outputs || []).some((item) => outputKind(item) === "video")) continue;
        const key = [job.prompt_id || job.id, output.type || "output", output.subfolder || "", output.filename || output.name].join("|");
        const existing = state.assets.find((asset) => asset.generatedKey === key);
        const data = {
          name: output.filename || output.name || "H3 output",
          kind,
          file: [output.subfolder, output.filename || output.name].filter(Boolean).join("/"),
          url: output.url,
          source: "generated",
          generatedKey: key,
          jobId: job.id,
          createdAt: job.completed_at || job.updated_at || Date.now() / 1000,
          summary: job.summary || {},
          thumbnailUrl: job.live?.previewUrl || job.live?.outputThumbnailUrl || "",
        };
        if (existing) Object.assign(existing, data);
        else {
          state.assets.unshift({ id: uid("generated"), ...data });
          changed = true;
        }
      }
      ensureJobThumbnail(job);
    }
    if (changed && persist) saveProject();
    return changed;
  }

  function imageSizeFromFile(file) {
    return new Promise((resolve) => {
      const url = URL.createObjectURL(file);
      const image = new Image();
      image.onload = () => { resolve({ width: image.naturalWidth, height: image.naturalHeight }); URL.revokeObjectURL(url); };
      image.onerror = () => { resolve({ width: 0, height: 0 }); URL.revokeObjectURL(url); };
      image.src = url;
    });
  }

  function ensureImageSize(asset) {
    if (!asset || asset.kind !== "image") return Promise.resolve(null);
    if (asset.width && asset.height) return Promise.resolve({ width: asset.width, height: asset.height });
    return new Promise((resolve, reject) => {
      const image = new Image();
      image.onload = () => {
        asset.width = image.naturalWidth;
        asset.height = image.naturalHeight;
        debounceSave();
        resolve({ width: asset.width, height: asset.height });
      };
      image.onerror = () => reject(new Error("无法读取图片尺寸"));
      image.src = asset.url;
    });
  }

  function makeCrop(asset, previous = null) {
    if (!asset?.width || !asset?.height) return null;
    const ratio = state.params.width / state.params.height;
    const sourceRatio = asset.width / asset.height;
    let maxWidth;
    let maxHeight;
    if (sourceRatio > ratio) {
      maxHeight = asset.height;
      maxWidth = maxHeight * ratio;
    } else {
      maxWidth = asset.width;
      maxHeight = maxWidth / ratio;
    }
    const zoom = clamp(Number(previous?.zoom || 1), 1, 6);
    const width = Math.max(8, Math.round(maxWidth / zoom));
    const height = Math.max(8, Math.round(maxHeight / zoom));
    const oldSourceWidth = Number(previous?.source_width || asset.width);
    const oldSourceHeight = Number(previous?.source_height || asset.height);
    const centreXRatio = previous ? (Number(previous.x) + Number(previous.width) / 2) / oldSourceWidth : 0.5;
    const centreYRatio = previous ? (Number(previous.y) + Number(previous.height) / 2) / oldSourceHeight : 0.5;
    const x = clamp(Math.round(centreXRatio * asset.width - width / 2), 0, asset.width - width);
    const y = clamp(Math.round(centreYRatio * asset.height - height / 2), 0, asset.height - height);
    return { x, y, width, height, source_width: asset.width, source_height: asset.height, zoom };
  }

  function reframeAssignedCrops() {
    for (const slot of ["first", "last"]) {
      const asset = assetById(slot === "first" ? state.firstFrame : state.lastFrame);
      if (asset?.width && asset?.height && state.frameCrops[slot]) state.frameCrops[slot] = makeCrop(asset, state.frameCrops[slot]);
    }
  }

  function selectedAsset() {
    return state.assets.find((asset) => asset.id === state.selectedAssetId) || null;
  }

  function assetById(id) {
    return state.assets.find((asset) => asset.id === id) || null;
  }

  function modelOptions(values, selected, placeholder) {
    const options = [`<option value="">${esc(placeholder)}</option>`];
    for (const value of values || []) options.push(`<option value="${esc(value)}" ${value === selected ? "selected" : ""}>${esc(value)}</option>`);
    return options.join("");
  }

  function icon(name) {
    const icons = {
      play: '<svg viewBox="0 0 24 24"><path d="M8 5v14l11-7z"/></svg>',
      stop: '<svg viewBox="0 0 24 24"><rect x="6" y="6" width="12" height="12" rx="2"/></svg>',
      refresh: '<svg viewBox="0 0 24 24"><path d="M20 6v5h-5M4 18v-5h5"/><path d="M6.1 9a7 7 0 0 1 11.4-2.3L20 9M4 15l2.5 2.3A7 7 0 0 0 17.9 15"/></svg>',
      settings: '<svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.7 1.7 0 0 0 .3 1.9l.1.1-2.8 2.8-.1-.1a1.7 1.7 0 0 0-1.9-.3 1.7 1.7 0 0 0-1 1.6v.2h-4V21a1.7 1.7 0 0 0-1-1.6 1.7 1.7 0 0 0-1.9.3l-.1.1L4.2 17l.1-.1a1.7 1.7 0 0 0 .3-1.9A1.7 1.7 0 0 0 3 14H2.8v-4H3a1.7 1.7 0 0 0 1.6-1 1.7 1.7 0 0 0-.3-1.9L4.2 7 7 4.2l.1.1a1.7 1.7 0 0 0 1.9.3A1.7 1.7 0 0 0 10 3V2.8h4V3a1.7 1.7 0 0 0 1 1.6 1.7 1.7 0 0 0 1.9-.3l.1-.1L19.8 7l-.1.1a1.7 1.7 0 0 0-.3 1.9 1.7 1.7 0 0 0 1.6 1h.2v4H21a1.7 1.7 0 0 0-1.6 1z"/></svg>',
      upload: '<svg viewBox="0 0 24 24"><path d="M12 16V4M7 9l5-5 5 5M5 20h14"/></svg>',
      dice: '<svg viewBox="0 0 24 24"><rect x="4" y="4" width="16" height="16" rx="3"/><circle cx="9" cy="9" r="1"/><circle cx="15" cy="15" r="1"/><circle cx="15" cy="9" r="1"/><circle cx="9" cy="15" r="1"/></svg>',
      close: '<svg viewBox="0 0 24 24"><path d="M6 6l12 12M18 6L6 18"/></svg>',
      plus: '<svg viewBox="0 0 24 24"><path d="M12 5v14M5 12h14"/></svg>',
      download: '<svg viewBox="0 0 24 24"><path d="M12 4v11M8 11l4 4 4-4M5 20h14"/></svg>',
    };
    return icons[name] || "";
  }

  function shellHtml() {
    return `
      <div class="h3s-shell">
        <header class="h3s-topbar">
          <div class="h3s-brand">
            <div class="h3s-brand-mark"><span>H3</span></div>
            <div><strong>MiniMax H3 Studio</strong><small>Forge Neo 视频工作台</small></div>
          </div>
          <div class="h3s-top-mode" data-role="mode-summary"><span>${MODE_META[state.mode].short}</span>${MODE_META[state.mode].label}</div>
          <div class="h3s-top-spacer"></div>
          <button class="h3s-backend-pill" data-action="open-task-panel" title="查看运行状态与任务队列">
            <i class="h3s-status-dot"></i><span data-role="backend-text">未连接</span><b data-role="queue-count">0</b>
          </button>
          <button class="h3s-icon-btn" data-action="refresh-catalog" title="刷新模型列表">${icon("refresh")}</button>
          <button class="h3s-icon-btn" data-action="open-settings" title="连接与启动设置">${icon("settings")}</button>
          <button class="h3s-primary-btn" data-action="generate">${icon("play")}<span>生成视频</span><kbd>G</kbd></button>
          <button class="h3s-icon-btn h3s-danger-btn" data-action="cancel-job" title="中断任务" disabled>${icon("stop")}</button>
        </header>
        <div class="h3s-body">
          <aside class="h3s-sidebar">
            <nav class="h3s-side-tabs">
              <button data-side-tab="assets" class="active">素材</button>
              <button data-side-tab="loras">LoRA <span class="h3s-tab-count" data-role="lora-count">0</span></button>
              <button data-side-tab="history">任务</button>
            </nav>
            <div class="h3s-side-content" data-role="sidebar-content"></div>
          </aside>
          <main class="h3s-workspace">
            <nav class="h3s-mode-tabs">
              ${Object.entries(MODE_META).map(([key, mode]) => `<button data-mode="${key}" class="${state.mode === key ? "active" : ""}"><i>${mode.icon}</i><span>${mode.label}</span><small>${mode.short}</small></button>`).join("")}
            </nav>
            <section class="h3s-canvas-wrap">
              <div class="h3s-canvas-head">
                <div><strong data-role="canvas-title">${MODE_META[state.mode].label}</strong><span data-role="canvas-hint">${MODE_META[state.mode].hint}</span></div>
                <div class="h3s-canvas-actions">
                  <button data-action="export-project">${icon("download")}项目</button>
                  <button data-action="preview-workflow">{ } 工作流</button>
                </div>
              </div>
              <div class="h3s-stage" data-role="stage"></div>
            </section>
            <section class="h3s-prompt-panel">
              <div class="h3s-prompt-head">
                <label>镜头提示词</label>
                <div data-role="anchor-tools"></div>
                <span data-role="prompt-count">0</span>
              </div>
              <textarea data-param="prompt" placeholder="描述人物、动作、镜头运动、环境和希望生成的声音……"></textarea>
            </section>
            <section class="h3s-timeline">
              <div class="h3s-timeline-toolbar">
                <strong>镜头时间线</strong><span>当前单镜头 · 后续项目可扩展为多段自动承接</span>
                <div class="h3s-top-spacer"></div><b data-role="duration-label">124f · 5.17s</b>
              </div>
              <div class="h3s-track"><div class="h3s-track-label"><i>V1</i><span>H3 镜头</span></div><div class="h3s-track-content" data-role="timeline-track"></div></div>
              <div class="h3s-track h3s-audio-track"><div class="h3s-track-label"><i>A1</i><span>AI 音频</span></div><div class="h3s-waveform">${Array.from({length: 80}, (_, i) => `<i style="height:${18 + ((i * 17) % 62)}%"></i>`).join("")}</div></div>
            </section>
          </main>
          <aside class="h3s-inspector">
            <nav class="h3s-inspector-tabs"><button data-inspector-tab="project" class="active">项目参数</button><button data-inspector-tab="asset">素材属性</button></nav>
            <div class="h3s-inspector-scroll" data-role="inspector-content"></div>
          </aside>
        </div>
        <div class="h3s-start-overlay" data-role="start-overlay" hidden>
          <div class="h3s-start-card"><div class="h3s-loader-orbit"><i></i><i></i><span>H3</span></div><strong data-role="start-title">正在启动 H3 后端</strong><p data-role="start-detail">检查本地 ComfyUI…</p><div class="h3s-progress-line"><i></i></div><button data-action="open-settings">打开连接设置</button></div>
        </div>
        <div class="h3s-modal-layer" data-role="modal-layer"></div>
        <div class="h3s-toasts"></div>
        <input type="file" data-role="asset-file-input" multiple hidden accept="image/*,video/*,audio/*">
        <input type="file" data-role="project-file-input" hidden accept="application/json,.json">
      </div>`;
  }

  function mount() {
    const target = root();
    if (!target || state.mounted) return;
    state.mounted = true;
    restoreProject();
    target.className = "h3s-root";
    target.innerHTML = shellHtml();
    bindShell();
    renderAll();
    bootstrap();
    watchVisibility();
  }

  async function bootstrap() {
    try {
      state.boot = await request("/bootstrap");
      state.config = state.boot.config || {};
      state.backend = state.boot.backend || state.backend;
      state.loraPresets = state.boot.lora_presets || [];
      state.jobs = (state.boot.jobs || []).map((job) => ({
        ...job,
        live: {
          ...(job.progress || {}),
          ...(job.preview_url ? { previewUrl: job.preview_url } : {}),
        },
      }));
      syncGeneratedAssets(state.jobs);
      state.jobs.filter((job) => ["queued", "running"].includes(job.state)).forEach(startJobTracking);
      if (!state.params.filename_prefix && state.config.output_prefix) state.params.filename_prefix = state.config.output_prefix;
      updateBackendUi();
      renderSidebar();
      if (state.backend.ready) await loadCatalog();
      if (isVisible()) onEnter();
    } catch (error) {
      toast(`工作台初始化失败：${error.message}`, "error", 8000);
      updateBackendUi();
    }
  }

  function watchVisibility() {
    const target = root();
    const observer = new IntersectionObserver((entries) => {
      if (entries.some((entry) => entry.isIntersecting && entry.intersectionRatio > 0)) onEnter();
    }, { threshold: 0.01 });
    observer.observe(target);
    const mutation = new MutationObserver(() => { if (isVisible()) onEnter(); });
    let ancestor = target.parentElement;
    for (let i = 0; ancestor && i < 5; i += 1, ancestor = ancestor.parentElement) mutation.observe(ancestor, { attributes: true, attributeFilter: ["class", "style", "hidden"] });
  }

  function isVisible() {
    const target = root();
    return !!target && target.offsetWidth > 0 && target.offsetHeight > 0 && getComputedStyle(target).visibility !== "hidden";
  }

  let configRefreshing = false;
  async function refreshConfig() {
    if (configRefreshing) return;
    configRefreshing = true;
    try {
      const fresh = await request("/settings");
      if (fresh && typeof fresh === "object") {
        state.config = fresh;
        if ($(".h3s-settings-modal", root())) openSettings();
      }
    } catch (e) { }
    finally { configRefreshing = false; }
  }

  function onEnter() {
    if (state.boot) refreshConfig();
    if (!state.boot || state.entered) return;
    state.entered = true;
    if (state.backend.ready) loadCatalog();
    else if (state.config.auto_start_on_tab !== false) ensureBackend();
  }

  async function ensureBackend(force = false) {
    if (state.backendStarting || state.backend.ready) return;
    if (!force && state.config.auto_start_on_tab === false) return;
    // 云端 API 模式不需要启动本地 ComfyUI，仅刷新后端状态
    if (state.config.backend_mode === "api") {
      try { state.backend = await request("/backend/status"); updateBackendUi(); } catch (_) { }
      return;
    }
    state.backendStarting = true;
    showStartOverlay(true, "正在启动 H3 后端", state.config.backend_mode === "external" ? "连接外部 ComfyUI…" : "启动本地托管 ComfyUI…");
    try {
      await request("/backend/start", { method: "POST" });
      const timeout = Number(state.config.startup_timeout || 180) * 1000;
      const begun = Date.now();
      while (Date.now() - begun < timeout) {
        await new Promise((resolve) => setTimeout(resolve, 1200));
        state.backend = await request("/backend/status");
        updateBackendUi();
        if (state.backend.ready) {
          showStartOverlay(false);
          toast("H3 后端已连接", "success");
          await loadCatalog(true);
          return;
        }
        if (state.backend.state === "error") throw new Error(state.backend.health?.error || `后端退出，代码 ${state.backend.exit_code}`);
        showStartOverlay(true, "正在载入 H3 后端", state.backend.process_running ? "等待 ComfyUI 加载节点和模型目录…" : "等待后端响应…");
      }
      throw new Error("启动超时，请检查后端日志");
    } catch (error) {
      showStartOverlay(true, "后端未能启动", error.message, true);
      toast(error.message, "error", 8000);
    } finally {
      state.backendStarting = false;
    }
  }

  function showStartOverlay(show, title = "", detail = "", failed = false) {
    const overlay = $("[data-role='start-overlay']", root());
    if (!overlay) return;
    overlay.hidden = !show;
    overlay.classList.toggle("failed", failed);
    if (title) $("[data-role='start-title']", overlay).textContent = title;
    if (detail) $("[data-role='start-detail']", overlay).textContent = detail;
  }

  async function loadCatalog(force = false) {
    if (state.catalogLoading || (!state.backend.ready && !force)) return;
    state.catalogLoading = true;
    const refresh = $("[data-action='refresh-catalog']", root());
    refresh?.classList.add("spinning");
    try {
      state.catalog = await request("/catalog");
      if (!state.catalog.h3_ready) toast(`H3 节点不完整：${state.catalog.missing_nodes.join(", ")}`, "warning", 9000);
      autoSelectModels();
      renderInspector();
      renderSidebar();
    } catch (error) {
      toast(`读取模型列表失败：${error.message}`, "error", 7000);
    } finally {
      state.catalogLoading = false;
      refresh?.classList.remove("spinning");
    }
  }

  function findPreferred(list, patterns, reject = []) {
    const lowered = patterns.map((value) => value.toLowerCase());
    const rejects = reject.map((value) => value.toLowerCase());
    return (list || []).find((item) => {
      const name = item.toLowerCase();
      return lowered.every((pattern) => name.includes(pattern)) && rejects.every((pattern) => !name.includes(pattern));
    }) || "";
  }

  function autoSelectModels() {
    const models = state.catalog.models || [];
    if (state.params.auto_model || !models.includes(state.params.model)) {
      const target = state.mode === "ref"
        ? findPreferred(models, ["minimax", "h3", "ref2va"]) || findPreferred(models, ["h3", "ref"])
        : findPreferred(models, ["minimax", "h3", "fl2va"]) || findPreferred(models, ["h3"], ["ref"]);
      if (target) state.params.model = target;
    }
    const clips = state.catalog.text_encoders || [];
    if (!clips.includes(state.params.text_encoder)) state.params.text_encoder = findPreferred(clips, ["minimax", "h3"]) || findPreferred(clips, ["minimax"]) || clips[0] || "";
    const vaes = state.catalog.vaes || [];
    if (!vaes.includes(state.params.video_vae)) state.params.video_vae = findPreferred(vaes, ["minimax", "h3", "video"]) || findPreferred(vaes, ["h3", "video"]) || "";
    if (!vaes.includes(state.params.audio_vae)) state.params.audio_vae = findPreferred(vaes, ["minimax", "h3", "audio"]) || findPreferred(vaes, ["h3", "audio"]) || "";
    const samplers = state.catalog.samplers || [];
    if (samplers.length && !samplers.includes(state.params.sampler)) state.params.sampler = samplers.includes("euler") ? "euler" : samplers[0];
    const schedulers = state.catalog.schedulers || [];
    if (schedulers.length && !schedulers.includes(state.params.scheduler)) state.params.scheduler = schedulers.includes("simple") ? "simple" : schedulers[0];
    debounceSave();
  }

  function updateBackendUi() {
    const pill = $(".h3s-backend-pill", root());
    if (!pill) return;
    pill.dataset.state = state.backend.ready ? "ready" : state.backend.state || "stopped";
    const labels = { ready: "H3 已连接", starting: "正在启动", stopped: "后端未启动", error: "后端错误" };
    const running = state.jobs.find((job) => job.state === "running");
    const queued = state.jobs.find((job) => job.state === "queued");
    $("[data-role='backend-text']", pill).textContent = running ? `H3 · ${jobProgress(running).overall.toFixed(0)}%` : queued ? "H3 · 排队中" : labels[state.backend.ready ? "ready" : state.backend.state] || "未连接";
    pill.title = running ? `${running.live?.nodeTitle || "正在运行"} · 点击查看任务` : "查看运行状态与任务队列";
    const active = state.jobs.filter((job) => ["queued", "running"].includes(job.state)).length;
    $("[data-role='queue-count']", pill).textContent = active;
    $("[data-role='queue-count']", pill).hidden = !active;
  }

  function renderAll() {
    renderSidebar();
    renderStage();
    renderPrompt();
    renderTimeline();
    renderInspector();
    updateBackendUi();
  }

  function renderSidebar() {
    const content = $("[data-role='sidebar-content']", root());
    if (!content) return;
    $$('[data-side-tab]', root()).forEach((button) => button.classList.toggle("active", button.dataset.sideTab === state.sidebarTab));
    const count = $("[data-role='lora-count']", root());
    if (count) count.textContent = state.loras.filter((item) => item.enabled).length;
    if (state.sidebarTab === "loras") content.innerHTML = loraSidebarHtml();
    else if (state.sidebarTab === "history") content.innerHTML = historySidebarHtml();
    else content.innerHTML = assetSidebarHtml();
  }

  function assetSidebarHtml() {
    const filtered = state.assets.filter((asset) => {
      const kindOk = state.assetFilter === "all" || asset.kind === state.assetFilter;
      const sourceOk = state.assetSource === "all" || (asset.source || "imported") === state.assetSource;
      const searchOk = !state.assetSearch || asset.name.toLowerCase().includes(state.assetSearch.toLowerCase());
      return kindOk && sourceOk && searchOk;
    });
    return `<div class="h3s-assets-head"><label class="h3s-search"><span>⌕</span><input data-role="asset-search" value="${esc(state.assetSearch)}" placeholder="搜索素材"></label><button class="h3s-upload-btn" data-action="choose-assets">${icon("upload")}导入</button></div>
      <div class="h3s-source-row"><button data-asset-source="all" class="${state.assetSource === "all" ? "active" : ""}">全部</button><button data-asset-source="generated" class="${state.assetSource === "generated" ? "active" : ""}">已生成 <b>${state.assets.filter((a) => a.source === "generated").length}</b></button><button data-asset-source="imported" class="${state.assetSource === "imported" ? "active" : ""}">已导入 <b>${state.assets.filter((a) => (a.source || "imported") === "imported").length}</b></button></div>
      <div class="h3s-filter-row">${["all", "image", "video", "audio"].map((kind) => `<button data-asset-filter="${kind}" class="${state.assetFilter === kind ? "active" : ""}">${kind === "all" ? "全部" : KIND_META[kind].label}<b>${kind === "all" ? state.assets.length : state.assets.filter((a) => a.kind === kind).length}</b></button>`).join("")}</div>
      <div class="h3s-dropzone ${state.uploadBusy ? "busy" : ""}" data-role="dropzone"><span>${state.uploadBusy ? "上传中…" : "+"}</span><div><strong>${state.uploadBusy ? "正在发送到 H3 后端" : "拖入图片、视频或音频"}</strong><small>素材保存在 ComfyUI input/forge_h3_studio</small></div></div>
      <div class="h3s-asset-grid">${filtered.length ? filtered.map(assetCardHtml).join("") : '<div class="h3s-empty"><i>◇</i><strong>暂无素材</strong><span>导入素材后可拖放到画布槽位</span></div>'}</div>`;
  }

  function assetCardHtml(asset) {
    let preview = `<div class="h3s-asset-icon">${KIND_META[asset.kind]?.icon || "◇"}</div>`;
    if (asset.kind === "image") preview = `<img src="${esc(asset.url)}" loading="lazy" alt="">`;
    else if (asset.kind === "video") preview = asset.thumbnailUrl
      ? `<img src="${esc(asset.thumbnailUrl)}" loading="lazy" alt="视频缩略图"><span class="h3s-play-badge">▶</span>`
      : `<video src="${esc(asset.url)}#t=0.1" muted playsinline preload="auto"></video><span class="h3s-play-badge">▶</span>`;
    else if (asset.kind === "audio") preview = `<div class="h3s-audio-thumb"><span>♫</span>${Array.from({length: 18}, (_, i) => `<i style="height:${22 + (i * 31) % 65}%"></i>`).join("")}</div>`;
    return `<article class="h3s-asset-card ${state.selectedAssetId === asset.id ? "selected" : ""}" draggable="${asset.source === "generated" ? "false" : "true"}" data-asset-id="${esc(asset.id)}"><div class="h3s-asset-preview">${preview}<div class="h3s-asset-card-actions"><button data-action="preview-asset" data-id="${esc(asset.id)}" title="大图浏览">⌕</button><button data-action="remove-asset" data-id="${esc(asset.id)}" title="从项目移除">×</button></div>${asset.source === "generated" ? '<em class="h3s-generated-badge">AI</em>' : ""}</div><div class="h3s-asset-name" title="${esc(asset.name)}">${esc(asset.name)}</div><small>${asset.source === "generated" ? "已生成" : KIND_META[asset.kind]?.label || asset.kind}${asset.summary?.resolution ? ` · ${esc(asset.summary.resolution)}` : asset.size ? ` · ${esc(formatBytes(asset.size))}` : ""}</small></article>`;
  }

  function loraSidebarHtml() {
    const rows = state.loras.map((lora, index) => `<div class="h3s-lora-row ${lora.enabled ? "" : "disabled"}" data-lora-index="${index}">
      <div class="h3s-lora-top"><button class="h3s-switch ${lora.enabled ? "on" : ""}" data-action="toggle-lora" data-index="${index}"><i></i></button><strong title="${esc(lora.name)}">${esc(lora.name)}</strong><div class="h3s-order"><button data-action="move-lora" data-index="${index}" data-dir="-1">↑</button><button data-action="move-lora" data-index="${index}" data-dir="1">↓</button></div><button class="h3s-row-delete" data-action="remove-lora" data-index="${index}">×</button></div>
      <div class="h3s-lora-weights"><label>模型<input type="number" min="-4" max="4" step="0.05" data-lora-field="model_strength" data-index="${index}" value="${Number(lora.model_strength ?? 1)}"></label><label>文本<input type="number" min="-4" max="4" step="0.05" data-lora-field="clip_strength" data-index="${index}" value="${Number(lora.clip_strength ?? 0)}" ${lora.apply_to_clip ? "" : "disabled"}></label><label class="h3s-check"><input type="checkbox" data-lora-field="apply_to_clip" data-index="${index}" ${lora.apply_to_clip ? "checked" : ""}><span>作用于编码器</span></label></div>
    </div>`).join("");
    return `<div class="h3s-lora-toolbar"><button class="h3s-accent-outline" data-action="open-lora-browser">${icon("plus")}添加 LoRA</button><button data-action="refresh-catalog">${icon("refresh")}</button></div>
      <div class="h3s-preset-bar"><select data-role="lora-preset"><option value="">LoRA 组合预设</option>${state.loraPresets.map((preset) => `<option value="${esc(preset.name)}">${esc(preset.name)}</option>`).join("")}</select><button data-action="save-lora-preset">保存</button><button data-action="delete-lora-preset">删除</button></div>
      <div class="h3s-lora-note"><i>i</i><span>多数 H3 LoRA 只修改扩散模型；确认 LoRA 包含文本编码器权重时再开启“作用于编码器”。加载顺序从上到下。</span></div>
      <div class="h3s-lora-list">${rows || '<div class="h3s-empty"><i>⌁</i><strong>未启用 LoRA</strong><span>可以添加 H3 Turbo、风格或角色 LoRA</span></div>'}</div>`;
  }

  function historySidebarHtml() {
    const active = state.jobs.filter((job) => ["queued", "running"].includes(job.state)).length;
    return `<div class="h3s-history-head"><div><strong>${active ? `${active} 个正在运行` : "生成任务"}</strong><small>${state.jobs.length} 个任务记录</small></div><button data-action="open-task-panel">展开</button><button data-action="refresh-jobs">${icon("refresh")}</button></div>${taskFiltersHtml()}<div class="h3s-job-list">${taskListHtml(true)}</div>`;
  }

  function taskFiltersHtml() {
    return `<div class="h3s-task-filters"><button data-task-filter="all" class="${state.taskFilter === "all" ? "active" : ""}">全部</button><button data-task-filter="completed" class="${state.taskFilter === "completed" ? "active" : ""}">已完成</button><span></span><button data-action="clear-queue" title="清除排队任务">清队列</button><button data-action="clear-completed" title="清除已完成记录">清完成</button></div>`;
  }

  function visibleJobs() {
    let jobs = state.jobs.filter((job) => state.taskFilter === "all" || job.state === "completed");
    jobs = jobs.slice().sort((a, b) => (state.taskSort === "oldest" ? 1 : -1) * (Number(a.created_at) - Number(b.created_at)));
    return jobs;
  }

  function jobPreviewHtml(job) {
    const preview = job.live?.previewUrl || job.live?.outputThumbnailUrl;
    const output = preferredOutput(job);
    if (preview) return `<img src="${esc(preview)}" alt="实时预览">`;
    if (outputKind(output) === "image") return `<img src="${esc(output.url)}" alt="输出预览">`;
    if (outputKind(output) === "video") return `<video src="${esc(output.url)}#t=0.1" muted playsinline preload="auto"></video><i>▶</i>`;
    return `<span>${job.state === "running" ? "⚡" : job.state === "queued" ? "◷" : "H3"}</span>`;
  }

  function jobCardHtml(job, compact = false) {
    const status = { queued: "排队", running: "生成中", completed: "已完成", failed: "失败", cancelled: "已中断" }[job.state] || job.state;
    const progress = jobProgress(job);
    const currentNode = job.live?.nodeTitle || (job.state === "queued" ? `队列第 ${job.queue_position || "?"} 位` : status);
    const eta = jobEta(job);
    return `<article class="h3s-job-card ${state.activeJobId === job.id ? "active" : ""}" data-state="${esc(job.state)}" data-action="select-job" data-id="${esc(job.id)}"><div class="h3s-job-thumb">${jobPreviewHtml(job)}</div><div class="h3s-job-main"><div class="h3s-job-title"><strong>${esc(job.summary?.mode_name || "H3 视频")}</strong><b>${job.state === "running" ? `${progress.overall.toFixed(0)}%` : status}</b></div><small>${esc(job.summary?.resolution || "")} · ${job.summary?.frames || "?"}f${eta ? ` · 约 ${formatElapsed(eta)}` : ""}</small><div class="h3s-job-progress overall"><i style="width:${progress.overall}%"></i></div>${!compact && ["queued", "running"].includes(job.state) ? `<div class="h3s-job-node"><span title="${esc(currentNode)}">${esc(currentNode)}</span><b>${progress.node.toFixed(0)}%</b></div><div class="h3s-job-progress node"><i style="width:${progress.node}%"></i></div>` : ""}</div><div class="h3s-job-actions">${["queued", "running"].includes(job.state) ? `<button data-action="cancel-specific-job" data-id="${esc(job.id)}" title="中断">×</button>` : ""}<button data-action="job-details" data-id="${esc(job.id)}" title="任务细节">•••</button></div></article>`;
  }

  function taskListHtml(compact = false) {
    const jobs = visibleJobs();
    return jobs.length ? jobs.map((job) => jobCardHtml(job, compact)).join("") : '<div class="h3s-empty"><i>◷</i><strong>暂无任务</strong><span>生成的视频和真实运行状态会显示在这里</span></div>';
  }

  function renderStage() {
    const stage = $("[data-role='stage']", root());
    if (!stage) return;
    $("[data-role='canvas-title']", root()).textContent = MODE_META[state.mode].label;
    $("[data-role='canvas-hint']", root()).textContent = MODE_META[state.mode].hint;
    $("[data-role='mode-summary']", root()).innerHTML = `<span>${MODE_META[state.mode].short}</span>${MODE_META[state.mode].label}`;
    $$('[data-mode]', root()).forEach((button) => button.classList.toggle("active", button.dataset.mode === state.mode));
    if (state.result) {
      stage.innerHTML = resultHtml(state.result);
      return;
    }
    if (state.mode === "t2v") {
      stage.innerHTML = `<div class="h3s-t2v-empty"><div class="h3s-nebula"><i></i><i></i><i></i><span>✦</span></div><strong>用文字构建一个有声音的镜头</strong><p>在下方描述场景、人物动作、镜头语言和环境声音。H3 将同步生成视频与音频。</p><div><span>${state.params.width} × ${state.params.height}</span><span>${alignFrames(state.params.frames)} 帧</span><span>24 FPS 基准</span></div></div>`;
    } else if (state.mode === "i2v" || state.mode === "fl2v") {
      stage.innerHTML = `<div class="h3s-keyframe-stage ${state.mode === "i2v" ? "single" : ""}">${slotHtml("first", "首帧", state.firstFrame)}${state.mode === "fl2v" ? '<div class="h3s-keyframe-arrow"><span>镜头演化</span><i>→</i></div>' + slotHtml("last", "尾帧", state.lastFrame) : ""}</div>`;
    } else {
      stage.innerHTML = refStageHtml();
    }
  }

  function slotHtml(key, label, assetId) {
    const asset = assetById(assetId);
    const crop = state.frameCrops[key];
    const position = crop ? `${((crop.x + crop.width / 2) / crop.source_width * 100).toFixed(2)}% ${((crop.y + crop.height / 2) / crop.source_height * 100).toFixed(2)}%` : "50% 50%";
    return `<div class="h3s-frame-slot ${asset ? "filled" : ""}" data-slot="${key}" style="--target-ratio:${state.params.width}/${state.params.height}">${asset ? `<img src="${esc(asset.url)}" alt="" style="object-position:${position}"><div class="h3s-slot-shade"></div><button class="h3s-slot-clear" data-action="clear-slot" data-slot="${key}">×</button><strong>${label}</strong><span>${esc(asset.name)}</span><button class="h3s-crop-button" data-action="open-crop" data-slot="${key}">⌗ ${crop ? "调整取景" : "框选生成范围"}</button>` : `<i>＋</i><strong>${label}</strong><span>拖入图片或点击选择</span><button data-action="fill-slot" data-slot="${key}">从素材库选择</button>`}</div>`;
  }

  function refStageHtml() {
    const groups = ["image", "video", "audio"].map((kind) => {
      const max = kind === "image" ? 9 : 3;
      const items = state.references[kind];
      const cells = Array.from({ length: max }, (_, index) => {
        const asset = assetById(items[index]);
        if (!asset) return `<div class="h3s-ref-cell" data-ref-kind="${kind}" data-ref-index="${index}"><b>${index + 1}</b><i>${KIND_META[kind].icon}</i><span>拖入</span></div>`;
        const preview = kind === "image" ? `<img src="${esc(asset.url)}">` : `<i>${KIND_META[kind].icon}</i>`;
        const audioToggle = kind === "video" ? `<button class="h3s-ref-audio-toggle ${state.refVideoAudio[index] !== false ? "active" : ""}" data-action="toggle-ref-audio" data-index="${index}" title="${state.refVideoAudio[index] !== false ? "使用参考视频音轨" : "忽略参考视频音轨"}">♫</button>` : "";
        return `<div class="h3s-ref-cell filled" data-ref-kind="${kind}" data-ref-index="${index}"><b>${index + 1}</b>${preview}<span title="${esc(asset.name)}">${esc(asset.name)}</span>${audioToggle}<button class="h3s-ref-remove" data-action="clear-ref" data-kind="${kind}" data-index="${index}">×</button></div>`;
      }).join("");
      return `<div class="h3s-ref-group"><header><i data-kind="${kind}"></i><strong>${KIND_META[kind].label}参考</strong><span>${items.filter(Boolean).length}/${max}</span></header><div class="h3s-ref-grid h3s-ref-${kind}">${cells}</div></div>`;
    }).join("");
    return `<div class="h3s-ref-stage">${groups}</div>`;
  }

  function resultHtml(result) {
    const outputs = result.outputs || [];
    const item = outputs.find((output) => /\.(mp4|webm|mkv|mov)$/i.test(output.filename || ""))
      || outputs.find((output) => /\.(png|jpe?g|webp|gif|wav|mp3|flac|m4a|ogg)$/i.test(output.filename || ""))
      || outputs[0];
    let media = '<div class="h3s-result-empty">任务已结束，但没有发现可预览输出</div>';
    if (item) {
      const ext = item.filename.split(".").pop().toLowerCase();
      if (["mp4", "webm", "mkv", "mov"].includes(ext)) media = `<video src="${esc(item.url)}" controls autoplay loop></video>`;
      else if (["png", "jpg", "jpeg", "webp", "gif"].includes(ext)) media = `<img src="${esc(item.url)}" alt="H3 output">`;
      else if (["wav", "mp3", "flac", "m4a", "ogg"].includes(ext)) media = `<audio src="${esc(item.url)}" controls autoplay></audio>`;
    }
    return `<div class="h3s-result-view">${media}<div class="h3s-result-overlay"><div><b>生成结果 · 已放入素材库</b><span>${esc(result.summary?.resolution || "")} · ${result.summary?.frames || ""}f · Seed ${esc(result.summary?.seed ?? "—")}</span></div><button data-action="close-result">返回编辑</button><button data-action="reuse-job" data-id="${esc(result.id)}">复用参数</button><a href="${API}/jobs/${encodeURIComponent(result.id)}/metadata" download>参数 JSON</a>${item ? `<button data-action="preview-output" data-job-id="${esc(result.id)}">全屏浏览</button><a href="${esc(item.url)}" download>${icon("download")}下载</a>` : ""}</div></div>`;
  }

  function renderPrompt() {
    const textarea = $("textarea[data-param='prompt']", root());
    if (textarea && textarea.value !== state.params.prompt) textarea.value = state.params.prompt;
    const count = $("[data-role='prompt-count']", root());
    if (count) count.textContent = `${state.params.prompt.length} 字`;
    const tools = $("[data-role='anchor-tools']", root());
    if (tools) {
      tools.innerHTML = state.mode === "ref" ? referenceAnchorToolsHtml() : '<span>H3 标准链路不使用负面提示词与 CFG</span>';
    }
  }

  function referenceAnchorToolsHtml() {
    const buttons = [];
    state.references.image.forEach((id, index) => {
      if (id) buttons.push(`<button data-action="insert-anchor" data-kind="image" data-index="${index}">Picture ${index + 1}</button>`);
    });
    let audioOrdinal = 0;
    state.references.video.forEach((id, index) => {
      if (!id) return;
      if (state.refVideoAudio[index] !== false) {
        buttons.push(`<button data-action="insert-anchor" data-kind="audio" data-index="${audioOrdinal}" title="参考视频 ${index + 1} 的音轨">Audio ${audioOrdinal + 1}·音轨</button>`);
        audioOrdinal += 1;
      }
      buttons.push(`<button data-action="insert-anchor" data-kind="video" data-index="${index}">Video ${index + 1}</button>`);
    });
    state.references.audio.forEach((id) => {
      if (!id) return;
      buttons.push(`<button data-action="insert-anchor" data-kind="audio" data-index="${audioOrdinal}">Audio ${audioOrdinal + 1}</button>`);
      audioOrdinal += 1;
    });
    return buttons.join("");
  }

  function renderTimeline() {
    const track = $("[data-role='timeline-track']", root());
    if (!track) return;
    const frames = alignFrames(state.params.frames);
    const duration = frames / 24;
    $("[data-role='duration-label']", root()).textContent = `${frames}f · ${duration.toFixed(2)}s`;
    let chips = `<span class="h3s-clip-mode">${MODE_META[state.mode].short}</span>`;
    if (state.firstFrame) chips += `<span class="h3s-clip-chip image">首帧 · ${esc(assetById(state.firstFrame)?.name || "图片")}</span>`;
    if (state.lastFrame) chips += `<span class="h3s-clip-chip image">尾帧 · ${esc(assetById(state.lastFrame)?.name || "图片")}</span>`;
    for (const [kind, items] of Object.entries(state.references)) for (const id of items) if (id) chips += `<span class="h3s-clip-chip ${kind}">${KIND_META[kind].icon} ${esc(assetById(id)?.name || kind)}</span>`;
    track.innerHTML = `<div class="h3s-video-clip"><div class="h3s-clip-thumbs">${Array.from({length: 12}, (_, i) => `<i style="opacity:${0.35 + i / 20}"></i>`).join("")}</div><div class="h3s-clip-info">${chips}<strong>${esc(state.params.prompt || "未填写提示词")}</strong></div></div>`;
  }

  function renderInspector() {
    const content = $("[data-role='inspector-content']", root());
    if (!content) return;
    $$('[data-inspector-tab]', root()).forEach((button) => button.classList.toggle("active", button.dataset.inspectorTab === state.inspectorTab));
    if (state.inspectorTab === "asset") content.innerHTML = assetInspectorHtml();
    else content.innerHTML = projectInspectorHtml();
  }

  function field(label, html, hint = "") { return `<label class="h3s-field"><span>${label}${hint ? `<i title="${esc(hint)}">?</i>` : ""}</span>${html}</label>`; }

  function projectInspectorHtml() {
    const p = state.params;
    const mpOptions = MEGAPIXELS.map((mp) => {
      const exact = p.aspect_ratio === "16:9" && Number(p.rounding_multiple) === 32 ? H3_16_9_32[String(Number(mp))] : null;
      const label = Number(mp) === 1 ? "1.0" : String(mp);
      return `<option value="${mp}" ${Number(p.megapixels) === Number(mp) ? "selected" : ""}>${label} MP${exact ? ` · ${exact[0]} × ${exact[1]}` : ""}</option>`;
    }).join("");
    const calculated = calculatedResolution();
    return `<section class="h3s-insp-section open"><header data-action="toggle-section"><div><i>01</i><strong>模型与组件</strong></div><span>⌃</span></header><div class="h3s-insp-body">
      ${field("H3 扩散模型", `<select data-param="model">${modelOptions(state.catalog.models, p.model, state.backend.ready ? "选择 H3 模型" : "连接后端后读取")}</select>`)}
      <label class="h3s-toggle-line"><input type="checkbox" data-param="auto_model" ${p.auto_model ? "checked" : ""}><span><b>按模式自动匹配模型</b><small>首尾帧使用 FL2VA，多参考使用 Ref2VA</small></span></label>
      ${field("MiniMax 文本编码器", `<select data-param="text_encoder">${modelOptions(state.catalog.text_encoders, p.text_encoder, "选择文本编码器")}</select>`)}
      <div class="h3s-field-grid">${field("视频 VAE", `<select data-param="video_vae">${modelOptions(state.catalog.vaes, p.video_vae, "视频 VAE")}</select>`)}${field("音频 VAE", `<select data-param="audio_vae">${modelOptions(state.catalog.vaes, p.audio_vae, "音频 VAE")}</select>`)}</div>
      <div class="h3s-field-grid">${field("模型精度", `<select data-param="weight_dtype"><option>default</option><option ${p.weight_dtype === "fp8_e4m3fn" ? "selected" : ""}>fp8_e4m3fn</option><option ${p.weight_dtype === "fp8_e4m3fn_fast" ? "selected" : ""}>fp8_e4m3fn_fast</option><option ${p.weight_dtype === "fp8_e5m2" ? "selected" : ""}>fp8_e5m2</option></select>`)}${field("编码器设备", `<select data-param="clip_device"><option value="default">自动</option><option value="cpu" ${p.clip_device === "cpu" ? "selected" : ""}>CPU</option></select>`)}</div>
      <button class="h3s-wide-secondary" data-side-tab="loras">管理 LoRA 栈 <b>${state.loras.filter((l) => l.enabled).length}</b></button>
    </div></section>
    <section class="h3s-insp-section open"><header data-action="toggle-section"><div><i>02</i><strong>画面与时长</strong></div><span>⌃</span></header><div class="h3s-insp-body">
      ${field("原生宽高比", `<select data-param="aspect_ratio">${Object.keys(ASPECT_RATIOS).map((ratio) => `<option value="${ratio}" ${p.aspect_ratio === ratio ? "selected" : ""}>${ratio}</option>`).join("")}</select>`, "裁剪框会始终和最终输出比例联动")}
      ${field("百万像素", `<select data-param="megapixels">${mpOptions}</select>`, "16:9 且倍数为 32 时使用 H3 原生推荐尺寸表")}
      <div class="h3s-resolution-row">${field("取整倍数", `<input type="number" data-param="rounding_multiple" min="1" max="512" step="1" value="${p.rounding_multiple}">`, "可自由输入，不强制为 32")}<button data-action="apply-resolution">应用比例参数</button></div>
      <label class="h3s-toggle-line"><input type="checkbox" data-param="resolution_linked" ${p.resolution_linked ? "checked" : ""}><span><b>比例、百万像素与宽高联动</b><small>关闭后可独立修改宽高；提交时仍按自定义倍数取整</small></span></label>
      <div class="h3s-number-pair">${field("宽度", `<input type="number" data-param="width" min="32" max="4096" step="1" value="${p.width}">`)}<button data-action="swap-resolution">⇄</button>${field("高度", `<input type="number" data-param="height" min="32" max="4096" step="1" value="${p.height}">`)}</div>
      <div class="h3s-resolution-note"><i></i><span>${p.resolution_linked ? `联动结果 ${calculated[0]} × ${calculated[1]}` : `自定义 ${p.width} × ${p.height}`} · ${esc(p.aspect_ratio)} · ${Number(p.megapixels)} MP · 倍数 ${Number(p.rounding_multiple)}</span></div>
      <div class="h3s-number-pair">${field("帧数", `<input type="number" data-param="frames" min="5" max="3600" step="17" value="${p.frames}">`, "提交时自动向上对齐 17k+5")}${field("秒数", `<input type="number" data-role="seconds" min="0.21" step="0.1" value="${(alignFrames(p.frames) / 24).toFixed(2)}">`, "H3 生成基准固定为 24 FPS")}</div>
      <div class="h3s-range-note ${alignFrames(p.frames) < 124 || alignFrames(p.frames) > 362 ? "warning" : ""}"><i></i><span>${alignFrames(p.frames)} 帧 · ${(alignFrames(p.frames) / 24).toFixed(2)} 秒${alignFrames(p.frames) < 124 || alignFrames(p.frames) > 362 ? " · 超出主要训练范围 124–362 帧" : " · 位于推荐训练范围"}</span></div>
      ${state.mode === "ref" ? field("参考图尺寸", `<select data-param="ref_image_size"><option value="match">匹配生成面积（较快）</option><option value="max" ${p.ref_image_size === "max" ? "selected" : ""}>2048 短边（身份更稳、更慢）</option></select>`) : ""}
    </div></section>
    <section class="h3s-insp-section open"><header data-action="toggle-section"><div><i>03</i><strong>采样参数</strong></div><span>⌃</span></header><div class="h3s-insp-body">
      <div class="h3s-field-grid">${field("Steps", `<input type="number" data-param="steps" min="1" max="200" value="${p.steps}">`)}${field("Denoise", `<input type="number" data-param="denoise" min="0.01" max="1" step="0.01" value="${p.denoise}">`)}</div>
      ${field("Seed", `<div class="h3s-seed"><input type="number" data-param="seed" min="0" value="${p.seed}" ${p.random_seed ? "disabled" : ""}><button data-action="random-seed">${icon("dice")}</button><label><input type="checkbox" data-param="random_seed" ${p.random_seed ? "checked" : ""}>随机</label></div>`)}
      <div class="h3s-field-grid">${field("Sampler", `<select data-param="sampler">${modelOptions(state.catalog.samplers, p.sampler, "euler")}</select>`)}${field("Scheduler", `<select data-param="scheduler">${modelOptions(state.catalog.schedulers, p.scheduler, "simple")}</select>`)}</div>
      <div class="h3s-field-grid">${field("Video Shift", `<input type="number" data-param="shift_video" min="0.01" max="100" step="0.01" value="${p.shift_video}">`)}${field("Audio Shift", `<input type="number" data-param="shift_audio" min="0.01" max="100" step="0.01" value="${p.shift_audio}">`)}</div>
    </div></section>
    <section class="h3s-insp-section"><header data-action="toggle-section"><div><i>04</i><strong>导出设置</strong></div><span>⌄</span></header><div class="h3s-insp-body">
      ${field("文件名前缀", `<input data-param="filename_prefix" value="${esc(p.filename_prefix)}">`)}
      <div class="h3s-field-grid">${field("封装", `<select data-param="output_format"><option value="auto">自动</option><option value="mp4" ${p.output_format === "mp4" ? "selected" : ""}>MP4</option><option value="webm" ${p.output_format === "webm" ? "selected" : ""}>WebM</option><option value="mkv" ${p.output_format === "mkv" ? "selected" : ""}>MKV</option></select>`)}${field("编码", `<select data-param="output_codec"><option value="auto">自动</option><option value="h264" ${p.output_codec === "h264" ? "selected" : ""}>H.264</option></select>`)}</div>
      <div class="h3s-field-grid">${field("导出 FPS", `<input type="number" data-param="output_fps" min="1" max="120" value="${p.output_fps}">`, "非 24 FPS 会改变播放速度")}${field("CRF", `<input type="number" data-param="output_crf" min="0" max="51" value="${p.output_crf}" ${p.output_codec === "h264" ? "" : "disabled"}>`)}</div>
      ${field("位深", `<select data-param="bit_depth"><option value="8">8-bit</option><option value="10" ${Number(p.bit_depth) === 10 ? "selected" : ""}>10-bit</option></select>`)}
    </div></section>`;
  }

  function assetInspectorHtml() {
    const asset = selectedAsset();
    if (!asset) return '<div class="h3s-inspector-empty"><i>◇</i><strong>未选择素材</strong><span>点击左侧素材卡查看属性</span></div>';
    const generatedActions = asset.jobId ? `<div class="h3s-asset-use"><button data-action="reuse-job" data-id="${esc(asset.jobId)}">复用 Seed 与参数</button><a href="${API}/jobs/${encodeURIComponent(asset.jobId)}/metadata" download>参数 JSON</a></div>` : "";
    return `<div class="h3s-selected-preview">${asset.kind === "image" ? `<img src="${esc(asset.url)}">` : asset.kind === "video" ? `<video src="${esc(asset.url)}" controls></video>` : '<div class="h3s-big-audio">♫</div>'}</div><section class="h3s-insp-section open"><header><div><i>${KIND_META[asset.kind].icon}</i><strong>${KIND_META[asset.kind].label}素材</strong></div></header><div class="h3s-insp-body"><dl class="h3s-meta"><dt>名称</dt><dd>${esc(asset.name)}</dd><dt>后端文件</dt><dd>${esc(asset.file)}</dd><dt>大小</dt><dd>${esc(formatBytes(asset.size) || "未知")}</dd>${asset.summary?.seed !== undefined ? `<dt>Seed</dt><dd>${esc(asset.summary.seed)}</dd>` : ""}</dl>${asset.source === "generated" ? generatedActions : asset.kind === "image" ? '<div class="h3s-asset-use"><button data-action="assign-selected" data-target="first">设为首帧</button><button data-action="assign-selected" data-target="last">设为尾帧</button><button data-action="assign-selected" data-target="ref">加入参考</button></div>' : `<div class="h3s-asset-use"><button data-action="assign-selected" data-target="ref">加入${KIND_META[asset.kind].label}参考</button></div>`}</div></section>`;
  }

  function bindShell() {
    const target = root();
    target.addEventListener("click", handleClick);
    target.addEventListener("input", handleInput);
    target.addEventListener("change", handleChange);
    target.addEventListener("dragstart", handleDragStart);
    target.addEventListener("dragover", handleDragOver);
    target.addEventListener("drop", handleDrop);
    $("[data-role='asset-file-input']", target).addEventListener("change", (event) => uploadFiles(event.target.files));
    $("[data-role='project-file-input']", target).addEventListener("change", importProjectFile);
    document.addEventListener("keydown", (event) => {
      if (!isVisible() || ["INPUT", "TEXTAREA", "SELECT"].includes(document.activeElement?.tagName)) return;
      if (event.key.toLowerCase() === "g") generate();
      if (event.key === "Escape") closeModal();
    });
  }

  function handleClick(event) {
    const mode = event.target.closest("[data-mode]");
    if (mode) return setMode(mode.dataset.mode);
    const assetFilter = event.target.closest("[data-asset-filter]");
    if (assetFilter) {
      state.assetFilter = assetFilter.dataset.assetFilter;
      renderSidebar();
      return;
    }
    const assetSource = event.target.closest("[data-asset-source]");
    if (assetSource) { state.assetSource = assetSource.dataset.assetSource; renderSidebar(); return; }
    const taskFilter = event.target.closest("[data-task-filter]");
    if (taskFilter) { state.taskFilter = taskFilter.dataset.taskFilter; renderTaskSurfaces(); return; }
    const sideTab = event.target.closest("[data-side-tab]");
    if (sideTab) { state.sidebarTab = sideTab.dataset.sideTab; renderSidebar(); return; }
    const inspTab = event.target.closest("[data-inspector-tab]");
    if (inspTab) { state.inspectorTab = inspTab.dataset.inspectorTab; renderInspector(); return; }
    const assetCard = event.target.closest(".h3s-asset-card");
    if (assetCard && !event.target.closest("button")) {
      const asset = assetById(assetCard.dataset.assetId);
      if (asset?.source === "generated") { openMediaViewer(asset); return; }
      state.selectedAssetId = assetCard.dataset.assetId;
      state.inspectorTab = "asset";
      renderSidebar(); renderInspector();
      if (state.mode !== "t2v") quickAssign(assetCard.dataset.assetId);
      return;
    }
    const action = event.target.closest("[data-action]");
    if (!action) return;
    const name = action.dataset.action;
    if (name === "choose-assets") $("[data-role='asset-file-input']", root()).click();
    else if (name === "refresh-catalog") loadCatalog(true);
    else if (name === "open-settings") openSettings();
    else if (name === "open-task-panel") openTaskPanel();
    else if (name === "generate") generate();
    else if (name === "cancel-job") cancelActiveJob();
    else if (name === "close-result") { state.result = null; renderStage(); }
    else if (name === "preview-asset") openMediaViewer(assetById(action.dataset.id));
    else if (name === "preview-output") { const job = state.jobs.find((item) => item.id === action.dataset.jobId) || state.result; openMediaViewer(preferredOutput(job), job?.summary); }
    else if (name === "clear-slot") { const slot = action.dataset.slot; state[slot === "first" ? "firstFrame" : "lastFrame"] = null; state.frameCrops[slot] = null; projectChanged(); }
    else if (name === "fill-slot") chooseSelectedForSlot(action.dataset.slot);
    else if (name === "open-crop") openCropEditor(action.dataset.slot);
    else if (name === "apply-crop") applyCropEditor();
    else if (name === "reset-crop") resetCropEditor();
    else if (name === "clear-ref") clearReference(action.dataset.kind, Number(action.dataset.index));
    else if (name === "toggle-ref-audio") { const index = Number(action.dataset.index); state.refVideoAudio[index] = state.refVideoAudio[index] === false; projectChanged(); }
    else if (name === "remove-asset") removeAsset(action.dataset.id);
    else if (name === "assign-selected") assignAsset(state.selectedAssetId, action.dataset.target);
    else if (name === "insert-anchor") insertAnchor(action.dataset.kind, Number(action.dataset.index));
    else if (name === "toggle-section") action.closest(".h3s-insp-section")?.classList.toggle("open");
    else if (name === "apply-resolution") { applyResolutionSelection(); projectChanged(); }
    else if (name === "swap-resolution") {
      const reciprocal = Object.keys(ASPECT_RATIOS).find((ratio) => {
        const pair = ASPECT_RATIOS[ratio]; const current = ASPECT_RATIOS[state.params.aspect_ratio];
        return current && pair[0] === current[1] && pair[1] === current[0];
      });
      if (state.params.resolution_linked && reciprocal) { state.params.aspect_ratio = reciprocal; applyResolutionSelection(); }
      else { [state.params.width, state.params.height] = [state.params.height, state.params.width]; reframeAssignedCrops(); }
      projectChanged();
    }
    else if (name === "random-seed") { state.params.seed = Math.floor(Math.random() * Number.MAX_SAFE_INTEGER); renderInspector(); debounceSave(); }
    else if (name === "open-lora-browser") openLoraBrowser();
    else if (name === "remove-lora") { state.loras.splice(Number(action.dataset.index), 1); projectChanged(); }
    else if (name === "toggle-lora") { const item = state.loras[Number(action.dataset.index)]; item.enabled = !item.enabled; projectChanged(); }
    else if (name === "move-lora") moveLora(Number(action.dataset.index), Number(action.dataset.dir));
    else if (name === "save-lora-preset") saveLoraPreset();
    else if (name === "delete-lora-preset") deleteLoraPreset();
    else if (name === "select-job") selectJob(action.dataset.id);
    else if (name === "cancel-specific-job") cancelJob(action.dataset.id);
    else if (name === "job-details") openJobDetails(action.dataset.id);
    else if (name === "reuse-job") reuseJobParameters(action.dataset.id);
    else if (name === "copy-job-id") copyText(action.dataset.value, "任务 ID 已复制");
    else if (name === "clear-queue") clearJobs("queue");
    else if (name === "clear-completed") clearJobs("completed");
    else if (name === "toggle-task-sort") { state.taskSort = state.taskSort === "newest" ? "oldest" : "newest"; renderTaskSurfaces(); }
    else if (name === "refresh-jobs") refreshJobs();
    else if (name === "export-project") exportProject();
    else if (name === "preview-workflow") previewWorkflow();
    else if (name === "close-modal") closeModal();
    else if (name === "save-settings") saveSettings();
    else if (name === "clear-api-key") clearApiKey();
    else if (name === "start-backend") { closeModal(); state.entered = true; ensureBackend(true); }
    else if (name === "stop-backend") stopBackend();
    else if (name === "refresh-logs") refreshLogs();
    else if (name === "import-project") $("[data-role='project-file-input']", root()).click();
    else if (name === "add-lora") addLora(action.dataset.name);
  }

  function handleInput(event) {
    if (event.target.matches("textarea[data-param='prompt']")) {
      state.params.prompt = event.target.value;
      $("[data-role='prompt-count']", root()).textContent = `${event.target.value.length} 字`;
      renderTimeline(); debounceSave();
    } else if (event.target.matches("[data-role='asset-search']")) {
      state.assetSearch = event.target.value;
      renderSidebar();
      const input = $("[data-role='asset-search']", root());
      input?.focus(); input?.setSelectionRange(state.assetSearch.length, state.assetSearch.length);
    } else if (event.target.matches("[data-role='lora-search']")) {
      renderLoraBrowserList(event.target.value);
    } else if (event.target.matches("[data-role='crop-zoom']")) {
      updateCropZoom(Number(event.target.value));
    }
  }

  function valueFromInput(input) {
    if (input.type === "checkbox") return input.checked;
    if (input.type === "number") return Number(input.value);
    return input.value;
  }

  function handleChange(event) {
    const setting = event.target.dataset.setting;
    if (setting === "backend_mode") {
      applySettingsModeVisibility();
      return;
    }
    const param = event.target.dataset.param;
    if (param && param !== "prompt") {
      state.params[param] = valueFromInput(event.target);
      if (param === "frames") state.params.frames = alignFrames(state.params.frames);
      if (param === "rounding_multiple") state.params.rounding_multiple = clamp(Math.round(Number(state.params.rounding_multiple) || 1), 1, 512);
      if (param === "random_seed") renderInspector();
      if (param === "auto_model" && state.params.auto_model) autoSelectModels();
      if (["aspect_ratio", "megapixels", "rounding_multiple"].includes(param) && state.params.resolution_linked) applyResolutionSelection();
      if (param === "rounding_multiple" && !state.params.resolution_linked) {
        state.params.width = alignDimension(state.params.width);
        state.params.height = alignDimension(state.params.height);
        reframeAssignedCrops();
      }
      if (param === "resolution_linked" && state.params.resolution_linked) applyResolutionSelection();
      if (["width", "height"].includes(param)) { state.params[param] = alignDimension(state.params[param]); state.params.resolution_linked = false; reframeAssignedCrops(); }
      projectChanged(param === "model" || param === "text_encoder" || param.includes("vae") ? "soft" : "full");
      return;
    }
    const loraField = event.target.dataset.loraField;
    if (loraField) {
      const item = state.loras[Number(event.target.dataset.index)];
      item[loraField] = valueFromInput(event.target);
      renderSidebar(); debounceSave();
      return;
    }
    if (event.target.matches("[data-role='seconds']")) {
      state.params.frames = alignFrames(Number(event.target.value) * 24);
      projectChanged();
    } else if (event.target.matches("[data-asset-filter]")) {
      state.assetFilter = event.target.dataset.assetFilter;
      renderSidebar();
    } else if (event.target.matches("[data-role='lora-preset']")) {
      const preset = state.loraPresets.find((item) => item.name === event.target.value);
      if (preset) { state.loras = JSON.parse(JSON.stringify(preset.loras)); projectChanged(); }
    }
  }

  function handleDragStart(event) {
    const card = event.target.closest("[data-asset-id]");
    if (!card) return;
    event.dataTransfer.setData("application/x-h3s-asset", card.dataset.assetId);
    event.dataTransfer.effectAllowed = "copy";
  }

  function handleDragOver(event) {
    if (event.target.closest("[data-slot], [data-ref-kind], [data-role='dropzone']")) {
      event.preventDefault();
      event.dataTransfer.dropEffect = "copy";
    }
  }

  function handleDrop(event) {
    const dropzone = event.target.closest("[data-role='dropzone']");
    const slot = event.target.closest("[data-slot]");
    const ref = event.target.closest("[data-ref-kind]");
    const localFiles = Array.from(event.dataTransfer.files || []);
    if (localFiles.length) {
      event.preventDefault();
      if (slot) {
        const image = localFiles.find((file) => kindFromName(file.name, file.type) === "image");
        if (!image) return toast("首尾帧槽位只能拖入图片文件", "warning");
        uploadFiles([image], { slot: slot.dataset.slot });
      } else if (ref) {
        const file = localFiles.find((item) => kindFromName(item.name, item.type) === ref.dataset.refKind);
        if (!file) return toast(`这里需要${KIND_META[ref.dataset.refKind].label}文件`, "warning");
        uploadFiles([file], { refKind: ref.dataset.refKind, refIndex: Number(ref.dataset.refIndex) });
      } else if (dropzone) uploadFiles(localFiles);
      return;
    }
    const id = event.dataTransfer.getData("application/x-h3s-asset");
    if (!id) return;
    if (slot) { event.preventDefault(); assignAsset(id, slot.dataset.slot); }
    else if (ref) { event.preventDefault(); assignReference(id, ref.dataset.refKind, Number(ref.dataset.refIndex)); }
  }

  function setMode(mode) {
    if (!MODE_META[mode] || mode === state.mode) return;
    state.mode = mode;
    state.result = null;
    if (state.params.auto_model) autoSelectModels();
    projectChanged();
  }

  function projectChanged(level = "full") {
    debounceSave();
    if (level !== "soft") {
      renderStage(); renderTimeline(); renderPrompt();
    }
    renderInspector(); renderSidebar();
  }

  async function uploadFiles(files, destination = null) {
    const accepted = Array.from(files || []).filter((file) => kindFromName(file.name, file.type));
    if (!accepted.length) return toast("没有可用的图片、视频或音频文件", "warning");
    if (!state.backend.ready) {
      toast("需要先启动并连接 H3 后端", "warning");
      await ensureBackend(true);
      if (!state.backend.ready) return;
    }
    state.uploadBusy = true; renderSidebar();
    const uploaded = [];
    try {
      for (const file of accepted) {
        try {
        const kind = kindFromName(file.name, file.type);
        const dimensions = kind === "image" ? await imageSizeFromFile(file) : { width: 0, height: 0 };
        const form = new FormData(); form.append("file", file, file.name);
        const result = await request("/assets/upload", { method: "POST", body: form });
        const asset = { id: uid("asset"), name: file.name, kind, file: result.file, url: result.url, size: file.size, source: "imported", width: dimensions.width, height: dimensions.height, createdAt: Date.now() / 1000 };
        state.assets.push(asset);
        uploaded.push(asset);
        state.selectedAssetId = asset.id;
        toast(`${file.name} 已加入素材库`, "success", 2400);
        } catch (error) { toast(`${file.name} 上传失败：${error.message}`, "error", 7000); }
      }
    } finally {
      state.uploadBusy = false;
      const input = $("[data-role='asset-file-input']", root()); if (input) input.value = "";
    }
    const assigned = uploaded[0];
    if (assigned && destination?.slot) assignAsset(assigned.id, destination.slot);
    else if (assigned && destination?.refKind) assignReference(assigned.id, destination.refKind, destination.refIndex);
    else projectChanged();
    return uploaded;
  }

  function quickAssign(id) {
    const asset = assetById(id); if (!asset) return;
    if (state.mode === "i2v" && asset.kind === "image" && !state.firstFrame) assignAsset(id, "first");
    else if (state.mode === "fl2v" && asset.kind === "image" && (!state.firstFrame || !state.lastFrame)) assignAsset(id, state.firstFrame ? "last" : "first");
  }

  function chooseSelectedForSlot(slot) {
    const asset = selectedAsset();
    if (!asset || asset.kind !== "image") return toast("请先在素材库选择一张图片", "warning");
    assignAsset(asset.id, slot);
  }

  function assignAsset(id, target) {
    const asset = assetById(id); if (!asset) return;
    if (asset.source === "generated") return toast("生成结果已归档，但作为输入前请先下载并重新导入", "warning", 6000);
    if ((target === "first" || target === "last") && asset.kind !== "image") return toast("首尾帧只能使用图片", "warning");
    if (target === "first") { state.firstFrame = id; state.frameCrops.first = null; }
    else if (target === "last") { state.lastFrame = id; state.frameCrops.last = null; }
    else assignReference(id, asset.kind);
    projectChanged();
    if (target === "first" || target === "last") {
      ensureImageSize(asset).then(() => {
        if (state[target === "first" ? "firstFrame" : "lastFrame"] === id && !state.frameCrops[target]) {
          state.frameCrops[target] = makeCrop(asset);
          projectChanged();
        }
      }).catch((error) => toast(error.message, "warning"));
    }
  }

  function assignReference(id, forcedKind = null, forcedIndex = null) {
    const asset = assetById(id); if (!asset) return;
    const kind = forcedKind || asset.kind;
    if (asset.kind !== kind) return toast(`这里需要${KIND_META[kind].label}素材`, "warning");
    const max = kind === "image" ? 9 : 3;
    const items = state.references[kind];
    const next = items.filter(Boolean).length;
    const requested = forcedIndex == null ? next : forcedIndex;
    const target = Math.min(requested, next);
    if (target >= max) return toast(`${KIND_META[kind].label}参考已达到 ${max} 个上限`, "warning");
    items[target] = id;
    if (kind === "video" && state.refVideoAudio[target] == null) state.refVideoAudio[target] = true;
    state.mode = "ref";
    projectChanged();
  }

  function clearReference(kind, index) {
    state.references[kind].splice(index, 1);
    if (kind === "video") state.refVideoAudio.splice(index, 1);
    projectChanged();
  }

  async function openCropEditor(slot) {
    const asset = assetById(slot === "first" ? state.firstFrame : state.lastFrame);
    if (!asset) return toast("请先放入首帧或尾帧图片", "warning");
    try { await ensureImageSize(asset); } catch (error) { return toast(error.message, "error"); }
    const crop = makeCrop(asset, state.frameCrops[slot]);
    state.cropEditor = { slot, asset, crop, image: null, transform: null, dragging: null };
    const layer = $("[data-role='modal-layer']", root());
    layer.innerHTML = `<div class="h3s-modal-backdrop" data-action="close-modal"></div><div class="h3s-modal h3s-crop-modal"><header><div><strong>${slot === "first" ? "首帧" : "尾帧"}取景</strong><span>裁剪框已锁定为 ${state.params.width}:${state.params.height}（${state.params.width} × ${state.params.height}）</span></div><button data-action="close-modal">${icon("close")}</button></header><div class="h3s-crop-workspace"><canvas data-role="crop-canvas" width="1000" height="620"></canvas><div class="h3s-crop-help">拖动亮框移动取景中心；下方滑杆缩放生成范围。原图不会被修改。</div></div><div class="h3s-crop-footer"><div class="h3s-crop-control"><span>显示范围</span><input type="range" min="1" max="6" step="0.01" value="${crop.zoom}" data-role="crop-zoom"><b data-role="crop-zoom-label">${crop.zoom.toFixed(2)}×</b></div><div class="h3s-crop-coordinates" data-role="crop-coordinates"></div><button data-action="reset-crop">重置</button><button class="h3s-primary-btn" data-action="apply-crop">应用取景</button></div></div>`;
    const image = new Image();
    image.onload = () => { if (state.cropEditor) { state.cropEditor.image = image; drawCropEditor(); bindCropCanvas(); } };
    image.onerror = () => toast("无法载入取景图片", "error");
    image.src = asset.url;
  }

  function drawCropEditor() {
    const editor = state.cropEditor;
    const canvas = $("[data-role='crop-canvas']", root());
    if (!editor?.image || !canvas) return;
    const context = canvas.getContext("2d");
    const scale = Math.min(canvas.width / editor.asset.width, canvas.height / editor.asset.height);
    const drawWidth = editor.asset.width * scale;
    const drawHeight = editor.asset.height * scale;
    const offsetX = (canvas.width - drawWidth) / 2;
    const offsetY = (canvas.height - drawHeight) / 2;
    editor.transform = { scale, offsetX, offsetY };
    context.clearRect(0, 0, canvas.width, canvas.height);
    context.fillStyle = "#090b12";
    context.fillRect(0, 0, canvas.width, canvas.height);
    context.drawImage(editor.image, offsetX, offsetY, drawWidth, drawHeight);
    context.fillStyle = "rgba(3, 5, 10, .68)";
    context.fillRect(offsetX, offsetY, drawWidth, drawHeight);
    const rect = {
      x: offsetX + editor.crop.x * scale,
      y: offsetY + editor.crop.y * scale,
      width: editor.crop.width * scale,
      height: editor.crop.height * scale,
    };
    context.save();
    context.beginPath(); context.rect(rect.x, rect.y, rect.width, rect.height); context.clip();
    context.drawImage(editor.image, offsetX, offsetY, drawWidth, drawHeight);
    context.restore();
    context.strokeStyle = "#ff714b";
    context.lineWidth = 4;
    context.strokeRect(rect.x, rect.y, rect.width, rect.height);
    context.strokeStyle = "rgba(255,255,255,.72)";
    context.lineWidth = 1;
    for (let index = 1; index < 3; index += 1) {
      context.beginPath(); context.moveTo(rect.x + rect.width * index / 3, rect.y); context.lineTo(rect.x + rect.width * index / 3, rect.y + rect.height); context.stroke();
      context.beginPath(); context.moveTo(rect.x, rect.y + rect.height * index / 3); context.lineTo(rect.x + rect.width, rect.y + rect.height * index / 3); context.stroke();
    }
    editor.screenRect = rect;
    const label = $("[data-role='crop-zoom-label']", root());
    if (label) label.textContent = `${editor.crop.zoom.toFixed(2)}×`;
    const coordinates = $("[data-role='crop-coordinates']", root());
    if (coordinates) coordinates.textContent = `${editor.crop.width} × ${editor.crop.height} · X ${editor.crop.x} · Y ${editor.crop.y}`;
  }

  function bindCropCanvas() {
    const canvas = $("[data-role='crop-canvas']", root());
    if (!canvas || canvas.dataset.bound) return;
    canvas.dataset.bound = "1";
    const point = (event) => {
      const bounds = canvas.getBoundingClientRect();
      return { x: (event.clientX - bounds.left) * canvas.width / bounds.width, y: (event.clientY - bounds.top) * canvas.height / bounds.height };
    };
    canvas.addEventListener("pointerdown", (event) => {
      const editor = state.cropEditor; if (!editor?.screenRect) return;
      const p = point(event); const r = editor.screenRect;
      if (p.x < r.x || p.x > r.x + r.width || p.y < r.y || p.y > r.y + r.height) return;
      editor.dragging = { start: p, x: editor.crop.x, y: editor.crop.y };
      canvas.setPointerCapture(event.pointerId);
    });
    canvas.addEventListener("pointermove", (event) => {
      const editor = state.cropEditor; if (!editor?.dragging) return;
      const p = point(event); const scale = editor.transform.scale;
      editor.crop.x = clamp(Math.round(editor.dragging.x + (p.x - editor.dragging.start.x) / scale), 0, editor.asset.width - editor.crop.width);
      editor.crop.y = clamp(Math.round(editor.dragging.y + (p.y - editor.dragging.start.y) / scale), 0, editor.asset.height - editor.crop.height);
      commitCropEditor(false);
      drawCropEditor();
    });
    const finish = () => {
      if (!state.cropEditor) return;
      state.cropEditor.dragging = null;
      commitCropEditor(false);
    };
    canvas.addEventListener("pointerup", finish);
    canvas.addEventListener("pointercancel", finish);
  }

  function updateCropZoom(zoom) {
    const editor = state.cropEditor; if (!editor) return;
    editor.crop = makeCrop(editor.asset, { ...editor.crop, zoom });
    commitCropEditor(false);
    drawCropEditor();
  }

  function resetCropEditor() {
    const editor = state.cropEditor; if (!editor) return;
    editor.crop = makeCrop(editor.asset);
    const slider = $("[data-role='crop-zoom']", root()); if (slider) slider.value = "1";
    commitCropEditor(false);
    drawCropEditor();
  }

  function commitCropEditor(render = false) {
    const editor = state.cropEditor;
    if (!editor?.crop) return false;
    state.frameCrops[editor.slot] = { ...editor.crop };
    debounceSave();
    if (render) { renderStage(); renderTimeline(); renderInspector(); }
    return true;
  }

  function applyCropEditor() {
    const editor = state.cropEditor; if (!editor) return;
    commitCropEditor(false);
    closeModal();
    toast("取景范围已应用，生成时不会拉伸原图", "success");
  }

  function openMediaViewer(item, summary = null) {
    if (!item?.url) return toast("该素材没有可浏览地址", "warning");
    const kind = item.kind || outputKind(item);
    const title = item.name || item.filename || "H3 生成结果";
    const detail = summary || item.summary || {};
    let media = `<div class="h3s-result-empty">无法预览该输出</div>`;
    if (kind === "video") media = `<video src="${esc(item.url)}" controls autoplay loop></video>`;
    else if (kind === "image") media = `<img src="${esc(item.url)}" alt="${esc(title)}">`;
    else if (kind === "audio") media = `<div class="h3s-viewer-audio"><span>♫</span><audio src="${esc(item.url)}" controls autoplay></audio></div>`;
    const layer = $("[data-role='modal-layer']", root());
    const reusable = item.jobId && detail.reproducible ? `<button data-action="reuse-job" data-id="${esc(item.jobId)}">复用参数</button>` : "";
    layer.innerHTML = `<div class="h3s-modal-backdrop" data-action="close-modal"></div><div class="h3s-media-viewer"><header><div><strong>${esc(title)}</strong><span>${esc(detail.resolution || "")}${detail.frames ? ` · ${detail.frames}f` : ""}${detail.seed !== undefined ? ` · Seed ${esc(detail.seed)}` : ""}</span></div>${reusable}<a href="${esc(item.url)}" download>${icon("download")}下载</a><button data-action="close-modal">${icon("close")}</button></header><div class="h3s-media-viewer-body">${media}</div></div>`;
  }

  function renderTaskSurfaces() {
    if (state.sidebarTab === "history") renderSidebar();
    const host = $("[data-role='task-panel-body']", root());
    if (host) host.innerHTML = `${taskFiltersHtml()}<div class="h3s-task-panel-list">${taskListHtml(false)}</div>`;
    const detail = $(".h3s-job-detail-modal[data-job-id]", root());
    if (detail?.dataset.jobId) openJobDetails(detail.dataset.jobId);
    updateBackendUi();
  }

  function scheduleTaskRender() {
    if (state.taskRenderQueued) return;
    state.taskRenderQueued = true;
    requestAnimationFrame(() => { state.taskRenderQueued = false; renderTaskSurfaces(); });
  }

  function openTaskPanel() {
    const active = state.jobs.filter((job) => ["queued", "running"].includes(job.state)).length;
    const layer = $("[data-role='modal-layer']", root());
    layer.innerHTML = `<div class="h3s-modal-backdrop" data-action="close-modal"></div><div class="h3s-modal h3s-task-modal"><header><div><strong>${active ? `${active} 个正在运行` : "H3 任务中心"}</strong><span>显示 ComfyUI 实时节点、采样进度和输出</span></div><button data-action="toggle-task-sort" title="切换排序">⇅</button><button data-action="refresh-jobs">${icon("refresh")}</button><button data-action="close-modal">${icon("close")}</button></header><div data-role="task-panel-body"></div></div>`;
    renderTaskSurfaces();
  }

  function openJobDetails(id) {
    const job = state.jobs.find((item) => item.id === id); if (!job) return;
    const progress = jobProgress(job); const eta = jobEta(job); const output = preferredOutput(job);
    const loras = (job.summary?.loras || []).filter((item) => item.enabled !== false).map((item) => `${item.name} (${item.model_strength})`).join("、") || "无";
    const row = (label, value) => `<dt>${label}</dt><dd>${value}</dd>`;
    const layer = $("[data-role='modal-layer']", root());
    layer.innerHTML = `<div class="h3s-modal-backdrop" data-action="close-modal"></div><div class="h3s-modal h3s-job-detail-modal" data-job-id="${esc(job.id)}"><header><div><strong>任务细节</strong><span>${esc(job.summary?.mode_name || "MiniMax H3")}</span></div><button data-action="close-modal">${icon("close")}</button></header><div class="h3s-job-detail-status" data-state="${esc(job.state)}"><div class="h3s-job-thumb">${jobPreviewHtml(job)}</div><div><strong>${job.state === "running" ? `全部：${progress.overall.toFixed(0)}%` : esc({completed: "已完成", failed: "失败", cancelled: "已中断", queued: "排队中"}[job.state] || job.state)}</strong><span>${esc(job.live?.nodeTitle || "等待节点状态")} ${progress.node.toFixed(0)}%</span><div class="h3s-job-progress overall"><i style="width:${progress.overall}%"></i></div></div></div><dl class="h3s-job-detail-grid">${row("工作流", esc(job.summary?.mode_name || job.summary?.mode || "H3"))}${row("任务 ID", `<code>${esc(job.id)}</code><button data-action="copy-job-id" data-value="${esc(job.id)}">复制</button>`)}${row("提交于", formatClock(job.created_at))}${row("开始于", formatClock(job.started_at || job.live?.startedAt))}${row("耗时", formatElapsed(jobElapsed(job)))}${row("预计完成", eta ? `约 ${formatElapsed(eta)}` : "—")}${row("队列位置", job.queue_position == null ? "—" : String(job.queue_position))}${row("当前节点", esc(job.live?.nodeTitle || "—"))}${row("采样步", job.live?.maxSteps ? `${job.live.step || 0} / ${job.live.maxSteps}` : "—")}${row("输出", esc(output?.filename || "—"))}${row("分辨率 / 帧", `${esc(job.summary?.resolution || "—")} · ${job.summary?.frames || "—"}f · ${job.summary?.duration_seconds || "—"}s`)}${row("模型", esc(job.summary?.model || "—"))}${row("LoRA", esc(loras))}${row("Seed", esc(job.summary?.seed ?? "—"))}</dl>${job.error ? `<div class="h3s-job-error"><strong>错误摘要</strong><pre>${esc(job.error)}</pre>${job.live?.fullError ? `<details><summary>完整错误数据</summary><pre>${esc(JSON.stringify(job.live.fullError, null, 2))}</pre></details>` : ""}</div>` : ""}<div class="h3s-job-detail-footer">${["queued", "running"].includes(job.state) ? `<button class="danger" data-action="cancel-specific-job" data-id="${esc(job.id)}">中断任务</button>` : ""}<button data-action="reuse-job" data-id="${esc(job.id)}">复用参数</button><a href="${API}/jobs/${encodeURIComponent(job.id)}/metadata" download>下载参数 JSON</a>${output ? `<button class="h3s-primary-btn" data-action="preview-output" data-job-id="${esc(job.id)}">浏览输出</button>` : ""}</div></div>`;
  }

  async function copyText(value, success) {
    try { await navigator.clipboard.writeText(value); toast(success, "success", 2200); }
    catch (_) { window.prompt("复制以下内容：", value); }
  }

  function removeAsset(id) {
    state.assets = state.assets.filter((asset) => asset.id !== id);
    if (state.selectedAssetId === id) state.selectedAssetId = null;
    if (state.firstFrame === id) { state.firstFrame = null; state.frameCrops.first = null; }
    if (state.lastFrame === id) { state.lastFrame = null; state.frameCrops.last = null; }
    for (const kind of Object.keys(state.references)) {
      if (kind === "video") {
        const kept = [];
        const audio = [];
        state.references.video.forEach((item, index) => {
          if (item !== id) { kept.push(item); audio.push(state.refVideoAudio[index] !== false); }
        });
        state.references.video = kept;
        state.refVideoAudio = audio;
      } else state.references[kind] = state.references[kind].filter((item) => item !== id);
    }
    projectChanged();
  }

  function insertAnchor(kind, index) {
    const tag = kind === "image" ? `<Picture ${index + 1}>` : kind === "video" ? `<Video ${index + 1}>` : `<Audio ${index + 1}>`;
    const textarea = $("textarea[data-param='prompt']", root());
    if (!textarea) return;
    const start = textarea.selectionStart || 0;
    const end = textarea.selectionEnd || start;
    const spacer = start && !/\s$/.test(textarea.value.slice(0, start)) ? " " : "";
    textarea.value = textarea.value.slice(0, start) + spacer + tag + " " + textarea.value.slice(end);
    state.params.prompt = textarea.value;
    textarea.focus(); textarea.selectionStart = textarea.selectionEnd = start + spacer.length + tag.length + 1;
    renderPrompt(); renderTimeline(); debounceSave();
  }

  function moveLora(index, direction) {
    const target = index + direction;
    if (target < 0 || target >= state.loras.length) return;
    [state.loras[index], state.loras[target]] = [state.loras[target], state.loras[index]];
    projectChanged();
  }

  function openLoraBrowser() {
    const layer = $("[data-role='modal-layer']", root());
    layer.innerHTML = `<div class="h3s-modal-backdrop" data-action="close-modal"></div><div class="h3s-modal h3s-lora-modal"><header><div><strong>LoRA 管理器</strong><span>后端共发现 ${state.catalog.loras?.length || 0} 个 LoRA</span></div><button data-action="close-modal">${icon("close")}</button></header><div class="h3s-modal-toolbar"><label class="h3s-search"><span>⌕</span><input data-role="lora-search" autofocus placeholder="搜索 H3 LoRA"></label><button data-action="refresh-catalog">${icon("refresh")}刷新</button></div><div class="h3s-lora-browser-list" data-role="lora-browser-list"></div></div>`;
    renderLoraBrowserList("");
  }

  function renderLoraBrowserList(search) {
    const host = $("[data-role='lora-browser-list']", root()); if (!host) return;
    const added = new Set(state.loras.map((item) => item.name));
    const values = (state.catalog.loras || []).filter((name) => !search || name.toLowerCase().includes(search.toLowerCase()));
    host.innerHTML = values.length ? values.map((name) => `<div class="h3s-lora-browser-item"><div class="h3s-lora-file-icon">L</div><div><strong title="${esc(name)}">${esc(name.split(/[\\/]/).pop())}</strong><small>${esc(name)}</small></div><button data-action="add-lora" data-name="${esc(name)}" ${added.has(name) ? "disabled" : ""}>${added.has(name) ? "已添加" : "+ 添加"}</button></div>`).join("") : '<div class="h3s-empty"><i>⌁</i><strong>没有匹配的 LoRA</strong></div>';
  }

  function addLora(name) {
    if (!name || state.loras.some((item) => item.name === name)) return;
    state.loras.push({ name, enabled: true, model_strength: 1, clip_strength: 0, apply_to_clip: false });
    renderLoraBrowserList($("[data-role='lora-search']", root())?.value || "");
    debounceSave();
    toast("LoRA 已加入加载栈", "success", 2200);
  }

  async function saveLoraPreset() {
    const name = window.prompt("为当前 LoRA 组合命名：");
    if (!name?.trim()) return;
    try {
      const response = await request("/lora-presets", { method: "POST", body: { name: name.trim(), loras: state.loras } });
      state.loraPresets = response.presets || []; renderSidebar(); toast("LoRA 预设已保存", "success");
    } catch (error) { toast(error.message, "error"); }
  }

  async function deleteLoraPreset() {
    const select = $("[data-role='lora-preset']", root());
    if (!select?.value) return toast("请先选择要删除的预设", "warning");
    try {
      const response = await request(`/lora-presets/${encodeURIComponent(select.value)}`, { method: "DELETE" });
      state.loraPresets = response.presets || []; renderSidebar();
    } catch (error) { toast(error.message, "error"); }
  }

  async function generationPayload() {
    const p = state.params;
    const refList = [];
    for (const [kind, ids] of Object.entries(state.references)) for (const [index, id] of ids.entries()) {
      const asset = assetById(id); if (asset) refList.push({ kind, file: asset.file, include_audio: kind === "video" && state.refVideoAudio[index] !== false });
    }
    const first = assetById(state.firstFrame);
    const last = assetById(state.lastFrame);
    let cropsChanged = false;
    if (["i2v", "fl2v"].includes(state.mode) && first) {
      await ensureImageSize(first);
      if (!state.frameCrops.first) { state.frameCrops.first = makeCrop(first); cropsChanged = true; }
    }
    if (state.mode === "fl2v" && last) {
      await ensureImageSize(last);
      if (!state.frameCrops.last) { state.frameCrops.last = makeCrop(last); cropsChanged = true; }
    }
    if (cropsChanged) { debounceSave(); renderStage(); }
    return {
      mode: state.mode,
      model: p.model,
      text_encoder: p.text_encoder,
      video_vae: p.video_vae,
      audio_vae: p.audio_vae,
      weight_dtype: p.weight_dtype,
      clip_device: p.clip_device,
      prompt: p.prompt,
      width: p.width,
      height: p.height,
      aspect_ratio: p.aspect_ratio,
      megapixels: p.megapixels,
      rounding_multiple: p.rounding_multiple,
      frames: p.frames,
      steps: p.steps,
      seed: p.random_seed ? Math.floor(Math.random() * Number.MAX_SAFE_INTEGER) : p.seed,
      sampler: p.sampler,
      scheduler: p.scheduler,
      shift_video: p.shift_video,
      shift_audio: p.shift_audio,
      denoise: p.denoise,
      ref_image_size: p.ref_image_size,
      first_frame: ["i2v", "fl2v"].includes(state.mode) ? first?.file || null : null,
      last_frame: state.mode === "fl2v" ? last?.file || null : null,
      first_frame_crop: ["i2v", "fl2v"].includes(state.mode) ? state.frameCrops.first : null,
      last_frame_crop: state.mode === "fl2v" ? state.frameCrops.last : null,
      references: state.mode === "ref" ? refList : [],
      loras: state.loras,
      output: { filename_prefix: p.filename_prefix, format: p.output_format, codec: p.output_codec, crf: p.output_crf, fps: p.output_fps, bit_depth: p.bit_depth },
    };
  }

  async function generate() {
    if (!state.backend.ready) { toast("正在准备 H3 后端…", "warning"); await ensureBackend(true); if (!state.backend.ready) return; }
    const button = $("[data-action='generate']", root());
    button.disabled = true; button.classList.add("busy");
    try {
      const payload = await generationPayload();
      const job = mergeJob(await request("/jobs", { method: "POST", body: payload }));
      state.activeJobId = job.id; state.result = null;
      state.sidebarTab = "history"; renderSidebar(); renderStage(); updateJobControls();
      startJobTracking(job); toast("H3 任务已提交", "success");
      if (job.summary?.warnings?.length) toast(job.summary.warnings.join("；"), "warning", 9000);
    } catch (error) { toast(`无法开始生成：${error.message}`, "error", 9000); }
    finally { button.disabled = false; button.classList.remove("busy"); }
  }

  function mergeJob(serverJob) {
    const index = state.jobs.findIndex((item) => item.id === serverJob.id);
    const previous = index >= 0 ? state.jobs[index] : null;
    const live = { ...(previous?.live || {}), ...(serverJob.progress || {}), ...(serverJob.live || {}) };
    const revision = Number(serverJob.progress?.previewRevision || 0);
    if (serverJob.preview_url && revision >= Number(previous?.live?.previewRevision || 0)) live.previewUrl = serverJob.preview_url;
    const merged = { ...serverJob, live };
    if (previous?.state === "running" && serverJob.state === "queued") merged.state = "running";
    if (index >= 0) state.jobs[index] = merged; else state.jobs.unshift(merged);
    if (merged.state === "completed") { syncGeneratedAssets(merged); ensureJobThumbnail(merged); }
    return merged;
  }

  function comfyWebSocketUrl(clientId) {
    const base = new URL(state.backend.url || state.config.comfy_url || "http://127.0.0.1:8189", window.location.href);
    base.protocol = base.protocol === "https:" ? "wss:" : "ws:";
    base.pathname = `${base.pathname.replace(/\/$/, "")}/ws`;
    base.search = `clientId=${encodeURIComponent(clientId)}`;
    base.hash = "";
    return base.toString();
  }

  function activeLocalJob(jobId) {
    return state.jobs.find((item) => item.id === jobId) || null;
  }

  function updateLiveOverall(job) {
    const live = job.live || (job.live = {});
    const count = Math.max(1, Number(job.summary?.node_count || 1));
    const completed = new Set(live.completedNodes || []);
    live.completedNodes = Array.from(completed);
    live.overallPercent = clamp((completed.size + Number(live.nodePercent || 0) / 100) / count * 100, 0, 99.9);
  }

  function setLiveNode(job, nodeId) {
    const live = job.live || (job.live = {});
    if (live.nodeId && live.nodeId !== nodeId) {
      const completed = new Set(live.completedNodes || []); completed.add(String(live.nodeId)); live.completedNodes = Array.from(completed);
    }
    live.nodeId = nodeId == null ? null : String(nodeId);
    live.nodeTitle = nodeId == null ? "正在整理输出" : (job.summary?.node_titles?.[String(nodeId)] || `节点 ${nodeId}`);
    live.nodePercent = nodeId == null ? 100 : 0;
    if (!live.startedAt) live.startedAt = Date.now() / 1000;
    updateLiveOverall(job);
  }

  function handleComfyEvent(jobId, message) {
    const job = activeLocalJob(jobId); if (!job || !message?.type) return;
    const data = message.data || {};
    if (data.prompt_id && String(data.prompt_id) !== String(job.prompt_id)) return;
    const live = job.live || (job.live = {});
    if (message.type === "execution_start") {
      live.startedAt = live.startedAt || Date.now() / 1000;
      job.state = "running";
    } else if (message.type === "executing") {
      setLiveNode(job, data.node);
      if (data.node != null) job.state = "running";
    } else if (message.type === "progress") {
      if (data.node != null && String(data.node) !== live.nodeId) setLiveNode(job, data.node);
      live.nodePercent = data.max ? clamp(Number(data.value) / Number(data.max) * 100, 0, 100) : 0;
      live.step = Number(data.value || 0); live.maxSteps = Number(data.max || 0);
      updateLiveOverall(job);
    } else if (message.type === "progress_state" && data.nodes) {
      const entries = Object.entries(data.nodes);
      const running = entries.find(([, value]) => value?.state === "running") || entries.find(([, value]) => Number(value?.value) < Number(value?.max));
      if (running) {
        const [nodeId, value] = running; if (nodeId !== live.nodeId) setLiveNode(job, nodeId);
        live.nodePercent = value.max ? clamp(Number(value.value) / Number(value.max) * 100, 0, 100) : 0;
        live.step = Number(value.value || 0); live.maxSteps = Number(value.max || 0); updateLiveOverall(job);
      }
    } else if (message.type === "executed") {
      const completed = new Set(live.completedNodes || []); if (data.node != null) completed.add(String(data.node)); live.completedNodes = Array.from(completed); updateLiveOverall(job);
    } else if (message.type === "execution_cached") {
      const completed = new Set(live.completedNodes || []); for (const node of data.nodes || []) completed.add(String(node)); live.completedNodes = Array.from(completed); updateLiveOverall(job);
    } else if (message.type === "status") {
      live.queueRemaining = data.status?.exec_info?.queue_remaining;
    } else if (message.type === "execution_error") {
      job.error = data.exception_message || data.exception_type || "ComfyUI 执行失败";
      live.fullError = data;
    }
    updateJobControls(job); scheduleTaskRender();
  }

  function connectJobSocket(jobId, tracker) {
    const job = activeLocalJob(jobId);
    if (!job?.client_id || tracker.finished) return;
    let socket;
    try { socket = new WebSocket(comfyWebSocketUrl(job.client_id)); }
    catch (error) { console.warn("[H3 Studio] websocket", error); return; }
    tracker.socket = socket;
    socket.binaryType = "blob";
    socket.onmessage = (event) => {
      if (typeof event.data === "string") {
        try { handleComfyEvent(jobId, JSON.parse(event.data)); } catch (error) { console.warn("[H3 Studio] event", error); }
      } else if (event.data instanceof Blob && event.data.size > 8) {
        const current = activeLocalJob(jobId); if (!current) return;
        const live = current.live || (current.live = {});
        if (live.previewUrl) URL.revokeObjectURL(live.previewUrl);
        live.previewUrl = URL.createObjectURL(event.data.slice(8));
        scheduleTaskRender();
      }
    };
    socket.onclose = () => {
      if (!tracker.finished && ["queued", "running"].includes(activeLocalJob(jobId)?.state)) {
        clearTimeout(tracker.reconnectTimer);
        tracker.reconnectTimer = setTimeout(() => connectJobSocket(jobId, tracker), 2500);
      }
    };
    socket.onerror = () => { /* HTTP polling remains as fallback */ };
  }

  function finishJobTracking(job, tracker) {
    tracker.finished = true;
    clearInterval(tracker.timer); clearTimeout(tracker.reconnectTimer);
    try { tracker.socket?.close(); } catch (_) { /* already closed */ }
    state.trackers.delete(job.id);
    updateJobControls(job); scheduleTaskRender();
    if (job.live?.notified) return;
    job.live = { ...(job.live || {}), notified: true };
    if (job.state === "completed") {
      syncGeneratedAssets(job); state.result = job; renderStage(); renderSidebar();
      toast("H3 视频生成完成，已放入素材库", "success", 7000);
    } else toast(job.error || "任务未完成", "error", 8000);
  }

  function startJobTracking(jobOrId) {
    const job = typeof jobOrId === "string" ? activeLocalJob(jobOrId) : jobOrId;
    if (!job || state.trackers.has(job.id) || !["queued", "running"].includes(job.state)) return;
    const tracker = { timer: null, socket: null, reconnectTimer: null, finished: false };
    state.trackers.set(job.id, tracker);
    const poll = async () => {
      try {
        const merged = mergeJob(await request(`/jobs/${job.id}`));
        updateJobControls(merged); scheduleTaskRender();
        if (["completed", "failed", "cancelled"].includes(merged.state)) finishJobTracking(merged, tracker);
      } catch (error) { console.warn("[H3 Studio] job poll", error); }
    };
    connectJobSocket(job.id, tracker);
    poll(); tracker.timer = setInterval(poll, 1500);
  }

  function updateJobControls(job = null) {
    const current = job || state.jobs.find((item) => item.id === state.activeJobId);
    const cancel = $("[data-action='cancel-job']", root());
    if (cancel) cancel.disabled = !current || !["queued", "running"].includes(current.state);
  }

  async function cancelActiveJob() {
    if (!state.activeJobId) return;
    cancelJob(state.activeJobId);
  }

  async function cancelJob(id) {
    try {
      const detailOpen = !!$(".h3s-job-detail-modal", root());
      const job = mergeJob(await request(`/jobs/${id}/cancel`, { method: "POST" }));
      const tracker = state.trackers.get(id); if (tracker) finishJobTracking(job, tracker);
      renderTaskSurfaces(); updateJobControls(job);
      if (detailOpen) openJobDetails(id);
    } catch (error) { toast(error.message, "error"); }
  }

  async function clearJobs(scope) {
    try {
      const result = await request("/jobs/clear", { method: "POST", body: { scope } });
      toast(`已清理 ${result.cleared || 0} 个${scope === "queue" ? "排队" : "完成"}任务`, "success");
      await refreshJobs();
    } catch (error) { toast(error.message, "error"); }
  }

  async function refreshJobs() {
    try {
      const response = await request("/jobs?limit=100");
      const previous = new Map(state.jobs.map((job) => [job.id, job]));
      state.jobs = (response.jobs || []).map((job) => {
        const old = previous.get(job.id);
        const live = { ...(old?.live || {}), ...(job.progress || {}) };
        if (job.preview_url) live.previewUrl = job.preview_url;
        return { ...job, live };
      });
      syncGeneratedAssets(state.jobs);
      state.jobs.filter((job) => ["queued", "running"].includes(job.state)).forEach(startJobTracking);
      renderTaskSurfaces(); updateBackendUi();
    }
    catch (error) { toast(error.message, "error"); }
  }

  async function selectJob(id) {
    state.activeJobId = id; renderTaskSurfaces();
    try {
      const job = mergeJob(await request(`/jobs/${id}`));
      if (job.state === "completed") { state.result = job; renderStage(); openMediaViewer(preferredOutput(job), job.summary); }
      else if (["queued", "running"].includes(job.state)) startJobTracking(job);
      else toast(job.error || "任务未完成", "warning");
    }
    catch (error) { toast(error.message, "error"); }
  }

  function reuseJobParameters(id) {
    const job = state.jobs.find((item) => item.id === id) || (state.result?.id === id ? state.result : null);
    const archived = state.assets.find((asset) => asset.jobId === id && asset.summary?.reproducible);
    const saved = job?.summary?.reproducible || archived?.summary?.reproducible;
    if (!saved || typeof saved !== "object") return toast("这个旧任务没有可复用的完整参数", "warning");
    if (MODE_META[saved.mode]) state.mode = saved.mode;
    const keys = [
      "model", "text_encoder", "video_vae", "audio_vae", "weight_dtype", "clip_device", "prompt",
      "width", "height", "aspect_ratio", "megapixels", "rounding_multiple", "frames", "steps", "seed",
      "sampler", "scheduler", "shift_video", "shift_audio", "denoise", "ref_image_size",
    ];
    for (const key of keys) if (saved[key] !== undefined) state.params[key] = saved[key];
    state.params.random_seed = false;
    state.params.resolution_linked = false;
    const output = saved.output || {};
    if (output.filename_prefix !== undefined) state.params.filename_prefix = output.filename_prefix;
    if (output.format !== undefined) state.params.output_format = output.format;
    if (output.codec !== undefined) state.params.output_codec = output.codec;
    if (output.crf !== undefined) state.params.output_crf = output.crf;
    if (output.fps !== undefined) state.params.output_fps = output.fps;
    if (output.bit_depth !== undefined) state.params.bit_depth = output.bit_depth;
    state.loras = Array.isArray(saved.loras) ? JSON.parse(JSON.stringify(saved.loras)) : [];

    const inputAsset = (file) => state.assets.find((asset) => asset.file === file && asset.source !== "generated") || null;
    const first = saved.first_frame ? inputAsset(saved.first_frame) : null;
    const last = saved.last_frame ? inputAsset(saved.last_frame) : null;
    state.firstFrame = first?.id || null;
    state.lastFrame = last?.id || null;
    state.frameCrops.first = saved.first_frame_crop ? { ...saved.first_frame_crop } : null;
    state.frameCrops.last = saved.last_frame_crop ? { ...saved.last_frame_crop } : null;
    state.references = { image: [], video: [], audio: [] };
    state.refVideoAudio = [];
    for (const ref of saved.references || []) {
      const asset = inputAsset(ref.file);
      if (!asset || !state.references[ref.kind]) continue;
      state.references[ref.kind].push(asset.id);
      if (ref.kind === "video") state.refVideoAudio.push(ref.include_audio !== false);
    }
    const missingInputs = (saved.first_frame && !first) || (saved.last_frame && !last)
      || (saved.references || []).some((ref) => !inputAsset(ref.file));
    state.result = null;
    state.inspectorTab = "project";
    closeModal();
    projectChanged();
    toast(missingInputs ? "参数与 Seed 已载入；部分原始素材需重新导入" : `已载入 Seed ${saved.seed} 和完整生成参数`, missingInputs ? "warning" : "success", 6500);
  }

  async function previewWorkflow() {
    try {
      const result = await request("/workflow/preview", { method: "POST", body: await generationPayload() });
      const blob = new Blob([JSON.stringify(result, null, 2)], { type: "application/json" });
      downloadBlob(blob, `h3_workflow_${Date.now()}.json`);
      if (result.summary.warnings?.length) toast(result.summary.warnings.join("；"), "warning", 9000); else toast("工作流 JSON 已导出", "success");
    } catch (error) { toast(`工作流检查失败：${error.message}`, "error", 8000); }
  }

  function exportProject() { downloadBlob(new Blob([JSON.stringify(projectPayload(), null, 2)], { type: "application/json" }), `h3_studio_project_${Date.now()}.json`); }
  function downloadBlob(blob, filename) { const url = URL.createObjectURL(blob); const link = document.createElement("a"); link.href = url; link.download = filename; document.body.appendChild(link); link.click(); link.remove(); URL.revokeObjectURL(url); }
  async function importProjectFile(event) {
    const file = event.target.files?.[0]; if (!file) return;
    try {
      const project = JSON.parse(await file.text());
      if (![1, 2].includes(project.version)) throw new Error("不支持的项目版本");
      localStorage.setItem(STORAGE_KEY, JSON.stringify(project));
      location.reload();
    } catch (error) { toast(`项目导入失败：${error.message}`, "error"); }
  }

  function applySettingsModeVisibility() {
    const modal = $(".h3s-settings-modal", root());
    if (!modal) return;
    const select = $("[data-setting='backend_mode']", modal);
    const mode = select?.value || state.config.backend_mode || "managed";
    const localSection = $("[data-role='local-backend-section']", modal);
    const cloudSection = $("[data-role='cloud-api-section']", modal);
    const isCloud = mode === "api";
    if (localSection) localSection.hidden = isCloud;
    if (cloudSection) cloudSection.hidden = !isCloud;
    for (const input of $$("[data-setting]", localSection || modal)) input.disabled = isCloud && !!localSection?.contains(input);
  }

  function openSettings() {
    const c = state.config;
    const discovered = state.backend.discovered_paths || [];
    const keySet = !!c.minimax_api_key_set;
    const layer = $("[data-role='modal-layer']", root());
    layer.innerHTML = `<div class="h3s-modal-backdrop" data-action="close-modal"></div><div class="h3s-modal h3s-settings-modal"><header><div><strong>后端连接与启动</strong><span>切换到工作台时可自动启动并连接</span></div><button data-action="close-modal">${icon("close")}</button></header><div class="h3s-settings-grid"><section><h3>连接方式</h3>
      ${field("后端模式", `<select data-setting="backend_mode" data-role="backend-mode"><option value="managed" ${c.backend_mode === "managed" || (!c.backend_mode && c.backend_mode !== "external" && c.backend_mode !== "api") ? "selected" : ""}>Forge 托管本地 ComfyUI</option><option value="external" ${c.backend_mode === "external" ? "selected" : ""}>连接已经运行的 ComfyUI</option><option value="api" ${c.backend_mode === "api" ? "selected" : ""}>云端 API（MiniMax H3）</option></select>`)}
      <div data-role="cloud-api-section" hidden>
        ${field("API Base URL", `<input data-setting="minimax_api_base" value="${esc(c.minimax_api_base || "https://api.minimaxi.com")}" placeholder="https://api.minimaxi.com">`, "国内站默认 https://api.minimaxi.com；国际站可改为 https://api.minimax.io")}
        ${field("API Key", `<div class="h3s-api-key-row"><input type="password" data-setting="minimax_api_key" autocomplete="off" placeholder="${keySet ? "已配置（留空保持不变）" : "sk-api-..."}"><button class="h3s-row-danger" data-action="clear-api-key" ${keySet ? "" : "hidden"}>清除已保存 Key</button></div>`, "密钥仅保存在本机 data/config.json，不会写入项目文件")}
        <div class="h3s-settings-note"><i>i</i><span>云端模式由后端直接调用 MiniMax H3 接口，无需启动本地 ComfyUI。保存后即生效。</span></div>
      </div>
      <div data-role="local-backend-section">
      ${field("ComfyUI 地址", `<input data-setting="comfy_url" value="${esc(c.comfy_url || "http://127.0.0.1:8189")}">`)}
      ${field("ComfyUI / Portable 目录", `<input data-setting="comfy_path" list="h3s-comfy-paths" value="${esc(c.comfy_path || discovered[0] || "")}" placeholder="例如 D:\\ComfyUI_windows_portable"><datalist id="h3s-comfy-paths">${discovered.map((path) => `<option value="${esc(path)}"></option>`).join("")}</datalist>`, "托管模式需要；可选择包含 ComfyUI 子目录的 Portable 根目录")}
      ${field("Python 可执行文件（可留空）", `<input data-setting="python_executable" value="${esc(c.python_executable || "")}" placeholder="自动寻找 python_embeded 或 venv">`)}
      <div class="h3s-field-grid">${field("端口", `<input type="number" data-setting="port" value="${Number(c.port || 8189)}">`)}${field("启动超时（秒）", `<input type="number" data-setting="startup_timeout" value="${Number(c.startup_timeout || 180)}">`)}</div>
      ${field("额外启动参数", `<input data-setting="extra_args" value="${esc(c.extra_args || "")}">`)}
      <label class="h3s-toggle-line"><input type="checkbox" data-setting="auto_start_on_tab" ${c.auto_start_on_tab !== false ? "checked" : ""}><span><b>进入 H3 页签时自动启动</b><small>已运行时只检查连接，不会重复创建进程</small></span></label>
      </div>
      <div class="h3s-settings-actions"><button data-action="save-settings" class="h3s-primary-btn">保存设置</button><button data-action="start-backend">启动/连接</button><button data-action="stop-backend" class="danger">停止托管后端</button></div></section>
      <section><h3>后端状态</h3><div class="h3s-backend-summary" data-state="${esc(state.backend.ready ? "ready" : state.backend.state)}"><i></i><div><strong>${state.backend.ready ? "已连接" : state.backend.state === "starting" ? "正在启动" : "未连接"}</strong><span>${esc(state.backend.url || c.comfy_url || "")}</span></div></div><div class="h3s-log-head"><span>启动日志</span><button data-action="refresh-logs">${icon("refresh")}</button></div><pre data-role="backend-logs">点击刷新读取日志…</pre><div class="h3s-settings-foot"><button data-action="import-project">导入项目 JSON</button><button data-action="export-project">导出当前项目</button></div></section></div></div>`;
    applySettingsModeVisibility();
    refreshLogs();
  }

  async function saveSettings() {
    const modal = $(".h3s-settings-modal", root()); if (!modal) return;
    const payload = {};
    $$('[data-setting]', modal).forEach((input) => {
      if (input.disabled) return;
      const key = input.dataset.setting;
      const value = valueFromInput(input);
      // 已配置 Key 时留空表示保持不变，不覆盖后端已保存的密钥
      if (key === "minimax_api_key" && value === "" && state.config.minimax_api_key_set) return;
      payload[key] = value;
    });
    try {
      state.config = await request("/settings", { method: "POST", body: payload });
      toast("设置已保存", "success");
      const wasCloud = state.config.backend_mode === "api";
      closeModal();
      state.backend = await request("/backend/status");
      updateBackendUi();
      if (!wasCloud && state.backend.ready) await loadCatalog(true);
    } catch (error) { toast(error.message, "error", 7000); }
  }

  async function clearApiKey() {
    if (!window.confirm("确定要清除已保存的 MiniMax API Key 吗？此操作不可撤销。")) return;
    try {
      state.config = await request("/settings", { method: "POST", body: { clear_minimax_api_key: true } });
      toast("API Key 已清除", "success");
      openSettings();
    } catch (error) { toast(error.message, "error", 7000); }
  }

  async function stopBackend() {
    try { state.backend = await request("/backend/stop", { method: "POST" }); updateBackendUi(); toast("托管后端已停止", "success"); closeModal(); }
    catch (error) { toast(error.message, "error"); }
  }

  async function refreshLogs() {
    const pre = $("[data-role='backend-logs']", root()); if (!pre) return;
    try { const response = await request("/backend/logs?limit=180"); pre.textContent = response.lines?.join("\n") || "暂无启动日志"; pre.scrollTop = pre.scrollHeight; }
    catch (error) { pre.textContent = error.message; }
  }

  function closeModal() {
    const cropSaved = commitCropEditor(false);
    state.cropEditor = null;
    const layer = $("[data-role='modal-layer']", root()); if (layer) layer.innerHTML = "";
    if (cropSaved) { renderStage(); renderTimeline(); renderInspector(); }
  }

  const start = () => { if (typeof window.onUiLoaded === "function") window.onUiLoaded(mount); else if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", mount); else mount(); };
  start();
})();
