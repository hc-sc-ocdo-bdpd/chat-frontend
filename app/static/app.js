const state = {
  catalog: null,
  projects: [],
  conversations: [],
  activeConversationId: null,
  activeConversation: null,
  activeProjectFilter: "all",
  editingProjectId: null,
  pendingAttachments: [],
  busy: false,
  abortController: null,
};

const el = {
  sidebar: document.getElementById("sidebar"),
  sidebarToggle: document.getElementById("sidebar-toggle"),
  conversationList: document.getElementById("conversation-list"),
  conversationCount: document.getElementById("conversation-count"),
  newChat: document.getElementById("new-chat"),
  projectList: document.getElementById("project-list"),
  addProject: document.getElementById("add-project"),
  allChatsFilter: document.getElementById("all-chats-filter"),
  generalChatsFilter: document.getElementById("general-chats-filter"),
  activeProjectBadge: document.getElementById("active-project-badge"),
  messages: document.getElementById("messages"),
  model: document.getElementById("model-select"),
  reasoning: document.getElementById("reasoning-select"),
  verbosity: document.getElementById("verbosity-select"),
  codeInterpreter: document.getElementById("code-interpreter"),
  webSearch: document.getElementById("web-search"),
  webSearchOptions: document.getElementById("web-search-options"),
  webAllowedDomains: document.getElementById("web-allowed-domains"),
  webBlockedDomains: document.getElementById("web-blocked-domains"),
  maxOutputTokens: document.getElementById("max-output-tokens"),
  settingsToggle: document.getElementById("settings-toggle"),
  settingsPanel: document.getElementById("settings-panel"),
  projectContext: document.getElementById("project-context"),
  pendingFiles: document.getElementById("pending-files"),
  fileInput: document.getElementById("file-input"),
  messageInput: document.getElementById("message-input"),
  sendButton: document.getElementById("send-button"),
  composerNote: document.getElementById("composer-note"),
  toast: document.getElementById("toast"),

  projectModal: document.getElementById("project-modal"),
  projectModalTitle: document.getElementById("project-modal-title"),
  projectModalClose: document.getElementById("project-modal-close"),
  projectName: document.getElementById("project-name"),
  projectInstructions: document.getElementById("project-instructions"),
  projectModel: document.getElementById("project-model"),
  projectFileInput: document.getElementById("project-file-input"),
  projectFilesList: document.getElementById("project-files-list"),
  saveProject: document.getElementById("save-project"),
  cancelProject: document.getElementById("cancel-project"),
  deleteProject: document.getElementById("delete-project"),
};

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: {
      ...(options.body instanceof FormData
        ? {}
        : { "Content-Type": "application/json" }),
      ...(options.headers || {}),
    },
    ...options,
  });

  if (!response.ok) {
    let message = `${response.status} ${response.statusText}`;
    try {
      const data = await response.json();
      message = data.detail || message;
    } catch (_) {}
    throw new Error(message);
  }

  if (response.status === 204) return null;
  return response.json();
}

async function streamApi(path, payload, onEvent, signal) {
  const response = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
    signal,
  });

  if (!response.ok) {
    let message = `${response.status} ${response.statusText}`;
    try {
      const data = await response.json();
      message = data.detail || message;
    } catch (_) {}
    throw new Error(message);
  }

  if (!response.body) {
    throw new Error("This browser did not provide a streaming response body");
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();
    buffer += decoder.decode(value || new Uint8Array(), { stream: !done });

    const blocks = buffer.split(/\r?\n\r?\n/);
    buffer = blocks.pop() || "";

    for (const block of blocks) {
      const data = block
        .split(/\r?\n/)
        .filter((line) => line.startsWith("data:"))
        .map((line) => line.slice(5).trimStart())
        .join("\n");

      if (!data) continue;
      await onEvent(JSON.parse(data));
    }

    if (done) break;
  }

  if (buffer.trim()) {
    const data = buffer
      .split(/\r?\n/)
      .filter((line) => line.startsWith("data:"))
      .map((line) => line.slice(5).trimStart())
      .join("\n");
    if (data) await onEvent(JSON.parse(data));
  }
}

function showToast(message) {
  el.toast.textContent = message;
  el.toast.classList.remove("hidden");
  clearTimeout(showToast.timer);
  showToast.timer = setTimeout(() => el.toast.classList.add("hidden"), 4500);
}

function currentModel() {
  return state.catalog.models.find((item) => item.id === el.model.value);
}

function projectById(projectId) {
  return state.projects.find((project) => project.id === projectId) || null;
}

function fillSelect(select, values, selectedValue, mapLabel = (value) => value) {
  select.innerHTML = "";
  values.forEach((value) => {
    const option = document.createElement("option");
    option.value = typeof value === "string" ? value : value.id;
    option.textContent = mapLabel(value);
    if (option.value === selectedValue) option.selected = true;
    select.appendChild(option);
  });
}

function refreshModelSelect(selectedModel) {
  const fallback = state.catalog.models.some(
    (model) => model.id === selectedModel
  )
    ? selectedModel
    : state.catalog.default_model;
  fillSelect(
    el.model,
    state.catalog.models,
    fallback,
    (model) => model.label
  );
  refreshModelControls();
}

function refreshModelControls() {
  const model = currentModel();
  if (!model) return;

  const previousReasoning = el.reasoning.value;
  const previousVerbosity = el.verbosity.value;

  fillSelect(
    el.reasoning,
    model.reasoning_efforts,
    model.reasoning_efforts.includes(previousReasoning)
      ? previousReasoning
      : model.default_reasoning_effort,
    (value) =>
      value === "auto"
        ? "Auto"
        : value[0].toUpperCase() + value.slice(1)
  );
  fillSelect(
    el.verbosity,
    model.verbosity_options,
    model.verbosity_options.includes(previousVerbosity)
      ? previousVerbosity
      : model.default_verbosity,
    (value) => value[0].toUpperCase() + value.slice(1)
  );

  el.codeInterpreter.checked = model.default_code_interpreter;
  el.codeInterpreter.disabled = !model.supports_code_interpreter;
  el.webSearch.checked = Boolean(model.default_web_search);
  el.webSearch.disabled = !model.supports_web_search;
  el.maxOutputTokens.value = model.default_max_output_tokens;
  updateToolControls();
}

function updateToolControls() {
  const webEnabled = el.webSearch.checked && !el.webSearch.disabled;
  el.webSearchOptions.classList.toggle("disabled", !webEnabled);
  el.webAllowedDomains.disabled = !webEnabled;
  el.webBlockedDomains.disabled = !webEnabled;

  const activeTools = [];
  if (el.codeInterpreter.checked && !el.codeInterpreter.disabled) {
    activeTools.push("Code");
  }
  if (webEnabled) {
    activeTools.push("Thorough web research");
  }

  el.composerNote.textContent = activeTools.length
    ? `${activeTools.join(" · ")} enabled for this message.`
    : "No tools enabled for this message.";
}

function updateWebSearchControls() {
  updateToolControls();
}

function parseDomainList(value) {
  const seen = new Set();
  return value
    .split(/[\s,]+/)
    .map((item) => item.trim().toLowerCase())
    .map((item) => item.replace(/^https?:\/\//, "").split("/")[0])
    .filter((item) => item && !seen.has(item) && seen.add(item))
    .slice(0, 100);
}

function applyProjectDefaults(project) {
  if (!project) return;
  refreshModelSelect(project.default_model_id);
}

function filteredConversations() {
  if (state.activeProjectFilter === "all") return state.conversations;
  if (state.activeProjectFilter === "general") {
    return state.conversations.filter((conversation) => !conversation.project_id);
  }
  return state.conversations.filter(
    (conversation) => conversation.project_id === state.activeProjectFilter
  );
}

function renderProjects() {
  el.allChatsFilter.classList.toggle(
    "active", state.activeProjectFilter === "all"
  );
  el.generalChatsFilter.classList.toggle(
    "active", state.activeProjectFilter === "general"
  );

  el.projectList.innerHTML = "";

  state.projects.forEach((project) => {
    const row = document.createElement("div");
    row.className = "project-row";
    if (state.activeProjectFilter === project.id) row.classList.add("active");

    const select = document.createElement("button");
    select.className = "project-select";
    select.type = "button";

    const name = document.createElement("span");
    name.className = "project-name";
    name.textContent = project.name;

    const count = document.createElement("span");
    count.className = "project-count";
    count.textContent = String(project.conversation_count || 0);

    select.append(name, count);
    select.addEventListener("click", () => selectProjectFilter(project.id));

    const settings = document.createElement("button");
    settings.className = "mini-button project-settings";
    settings.type = "button";
    settings.title = "Project settings";
    settings.textContent = "•••";
    settings.addEventListener("click", (event) => {
      event.stopPropagation();
      openProjectModal(project.id);
    });

    row.append(select, settings);
    el.projectList.appendChild(row);
  });
}

function renderConversations() {
  const conversations = filteredConversations();
  el.conversationList.innerHTML = "";
  el.conversationCount.textContent = String(conversations.length);

  conversations.forEach((conversation) => {
    const row = document.createElement("div");
    row.className = "conversation-item";
    if (conversation.id === state.activeConversationId) {
      row.classList.add("active");
    }

    const titleWrap = document.createElement("div");
    titleWrap.className = "conversation-title-wrap";

    const title = document.createElement("div");
    title.className = "conversation-title";
    title.textContent = conversation.title;
    titleWrap.appendChild(title);

    if (state.activeProjectFilter === "all" && conversation.project_name) {
      const projectLabel = document.createElement("div");
      projectLabel.className = "conversation-project-label";
      projectLabel.textContent = conversation.project_name;
      titleWrap.appendChild(projectLabel);
    }

    const actions = document.createElement("div");
    actions.className = "conversation-actions";

    const move = document.createElement("button");
    move.className = "mini-button";
    move.title = "Move chat";
    move.textContent = "↪";
    move.addEventListener("click", async (event) => {
      event.stopPropagation();
      await moveConversation(conversation);
    });

    const rename = document.createElement("button");
    rename.className = "mini-button";
    rename.title = "Rename";
    rename.textContent = "✎";
    rename.addEventListener("click", async (event) => {
      event.stopPropagation();
      const next = prompt("Conversation title", conversation.title);
      if (!next?.trim()) return;
      await api(`/api/conversations/${conversation.id}`, {
        method: "PATCH",
        body: JSON.stringify({ title: next.trim() }),
      });
      await loadConversations();
    });

    const remove = document.createElement("button");
    remove.className = "mini-button";
    remove.title = "Delete";
    remove.textContent = "×";
    remove.addEventListener("click", async (event) => {
      event.stopPropagation();
      if (!confirm(`Delete "${conversation.title}"?`)) return;
      await api(`/api/conversations/${conversation.id}`, { method: "DELETE" });
      if (state.activeConversationId === conversation.id) {
        state.activeConversationId = null;
        state.activeConversation = null;
        state.pendingAttachments = [];
      }
      await loadConversations();
      const next = filteredConversations()[0];
      if (next) {
        await openConversation(next.id);
      } else {
        renderEmpty();
      }
    });

    actions.append(move, rename, remove);
    row.append(titleWrap, actions);
    row.addEventListener("click", () => openConversation(conversation.id));
    el.conversationList.appendChild(row);
  });
}

function renderEmpty() {
  const project =
    state.activeProjectFilter !== "all" &&
    state.activeProjectFilter !== "general"
      ? projectById(state.activeProjectFilter)
      : null;

  el.messages.innerHTML = `
    <div class="welcome">
      <h1>${escapeHtml(project?.name || state.catalog?.title || "Foundry Chat")}</h1>
      <p>${
        project
          ? "Start a chat using this project's instructions and persistent files."
          : "Select a model, attach files, and start a conversation."
      }</p>
    </div>
  `;
  renderProjectContext();
}

function escapeHtml(value) {
  const node = document.createElement("div");
  node.textContent = value ?? "";
  return node.innerHTML;
}


let mathTypesetTimer = null;
const pendingMathContainers = new Set();

function clearMathTypeset(container) {
  if (!container || !window.MathJax?.typesetClear) return;
  try {
    window.MathJax.typesetClear([container]);
  } catch (error) {
    console.debug("MathJax clear skipped", error);
  }
}

function scheduleMathTypeset(container) {
  if (!container) return;
  pendingMathContainers.add(container);
  clearTimeout(mathTypesetTimer);
  mathTypesetTimer = setTimeout(async () => {
    const containers = Array.from(pendingMathContainers).filter((item) =>
      document.body.contains(item)
    );
    pendingMathContainers.clear();
    if (!containers.length) return;

    try {
      if (!window.MathJax?.startup?.promise) return;
      await window.MathJax.startup.promise;
      await window.MathJax.typesetPromise(containers);
    } catch (error) {
      console.warn("Math rendering failed", error);
    }
  }, 80);
}

function appendPlainText(container, text) {
  if (!text) return;
  container.appendChild(document.createTextNode(text));
}

function normalizedBasename(value) {
  let decoded = value;
  try {
    decoded = decodeURIComponent(value);
  } catch (_) {}

  const normalized = decoded.replace(/\\/g, "/");
  return normalized.split("/").filter(Boolean).pop()?.toLowerCase() || "";
}

function resolveMarkdownLink(target, generatedFiles) {
  const trimmed = target.trim();

  if (/^sandbox:\/+/i.test(trimmed)) {
    const targetName = normalizedBasename(trimmed);
    const generated = generatedFiles.find(
      (file) => normalizedBasename(file.filename) === targetName
    );

    if (!generated) return null;

    return {
      href: generated.url,
      isDownload: true,
      filename: generated.filename,
    };
  }

  if (/^https?:\/\//i.test(trimmed)) {
    return {
      href: trimmed,
      isDownload: false,
      filename: null,
    };
  }

  return null;
}


function applyCitationMarkers(text, citations = []) {
  if (!Array.isArray(citations) || !citations.length) return text;

  const markers = [];
  const seen = new Set();
  citations.forEach((citation) => {
    const endIndex = Number(citation.end_index);
    const sourceIndex = Number(citation.source_index);
    const url = String(citation.url || "");
    if (
      !Number.isInteger(endIndex) ||
      endIndex < 0 ||
      endIndex > text.length ||
      !Number.isInteger(sourceIndex) ||
      sourceIndex < 1 ||
      !/^https?:\/\//i.test(url)
    ) {
      return;
    }

    const key = `${endIndex}:${sourceIndex}`;
    if (seen.has(key)) return;
    seen.add(key);
    markers.push({ endIndex, sourceIndex, url });
  });

  markers.sort((a, b) => b.endIndex - a.endIndex || b.sourceIndex - a.sourceIndex);
  let result = text;
  markers.forEach((marker) => {
    const safeUrl = encodeURI(marker.url)
      .replace(/\(/g, "%28")
      .replace(/\)/g, "%29");
    const token = ` [${marker.sourceIndex}](${safeUrl})`;
    result = result.slice(0, marker.endIndex) + token + result.slice(marker.endIndex);
  });
  return result;
}

function appendInlineMarkdown(
  container,
  text,
  generatedFiles,
  { allowLinks = true } = {}
) {
  if (!text) return;

  const tokenPattern =
    /(\\\([^\n]*?\\\)|\$\$[^\n]*?\$\$|\\\[[^\n]*?\\\]|\$[^$\n]+?\$|`[^`\n]+`|\[([^\]\n]+)\]\(([^)\n]+)\)|\*\*([^*\n]+)\*\*|\*([^*\n]+)\*)/g;

  let cursor = 0;
  let match;

  while ((match = tokenPattern.exec(text)) !== null) {
    appendPlainText(container, text.slice(cursor, match.index));
    const token = match[0];

    if (
      token.startsWith("\\(") ||
      token.startsWith("\\[") ||
      token.startsWith("$")
    ) {
      appendPlainText(container, token);
    } else if (token.startsWith("`")) {
      const inlineCode = document.createElement("code");
      inlineCode.className = "inline-code";
      inlineCode.textContent = token.slice(1, -1);
      container.appendChild(inlineCode);
    } else if (token.startsWith("[") && allowLinks) {
      const labelText = match[2];
      const target = match[3];
      const resolved = resolveMarkdownLink(target, generatedFiles);

      if (resolved) {
        const link = document.createElement("a");
        const citationLink = /^\d+$/.test(labelText);
        link.className = resolved.isDownload
          ? "message-link message-download-link"
          : citationLink
            ? "message-link message-citation-link"
            : "message-link";
        link.href = resolved.href;

        if (resolved.isDownload) {
          link.download = resolved.filename || "";
        } else {
          link.target = "_blank";
          link.rel = "noopener noreferrer";
        }

        appendInlineMarkdown(link, labelText, generatedFiles, {
          allowLinks: false,
        });
        container.appendChild(link);
      } else {
        appendPlainText(container, token);
      }
    } else if (token.startsWith("**")) {
      const strong = document.createElement("strong");
      appendInlineMarkdown(strong, match[4], generatedFiles, {
        allowLinks,
      });
      container.appendChild(strong);
    } else if (token.startsWith("*")) {
      const emphasis = document.createElement("em");
      appendInlineMarkdown(emphasis, match[5], generatedFiles, {
        allowLinks,
      });
      container.appendChild(emphasis);
    } else {
      appendPlainText(container, token);
    }

    cursor = tokenPattern.lastIndex;
  }

  appendPlainText(container, text.slice(cursor));
}

function createCodeBlock(language, codeText) {
  const wrapper = document.createElement("div");
  wrapper.className = "code-block";

  const header = document.createElement("div");
  header.className = "code-block-header";

  const label = document.createElement("span");
  label.className = "code-language";
  label.textContent = language || "code";

  const copyButton = document.createElement("button");
  copyButton.className = "code-copy-button";
  copyButton.type = "button";
  copyButton.textContent = "Copy";
  copyButton.addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText(codeText);
      copyButton.textContent = "Copied";
      setTimeout(() => {
        copyButton.textContent = "Copy";
      }, 1400);
    } catch (_) {
      showToast("Could not copy code");
    }
  });

  const pre = document.createElement("pre");
  const code = document.createElement("code");
  code.className = `language-${(language || "code").toLowerCase()}`;
  code.textContent = codeText;
  pre.appendChild(code);

  header.append(label, copyButton);
  wrapper.append(header, pre);
  return wrapper;
}

function renderMarkdownBlocks(container, text, generatedFiles) {
  const lines = text.replace(/\r\n?/g, "\n").split("\n");
  let paragraphLines = [];
  let activeList = null;
  let activeListType = null;
  let quoteLines = [];

  function flushParagraph() {
    if (!paragraphLines.length) return;

    const paragraph = document.createElement("p");
    paragraph.className = "markdown-paragraph";

    paragraphLines.forEach((line, index) => {
      if (index > 0) paragraph.appendChild(document.createElement("br"));
      appendInlineMarkdown(paragraph, line, generatedFiles);
    });

    container.appendChild(paragraph);
    paragraphLines = [];
  }

  function flushList() {
    if (!activeList) return;
    container.appendChild(activeList);
    activeList = null;
    activeListType = null;
  }

  function flushQuote() {
    if (!quoteLines.length) return;

    const quote = document.createElement("blockquote");
    quote.className = "markdown-blockquote";

    quoteLines.forEach((line, index) => {
      if (index > 0) quote.appendChild(document.createElement("br"));
      appendInlineMarkdown(quote, line, generatedFiles);
    });

    container.appendChild(quote);
    quoteLines = [];
  }

  function flushAll() {
    flushParagraph();
    flushList();
    flushQuote();
  }

  for (let lineIndex = 0; lineIndex < lines.length; lineIndex += 1) {
    const rawLine = lines[lineIndex];
    const line = rawLine.replace(/\s+$/, "");
    const trimmedLine = line.trim();

    if (trimmedLine === "\\[" || trimmedLine === "$$") {
      const closingDelimiter = trimmedLine === "\\[" ? "\\]" : "$$";
      const mathLines = [];
      let closingIndex = lineIndex + 1;

      while (
        closingIndex < lines.length &&
        lines[closingIndex].trim() !== closingDelimiter
      ) {
        mathLines.push(lines[closingIndex]);
        closingIndex += 1;
      }

      if (closingIndex < lines.length) {
        flushAll();
        const mathBlock = document.createElement("div");
        mathBlock.className = "markdown-math-block";
        mathBlock.textContent =
          `${trimmedLine}\n${mathLines.join("\n")}\n${closingDelimiter}`;
        container.appendChild(mathBlock);
        lineIndex = closingIndex;
        continue;
      }
    }

    if (!trimmedLine) {
      flushAll();
      continue;
    }

    const headingMatch = /^(#{1,6})\s+(.+)$/.exec(line);
    if (headingMatch) {
      flushAll();
      const level = Math.min(headingMatch[1].length, 6);
      const heading = document.createElement(`h${level}`);
      heading.className = `markdown-heading markdown-heading-${level}`;
      appendInlineMarkdown(heading, headingMatch[2], generatedFiles);
      container.appendChild(heading);
      continue;
    }

    if (/^\s*((-\s*){3,}|(\*\s*){3,}|(_\s*){3,})$/.test(line)) {
      flushAll();
      const rule = document.createElement("hr");
      rule.className = "markdown-rule";
      container.appendChild(rule);
      continue;
    }

    const unorderedMatch = /^\s*[-+*]\s+(.+)$/.exec(line);
    if (unorderedMatch) {
      flushParagraph();
      flushQuote();

      if (!activeList || activeListType !== "ul") {
        flushList();
        activeList = document.createElement("ul");
        activeList.className = "markdown-list";
        activeListType = "ul";
      }

      const item = document.createElement("li");
      appendInlineMarkdown(item, unorderedMatch[1], generatedFiles);
      activeList.appendChild(item);
      continue;
    }

    const orderedMatch = /^\s*\d+[.)]\s+(.+)$/.exec(line);
    if (orderedMatch) {
      flushParagraph();
      flushQuote();

      if (!activeList || activeListType !== "ol") {
        flushList();
        activeList = document.createElement("ol");
        activeList.className = "markdown-list";
        activeListType = "ol";
      }

      const item = document.createElement("li");
      appendInlineMarkdown(item, orderedMatch[1], generatedFiles);
      activeList.appendChild(item);
      continue;
    }

    const quoteMatch = /^\s*>\s?(.*)$/.exec(line);
    if (quoteMatch) {
      flushParagraph();
      flushList();
      quoteLines.push(quoteMatch[1]);
      continue;
    }

    flushList();
    flushQuote();
    paragraphLines.push(line);
  }

  flushAll();
}

function renderCodeAwareContent(container, text, generatedFiles = []) {
  container.classList.add("markdown-rendered");

  const fencePattern = /```([A-Za-z0-9_+#.\-]*)\r?\n([\s\S]*?)```/g;
  let cursor = 0;
  let match;

  while ((match = fencePattern.exec(text)) !== null) {
    renderMarkdownBlocks(
      container,
      text.slice(cursor, match.index),
      generatedFiles
    );

    const language = match[1] || "code";
    const codeText = match[2].replace(/\r?\n$/, "");
    container.appendChild(createCodeBlock(language, codeText));
    cursor = fencePattern.lastIndex;
  }

  renderMarkdownBlocks(container, text.slice(cursor), generatedFiles);
  scheduleMathTypeset(container);
}

function formatDuration(seconds) {
  const value = Number(seconds);
  if (!Number.isFinite(value) || value < 0) return "";
  if (value < 10) return `${value.toFixed(1)}s`;
  return `${Math.round(value)}s`;
}

function reasoningPanelElement(metadata = {}, streaming = false) {
  const summaryText = metadata.reasoning_summary || "";
  const activities = metadata.activities || [];
  const hasDetails = Boolean(summaryText || activities.length || streaming);
  if (!hasDetails) return null;

  const details = document.createElement("details");
  details.className = "reasoning-panel";
  details.open = streaming;

  const header = document.createElement("summary");
  header.className = "reasoning-header";

  const spinner = document.createElement("span");
  spinner.className = "reasoning-spinner";
  spinner.textContent = "✦";

  const label = document.createElement("span");
  label.className = "reasoning-label";
  const duration = formatDuration(metadata.duration_seconds);
  label.textContent = streaming
    ? "Thinking"
    : duration
      ? `Reasoned for ${duration}`
      : "Reasoning summary";

  header.append(spinner, label);

  const body = document.createElement("div");
  body.className = "reasoning-body";

  const summary = document.createElement("div");
  summary.className = "reasoning-summary";
  summary.textContent = summaryText;
  if (!summaryText) summary.classList.add("hidden");

  const activityList = document.createElement("div");
  activityList.className = "reasoning-activities";
  activities.forEach((activity) => {
    const item = document.createElement("div");
    item.className = "reasoning-activity";
    item.textContent = activity;
    activityList.appendChild(item);
  });

  body.append(summary, activityList);
  details.append(header, body);

  return {
    element: details,
    label,
    spinner,
    summary,
    activityList,
  };
}

function addStreamingAssistant() {
  const row = document.createElement("div");
  row.className = "message-row assistant streaming-message";
  row.dataset.streaming = "true";

  const inner = document.createElement("div");
  inner.className = "message-inner streaming-message-inner";

  const panel = reasoningPanelElement({}, true);
  const content = document.createElement("div");
  content.className = "message-content streaming-answer";

  inner.append(panel.element, content);
  row.appendChild(inner);
  el.messages.appendChild(row);

  const state = {
    row,
    panel,
    content,
    answerText: "",
    reasoningText: "",
    activities: new Set(),
    startedAt: performance.now(),
    renderQueued: false,
    statusLabel: "Thinking",
  };

  state.timer = setInterval(() => {
    const seconds = (performance.now() - state.startedAt) / 1000;
    state.panel.label.textContent =
      `${state.statusLabel} · ${formatDuration(seconds)}`;
  }, 100);

  scrollToBottom();
  return state;
}

function scheduleStreamingAnswerRender(streaming) {
  if (streaming.renderQueued) return;
  streaming.renderQueued = true;
  requestAnimationFrame(() => {
    streaming.renderQueued = false;
    clearMathTypeset(streaming.content);
    streaming.content.innerHTML = "";
    renderCodeAwareContent(streaming.content, streaming.answerText, []);
    scrollToBottom();
  });
}

function updateStreamingStatus(streaming, label) {
  streaming.statusLabel = label || "Thinking";
  const seconds = (performance.now() - streaming.startedAt) / 1000;
  streaming.panel.label.textContent =
    `${streaming.statusLabel} · ${formatDuration(seconds)}`;
}

function appendStreamingReasoning(streaming, delta) {
  streaming.reasoningText += delta;
  streaming.panel.summary.textContent = streaming.reasoningText;
  streaming.panel.summary.classList.remove("hidden");
  scrollToBottom();
}

function setStreamingReasoning(streaming, text) {
  streaming.reasoningText = text || "";
  streaming.panel.summary.textContent = streaming.reasoningText;
  streaming.panel.summary.classList.toggle("hidden", !streaming.reasoningText);
  scrollToBottom();
}

function addStreamingActivity(streaming, label) {
  if (!label || streaming.activities.has(label)) return;
  streaming.activities.add(label);

  const item = document.createElement("div");
  item.className = "reasoning-activity";
  item.textContent = label;
  streaming.panel.activityList.appendChild(item);
  scrollToBottom();
}

function finishStreamingAssistant(streaming) {
  clearInterval(streaming.timer);
  streaming.row.remove();
}


function safeHostname(url) {
  try {
    return new URL(url).hostname.replace(/^www\./, "");
  } catch (_) {
    return url;
  }
}

function webResearchElement(metadata = {}) {
  const research = metadata.web_research || {};
  const sources = Array.isArray(research.sources) ? research.sources : [];
  const actions = Array.isArray(research.actions) ? research.actions : [];
  if (!research.used && !sources.length && !actions.length) return null;

  const details = document.createElement("details");
  details.className = "web-sources-panel";

  const summary = document.createElement("summary");
  summary.className = "web-sources-summary";

  const icon = document.createElement("span");
  icon.className = "web-sources-icon";
  icon.textContent = "◎";

  const label = document.createElement("span");
  const depth = "Web research";
  label.textContent = sources.length
    ? `${depth} · ${sources.length} source${sources.length === 1 ? "" : "s"}`
    : depth;

  summary.append(icon, label);

  const body = document.createElement("div");
  body.className = "web-sources-body";

  if (sources.length) {
    const sourceList = document.createElement("div");
    sourceList.className = "web-source-list";

    sources.forEach((source, position) => {
      const link = document.createElement("a");
      link.className = "web-source-card";
      link.href = source.url;
      link.target = "_blank";
      link.rel = "noopener noreferrer";

      const number = document.createElement("span");
      number.className = "web-source-number";
      number.textContent = String(source.index || position + 1);

      const text = document.createElement("span");
      text.className = "web-source-text";

      const title = document.createElement("span");
      title.className = "web-source-title";
      title.textContent = source.title || safeHostname(source.url);

      const domain = document.createElement("span");
      domain.className = "web-source-domain";
      domain.textContent = safeHostname(source.url);

      text.append(title, domain);

      if (source.snippet) {
        const snippet = document.createElement("span");
        snippet.className = "web-source-snippet";
        snippet.textContent = source.snippet;
        text.appendChild(snippet);
      }

      link.append(number, text);
      sourceList.appendChild(link);
    });

    body.appendChild(sourceList);
  }

  if (actions.length) {
    const actionDetails = document.createElement("details");
    actionDetails.className = "web-actions-details";
    const actionSummary = document.createElement("summary");
    actionSummary.textContent = `${actions.length} research action${actions.length === 1 ? "" : "s"}`;
    const actionList = document.createElement("div");
    actionList.className = "web-action-list";

    actions.forEach((action) => {
      const item = document.createElement("div");
      item.className = "web-action-item";
      if (action.type === "search") {
        item.textContent = `Search: ${action.query || "web query"}`;
      } else if (action.type === "open_page") {
        item.textContent = `Opened: ${action.url || "page"}`;
      } else if (action.type === "find_in_page") {
        item.textContent = `Found in page: ${action.pattern || "text"}`;
      } else {
        item.textContent = action.type || "Web action";
      }
      actionList.appendChild(item);
    });

    actionDetails.append(actionSummary, actionList);
    body.appendChild(actionDetails);
  }

  details.append(summary, body);
  return details;
}

function makeMessageAction(label, title, handler) {
  const button = document.createElement("button");
  button.className = "message-action";
  button.type = "button";
  button.textContent = label;
  button.title = title;
  button.addEventListener("click", handler);
  return button;
}

function messageElement(message, temporary = false) {
  const row = document.createElement("div");
  row.className = `message-row ${message.role}`;
  row.dataset.messageId = message.id || "";
  if (message.metadata?.error) row.classList.add("error");
  if (message.metadata?.stopped) row.classList.add("stopped");
  if (temporary) row.dataset.temporary = "true";

  const inner = document.createElement("div");
  inner.className = "message-inner";

  const content = document.createElement("div");
  content.className = "message-content";
  const generated = message.metadata?.generated_files || [];

  if (message.role === "assistant") {
    const reasoningPanel = reasoningPanelElement(message.metadata || {});
    if (reasoningPanel) inner.appendChild(reasoningPanel.element);
    const citedContent = applyCitationMarkers(
      message.content,
      message.metadata?.web_research?.citations || []
    );
    renderCodeAwareContent(content, citedContent, generated);
  } else {
    content.textContent = message.content;
  }

  inner.appendChild(content);

  if (message.metadata?.stopped) {
    const stopped = document.createElement("div");
    stopped.className = "stopped-label";
    stopped.textContent = "Generation stopped";
    inner.appendChild(stopped);
  }

  if (generated.length) {
    const links = document.createElement("div");
    links.className = "generated-files";
    generated.forEach((file) => {
      const link = document.createElement("a");
      link.className = "generated-file";
      link.href = file.url;
      link.textContent = `⬇ ${file.filename}`;
      links.appendChild(link);
    });
    content.appendChild(links);
  }

  if (message.role === "assistant") {
    const webResearch = webResearchElement(message.metadata || {});
    if (webResearch) inner.appendChild(webResearch);
  }

  if (!temporary && message.id) {
    const actions = document.createElement("div");
    actions.className = "message-actions";

    if (message.role === "user") {
      actions.appendChild(
        makeMessageAction("Edit", "Edit and resend from here", () =>
          startInlineEdit(row, message)
        )
      );
    } else if (message.role === "assistant" && !message.metadata?.error) {
      actions.appendChild(
        makeMessageAction(
          "Regenerate",
          "Regenerate with the currently selected model",
          () => regenerateMessage(message)
        )
      );
    }

    actions.appendChild(
      makeMessageAction("Branch", "Start a new chat from here", () =>
        branchFromMessage(message)
      )
    );
    actions.appendChild(
      makeMessageAction("Copy", "Copy message", () =>
        copyMessage(message.content)
      )
    );
    actions.appendChild(
      makeMessageAction("Delete", "Delete this message and everything after it", () =>
        deleteFromMessage(message)
      )
    );

    inner.appendChild(actions);
  }

  row.appendChild(inner);
  return row;
}

function renderMessages(messages) {
  clearMathTypeset(el.messages);
  el.messages.innerHTML = "";
  if (!messages.length) {
    renderEmpty();
    return;
  }
  messages.forEach((message) => el.messages.appendChild(messageElement(message)));
  scrollToBottom();
}

function scrollToBottom() {
  requestAnimationFrame(() => {
    el.messages.scrollTop = el.messages.scrollHeight;
  });
}

function renderPendingFiles() {
  el.pendingFiles.innerHTML = "";
  state.pendingAttachments.forEach((attachment) => {
    const chip = document.createElement("div");
    chip.className = "pending-file";

    const name = document.createElement("span");
    name.textContent = attachment.original_name;

    const remove = document.createElement("button");
    remove.textContent = "×";
    remove.title = "Remove from next message";
    remove.addEventListener("click", () => {
      state.pendingAttachments = state.pendingAttachments.filter(
        (item) => item.id !== attachment.id
      );
      renderPendingFiles();
    });

    chip.append(name, remove);
    el.pendingFiles.appendChild(chip);
  });
}

function renderProjectContext() {
  let project = state.activeConversation?.project || null;
  let files = state.activeConversation?.project_files || [];

  if (!project &&
      state.activeProjectFilter !== "all" &&
      state.activeProjectFilter !== "general") {
    project = projectById(state.activeProjectFilter);
  }

  if (!project) {
    el.projectContext.classList.add("hidden");
    el.activeProjectBadge.classList.add("hidden");
    return;
  }

  const activeFileCount = files.length
    ? files.filter((file) => file.is_active).length
    : project.active_file_count || 0;

  el.projectContext.classList.remove("hidden");
  el.projectContext.textContent =
    `${project.name} · ${activeFileCount} active project file${
      activeFileCount === 1 ? "" : "s"
    }`;

  el.activeProjectBadge.classList.remove("hidden");
  el.activeProjectBadge.textContent = project.name;
  el.activeProjectBadge.title = "Open project settings";
  el.activeProjectBadge.onclick = () => openProjectModal(project.id);
}

async function loadProjects() {
  state.projects = await api("/api/projects");
  renderProjects();
  renderProjectContext();
}

async function loadConversations() {
  state.conversations = await api("/api/conversations");
  renderConversations();
}

async function loadWorkspace() {
  await Promise.all([loadProjects(), loadConversations()]);
}

async function selectProjectFilter(filter) {
  state.activeProjectFilter = filter;
  renderProjects();
  renderConversations();

  if (filter !== "all" && filter !== "general") {
    applyProjectDefaults(projectById(filter));
  }

  const currentStillVisible = filteredConversations().some(
    (conversation) => conversation.id === state.activeConversationId
  );

  if (!currentStillVisible) {
    const first = filteredConversations()[0];
    if (first) {
      await openConversation(first.id);
    } else {
      state.activeConversationId = null;
      state.activeConversation = null;
      renderEmpty();
    }
  } else {
    renderProjectContext();
  }
}

async function createConversation() {
  const projectId =
    state.activeProjectFilter !== "all" &&
    state.activeProjectFilter !== "general"
      ? state.activeProjectFilter
      : null;

  const conversation = await api("/api/conversations", {
    method: "POST",
    body: JSON.stringify({
      model_id: el.model.value || state.catalog.default_model,
      project_id: projectId,
    }),
  });

  state.pendingAttachments = [];
  renderPendingFiles();
  await loadWorkspace();
  await openConversation(conversation.id);
  el.messageInput.focus();
}

async function openConversation(id) {
  state.activeConversationId = id;
  state.activeConversation = await api(`/api/conversations/${id}`);

  refreshModelSelect(state.activeConversation.model_id);
  renderMessages(state.activeConversation.messages);

  state.pendingAttachments = [];
  renderPendingFiles();
  renderConversations();
  renderProjectContext();
  el.sidebar.classList.remove("open");
}

async function refreshActiveConversation() {
  if (!state.activeConversationId) return;
  await openConversation(state.activeConversationId);
}

async function ensureConversation() {
  if (!state.activeConversationId) {
    await createConversation();
  }
  return state.activeConversationId;
}

async function uploadFiles(fileList) {
  if (!fileList.length) return;

  const conversationId = await ensureConversation();
  const form = new FormData();
  Array.from(fileList).forEach((file) => form.append("files", file));

  showToast("Uploading files locally...");
  const attachments = await api(
    `/api/conversations/${conversationId}/attachments`,
    { method: "POST", body: form }
  );
  state.pendingAttachments.push(...attachments);
  renderPendingFiles();
  showToast(
    `${attachments.length} file${attachments.length === 1 ? "" : "s"} attached`
  );
}

function autosizeTextarea() {
  el.messageInput.style.height = "auto";
  el.messageInput.style.height =
    `${Math.min(el.messageInput.scrollHeight, 180)}px`;
}

function setBusy(busy) {
  state.busy = busy;
  el.sendButton.classList.toggle("stop-mode", busy);
  el.sendButton.textContent = busy ? "■" : "↑";
  el.sendButton.title = busy ? "Stop generation" : "Send";
}

async function stopGeneration() {
  if (!state.busy || !state.abortController || !state.activeConversationId) {
    return;
  }

  const controller = state.abortController;
  try {
    await api(
      `/api/conversations/${state.activeConversationId}/cancel`,
      { method: "POST" }
    );
  } catch (error) {
    console.warn("Could not signal backend cancellation", error);
  } finally {
    controller.abort();
  }
}

function messageRequestPayload(content, attachmentIds) {
  return {
    content,
    model_id: el.model.value,
    reasoning_effort: el.reasoning.value,
    verbosity: el.verbosity.value,
    max_output_tokens: Number(el.maxOutputTokens.value),
    use_code_interpreter: el.codeInterpreter.checked,
    use_web_search: el.webSearch.checked,
    research_depth: "thorough",
    web_allowed_domains: parseDomainList(el.webAllowedDomains.value),
    web_blocked_domains: parseDomainList(el.webBlockedDomains.value),
    attachment_ids: attachmentIds,
  };
}

async function sendMessageWithContent(
  content,
  attachmentIds = [],
  { clearComposer = false } = {}
) {
  const cleaned = content.trim();
  if (!cleaned || state.busy) return;

  const conversationId = await ensureConversation();
  setBusy(true);
  state.abortController = new AbortController();

  const optimistic = {
    role: "user",
    content: cleaned,
    metadata: { attachment_ids: attachmentIds },
  };

  if (el.messages.querySelector(".welcome")) el.messages.innerHTML = "";
  const optimisticElement = messageElement(optimistic, true);
  el.messages.appendChild(optimisticElement);
  const streaming = addStreamingAssistant();

  if (clearComposer) {
    state.pendingAttachments = [];
    renderPendingFiles();
    el.messageInput.value = "";
    autosizeTextarea();
  }

  let completed = false;
  let serverFailed = false;
  let serverFailureText = "";

  try {
    await streamApi(
      `/api/conversations/${conversationId}/messages/stream`,
      messageRequestPayload(cleaned, attachmentIds),
      async (event) => {
        switch (event.type) {
          case "started":
            optimisticElement.replaceWith(messageElement(event.user_message));
            break;

          case "status":
            updateStreamingStatus(streaming, event.label);
            break;

          case "reasoning_delta":
            appendStreamingReasoning(streaming, event.delta || "");
            break;

          case "reasoning_done":
            setStreamingReasoning(streaming, event.text || "");
            break;

          case "activity":
            addStreamingActivity(streaming, event.label);
            updateStreamingStatus(streaming, event.label);
            break;

          case "output_delta":
            streaming.answerText += event.delta || "";
            updateStreamingStatus(streaming, "Writing response");
            scheduleStreamingAnswerRender(streaming);
            break;

          case "done":
            completed = true;
            finishStreamingAssistant(streaming);
            break;

          case "error":
            serverFailed = true;
            serverFailureText = event.message || "The streamed request failed";
            if (document.body.contains(streaming.row)) {
              finishStreamingAssistant(streaming);
            }
            break;

          default:
            break;
        }
      },
      state.abortController.signal
    );

    if (serverFailed) {
      showToast(serverFailureText);
      await refreshActiveConversation();
    } else if (!completed) {
      throw new Error("The response stream ended before completion");
    } else {
      await refreshActiveConversation();
    }

    await loadWorkspace();
  } catch (error) {
    if (error.name === "AbortError") {
      const duration =
        (performance.now() - streaming.startedAt) / 1000;

      try {
        await api(
          `/api/conversations/${conversationId}/messages/partial`,
          {
            method: "POST",
            body: JSON.stringify({
              content: streaming.answerText,
              model_id: el.model.value,
              reasoning_summary: streaming.reasoningText,
              activities: Array.from(streaming.activities),
              duration_seconds: duration,
            }),
          }
        );
      } catch (saveError) {
        console.error("Could not save partial response", saveError);
      }

      if (document.body.contains(streaming.row)) {
        finishStreamingAssistant(streaming);
      }
      showToast("Generation stopped");
      await new Promise((resolve) => setTimeout(resolve, 150));
      await refreshActiveConversation();
      await loadWorkspace();
    } else {
      if (document.body.contains(streaming.row)) {
        finishStreamingAssistant(streaming);
      }
      showToast(error.message);
      await refreshActiveConversation();
    }
  } finally {
    state.abortController = null;
    setBusy(false);
    el.messageInput.focus();
  }
}

async function sendMessage() {
  const content = el.messageInput.value;
  const attachmentIds = state.pendingAttachments.map((item) => item.id);
  await sendMessageWithContent(content, attachmentIds, {
    clearComposer: true,
  });
}

async function copyMessage(content) {
  try {
    await navigator.clipboard.writeText(content);
    showToast("Copied");
  } catch (_) {
    showToast("Could not copy message");
  }
}

function startInlineEdit(row, message) {
  if (state.busy) return;

  const inner = row.querySelector(".message-inner");
  if (!inner || inner.querySelector(".message-editor")) return;

  const original = inner.innerHTML;
  inner.innerHTML = "";

  const editor = document.createElement("textarea");
  editor.className = "message-editor";
  editor.value = message.content;
  editor.rows = Math.max(3, message.content.split("\n").length);

  const controls = document.createElement("div");
  controls.className = "message-editor-controls";

  const cancel = document.createElement("button");
  cancel.className = "secondary-button";
  cancel.textContent = "Cancel";
  cancel.addEventListener("click", () => {
    inner.innerHTML = original;
    renderMessages(state.activeConversation.messages);
  });

  const submit = document.createElement("button");
  submit.className = "primary-button";
  submit.textContent = "Send";
  submit.addEventListener("click", async () => {
    const next = editor.value.trim();
    if (!next) return;
    const attachmentIds = message.metadata?.attachment_ids || [];
    await api(
      `/api/conversations/${state.activeConversationId}/messages/${message.id}`,
      { method: "DELETE" }
    );
    await refreshActiveConversation();
    await sendMessageWithContent(next, attachmentIds);
  });

  controls.append(cancel, submit);
  inner.append(editor, controls);
  editor.focus();
  editor.setSelectionRange(editor.value.length, editor.value.length);
}

async function regenerateMessage(message) {
  if (state.busy || !state.activeConversation) return;

  const messages = state.activeConversation.messages;
  const assistantIndex = messages.findIndex((item) => item.id === message.id);
  let userMessage = null;

  for (let index = assistantIndex - 1; index >= 0; index -= 1) {
    if (messages[index].role === "user") {
      userMessage = messages[index];
      break;
    }
  }

  if (!userMessage) {
    showToast("No user message was found to regenerate");
    return;
  }

  const attachmentIds = userMessage.metadata?.attachment_ids || [];
  await api(
    `/api/conversations/${state.activeConversationId}/messages/${userMessage.id}`,
    { method: "DELETE" }
  );
  await refreshActiveConversation();
  await sendMessageWithContent(userMessage.content, attachmentIds);
}

async function branchFromMessage(message) {
  if (state.busy) return;

  const branch = await api(
    `/api/conversations/${state.activeConversationId}/branch/${message.id}`,
    { method: "POST" }
  );

  if (branch.project_id) {
    state.activeProjectFilter = branch.project_id;
  } else {
    state.activeProjectFilter = "general";
  }

  await loadWorkspace();
  await openConversation(branch.id);
  showToast("Conversation branched");
}

async function deleteFromMessage(message) {
  if (state.busy) return;
  if (!confirm("Delete this message and everything after it?")) return;

  await api(
    `/api/conversations/${state.activeConversationId}/messages/${message.id}`,
    { method: "DELETE" }
  );
  await refreshActiveConversation();
  await loadWorkspace();
}

async function moveConversation(conversation) {
  const choices = [
    "0: General",
    ...state.projects.map(
      (project, index) => `${index + 1}: ${project.name}`
    ),
  ].join("\n");

  const answer = prompt(`Move chat to:\n\n${choices}`, "0");
  if (answer === null) return;

  const number = Number(answer);
  if (!Number.isInteger(number) || number < 0 || number > state.projects.length) {
    showToast("Invalid project selection");
    return;
  }

  const projectId = number === 0 ? null : state.projects[number - 1].id;
  await api(`/api/conversations/${conversation.id}`, {
    method: "PATCH",
    body: JSON.stringify({ project_id: projectId }),
  });

  await loadWorkspace();
  if (state.activeConversationId === conversation.id) {
    await refreshActiveConversation();
  }
}

function refreshProjectModelSelect(selectedModel) {
  const fallback = state.catalog.models.some(
    (model) => model.id === selectedModel
  )
    ? selectedModel
    : state.catalog.default_model;

  fillSelect(
    el.projectModel,
    state.catalog.models,
    fallback,
    (model) => model.label
  );
}

async function createProject() {
  const project = await api("/api/projects", {
    method: "POST",
    body: JSON.stringify({
      name: "New project",
      instructions: "",
      default_model_id: el.model.value || state.catalog.default_model,
    }),
  });

  state.activeProjectFilter = project.id;
  await loadWorkspace();
  await openProjectModal(project.id);
}

async function openProjectModal(projectId) {
  const project = await api(`/api/projects/${projectId}`);
  state.editingProjectId = projectId;

  el.projectModalTitle.textContent = project.name;
  el.projectName.value = project.name;
  el.projectInstructions.value = project.instructions || "";

  refreshProjectModelSelect(project.default_model_id);

  renderProjectFiles(project);
  el.projectModal.classList.remove("hidden");
  el.projectName.focus();
}

function closeProjectModal() {
  state.editingProjectId = null;
  el.projectModal.classList.add("hidden");
  el.projectFileInput.value = "";
}

function renderProjectFiles(project) {
  el.projectFilesList.innerHTML = "";

  if (!project.files.length) {
    const empty = document.createElement("div");
    empty.className = "project-files-empty";
    empty.textContent = "No project files yet.";
    el.projectFilesList.appendChild(empty);
    return;
  }

  project.files.forEach((file) => {
    const row = document.createElement("div");
    row.className = "project-file-row";

    const active = document.createElement("input");
    active.type = "checkbox";
    active.checked = file.is_active;
    active.title = "Use this file in project chats";
    active.addEventListener("change", async () => {
      await api(
        `/api/projects/${project.id}/files/${file.id}`,
        {
          method: "PATCH",
          body: JSON.stringify({ is_active: active.checked }),
        }
      );
      await loadProjects();
    });

    const info = document.createElement("div");
    info.className = "project-file-info";

    const name = document.createElement("a");
    name.className = "project-file-name";
    name.textContent = file.original_name;
    name.href = `/api/projects/${project.id}/files/${file.id}/download`;

    const size = document.createElement("span");
    size.className = "project-file-size";
    size.textContent = formatBytes(file.size_bytes);

    info.append(name, size);

    const remove = document.createElement("button");
    remove.className = "mini-button";
    remove.title = "Delete project file";
    remove.textContent = "×";
    remove.addEventListener("click", async () => {
      if (!confirm(`Delete "${file.original_name}" from this project?`)) return;
      await api(
        `/api/projects/${project.id}/files/${file.id}`,
        { method: "DELETE" }
      );
      await openProjectModal(project.id);
      await loadProjects();
    });

    row.append(active, info, remove);
    el.projectFilesList.appendChild(row);
  });
}

function formatBytes(bytes) {
  const value = Number(bytes);
  if (!Number.isFinite(value)) return "";
  if (value < 1024) return `${value} B`;
  if (value < 1024 * 1024) return `${(value / 1024).toFixed(1)} KB`;
  return `${(value / (1024 * 1024)).toFixed(1)} MB`;
}

async function saveProject() {
  if (!state.editingProjectId) return;

  const projectId = state.editingProjectId;
  const name = el.projectName.value.trim();
  if (!name) {
    showToast("Project name is required");
    return;
  }

  await api(`/api/projects/${projectId}`, {
    method: "PATCH",
    body: JSON.stringify({
      name,
      instructions: el.projectInstructions.value,
      default_model_id: el.projectModel.value,
    }),
  });

  closeProjectModal();
  await loadWorkspace();
  if (state.activeConversation?.project_id === projectId) {
    await refreshActiveConversation();
  }
  showToast("Project saved");
}

async function deleteProject() {
  if (!state.editingProjectId) return;

  const projectId = state.editingProjectId;
  const name = el.projectName.value || "this project";
  if (!confirm(`Delete "${name}"? Its chats will be kept under General.`)) {
    return;
  }

  await api(`/api/projects/${projectId}`, { method: "DELETE" });
  closeProjectModal();

  if (state.activeProjectFilter === projectId) {
    state.activeProjectFilter = "general";
  }

  await loadWorkspace();
  if (state.activeConversationId) {
    await refreshActiveConversation();
  }
  showToast("Project deleted. Its chats were moved to General.");
}

async function uploadProjectFiles(fileList) {
  if (!state.editingProjectId || !fileList.length) return;

  const form = new FormData();
  Array.from(fileList).forEach((file) => form.append("files", file));

  showToast("Uploading project files...");
  await api(
    `/api/projects/${state.editingProjectId}/files`,
    { method: "POST", body: form }
  );

  await openProjectModal(state.editingProjectId);
  await loadProjects();
  showToast("Project files added");
}

async function initialize() {
  try {
    state.catalog = await api("/api/catalog");
    document.title = state.catalog.title;

    refreshModelSelect(state.catalog.default_model);

    await loadWorkspace();

    if (state.conversations.length) {
      await openConversation(state.conversations[0].id);
    } else {
      renderEmpty();
    }
  } catch (error) {
    showToast(error.message);
  }
}

el.newChat.addEventListener("click", createConversation);
el.addProject.addEventListener("click", createProject);
el.allChatsFilter.addEventListener("click", () => selectProjectFilter("all"));
el.generalChatsFilter.addEventListener("click", () =>
  selectProjectFilter("general")
);

el.model.addEventListener("change", refreshModelControls);

el.settingsToggle.addEventListener("click", () => {
  el.settingsPanel.classList.toggle("hidden");
});
el.webSearch.addEventListener("change", updateToolControls);
el.codeInterpreter.addEventListener("change", updateToolControls);

el.sidebarToggle.addEventListener("click", () => {
  el.sidebar.classList.toggle("open");
});

el.fileInput.addEventListener("change", async () => {
  try {
    await uploadFiles(el.fileInput.files);
  } catch (error) {
    showToast(error.message);
  } finally {
    el.fileInput.value = "";
  }
});

el.sendButton.addEventListener("click", () => {
  if (state.busy) {
    stopGeneration();
  } else {
    sendMessage();
  }
});

el.messageInput.addEventListener("input", autosizeTextarea);
el.messageInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    if (!state.busy) sendMessage();
  }
});

el.projectModalClose.addEventListener("click", closeProjectModal);
el.cancelProject.addEventListener("click", closeProjectModal);
el.saveProject.addEventListener("click", saveProject);
el.deleteProject.addEventListener("click", deleteProject);
el.projectFileInput.addEventListener("change", async () => {
  try {
    await uploadProjectFiles(el.projectFileInput.files);
  } catch (error) {
    showToast(error.message);
  } finally {
    el.projectFileInput.value = "";
  }
});
el.projectModal.addEventListener("click", (event) => {
  if (event.target === el.projectModal) closeProjectModal();
});

initialize();
