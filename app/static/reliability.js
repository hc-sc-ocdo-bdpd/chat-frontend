// Reliability and UX extensions for Foundry Chat.
// Loaded after app.js so this can preserve the existing UI while improving
// settings persistence, submitted-file visibility, and per-chat concurrency.
(() => {
  const originalRefreshModelControls = refreshModelControls;
  const originalMessageElement = messageElement;
  const originalOpenConversation = openConversation;
  const originalRenderConversations = renderConversations;

  const generations = new Map();
  let settingsInitialized = false;
  let desiredSettings = null;

  function readDesiredSettings() {
    return {
      codeInterpreter: Boolean(el.codeInterpreter.checked),
      webSearch: Boolean(el.webSearch.checked),
      maxOutputTokens: el.maxOutputTokens.value,
    };
  }

  function rememberDesiredSettings() {
    if (!settingsInitialized) return;
    desiredSettings = readDesiredSettings();
  }

  function applyDesiredSettings() {
    if (!desiredSettings) return;
    const model = currentModel();
    if (!model) return;

    el.codeInterpreter.disabled = !model.supports_code_interpreter;
    el.webSearch.disabled = !model.supports_web_search;
    el.codeInterpreter.checked = model.supports_code_interpreter
      ? desiredSettings.codeInterpreter
      : false;
    el.webSearch.checked = model.supports_web_search
      ? desiredSettings.webSearch
      : false;

    if (desiredSettings.maxOutputTokens) {
      el.maxOutputTokens.value = desiredSettings.maxOutputTokens;
    }
    updateToolControls();
  }

  // app.js intentionally applies model defaults on first load. After that,
  // changing/opening a model should not silently overwrite the user's tools or
  // token setting.
  refreshModelControls = function reliableRefreshModelControls() {
    if (!settingsInitialized) {
      originalRefreshModelControls();
      settingsInitialized = true;
      desiredSettings = readDesiredSettings();
      return;
    }

    originalRefreshModelControls();
    applyDesiredSettings();
  };

  // app.js registered its model-change listener before this file loads, so this
  // listener runs afterwards and restores the user's intended settings.
  el.model.addEventListener("change", applyDesiredSettings);
  el.codeInterpreter.addEventListener("change", rememberDesiredSettings);
  el.webSearch.addEventListener("change", rememberDesiredSettings);
  el.maxOutputTokens.addEventListener("input", rememberDesiredSettings);

  function findAttachment(attachmentId) {
    const conversationAttachments = state.activeConversation?.attachments || [];
    const active = conversationAttachments.find((item) => item.id === attachmentId);
    if (active) return active;

    const pending = state.pendingAttachments.find((item) => item.id === attachmentId);
    if (pending) return pending;

    for (const generation of generations.values()) {
      const attached = generation.attachments.find(
        (item) => item.id === attachmentId
      );
      if (attached) return attached;
    }
    return null;
  }

  function submittedFilesElement(attachmentIds) {
    if (!attachmentIds.length) return null;

    const files = document.createElement("div");
    files.className = "generated-files submitted-files";

    attachmentIds.forEach((attachmentId) => {
      const attachment = findAttachment(attachmentId);
      const chip = document.createElement("span");
      chip.className = "generated-file";
      if (attachment) {
        const size = formatBytes(attachment.size_bytes);
        chip.textContent = `📎 ${attachment.original_name}${size ? ` · ${size}` : ""}`;
        chip.title = attachment.original_name;
      } else {
        chip.textContent = `📎 Attached file`;
        chip.title = attachmentId;
      }
      files.appendChild(chip);
    });

    return files;
  }

  messageElement = function reliableMessageElement(message, temporary = false) {
    const row = originalMessageElement(message, temporary);
    if (message.role !== "user") return row;

    const attachmentIds = Array.isArray(message.metadata?.attachment_ids)
      ? message.metadata.attachment_ids
      : [];
    const submittedFiles = submittedFilesElement(attachmentIds);
    const content = row.querySelector(".message-content");
    if (submittedFiles && content) content.appendChild(submittedFiles);
    return row;
  };

  function activeGeneration() {
    return state.activeConversationId
      ? generations.get(state.activeConversationId) || null
      : null;
  }

  function syncBusyState() {
    const generation = activeGeneration();
    const busy = Boolean(generation && !generation.done);
    state.busy = busy;
    state.abortController = busy ? generation.controller : null;
    el.sendButton.classList.toggle("stop-mode", busy);
    el.sendButton.textContent = busy ? "■" : "↑";
    el.sendButton.title = busy ? "Stop generation" : "Send";
  }

  function generationStreamingUi(generation) {
    if (state.activeConversationId !== generation.conversationId || generation.done) {
      return null;
    }

    if (
      generation.streaming &&
      generation.streaming.row &&
      document.body.contains(generation.streaming.row)
    ) {
      return generation.streaming;
    }

    if (el.messages.querySelector(".welcome")) el.messages.innerHTML = "";
    const streaming = addStreamingAssistant();
    streaming.startedAt = generation.startedAt;
    streaming.answerText = generation.answerText;
    streaming.reasoningText = generation.reasoningText;
    streaming.activities = new Set(generation.activities);
    streaming.statusLabel = generation.statusLabel;
    setStreamingReasoning(streaming, generation.reasoningText);
    generation.activities.forEach((label) => addStreamingActivity(streaming, label));
    updateStreamingStatus(streaming, generation.statusLabel);
    if (generation.answerText) scheduleStreamingAnswerRender(streaming);
    generation.streaming = streaming;
    return streaming;
  }

  function renderGenerationIndicator() {
    const generation = activeGeneration();
    if (generation && !generation.done) generationStreamingUi(generation);
  }

  openConversation = async function reliableOpenConversation(id) {
    const result = await originalOpenConversation(id);
    syncBusyState();
    renderGenerationIndicator();
    return result;
  };

  renderConversations = function reliableRenderConversations() {
    originalRenderConversations();
    const visible = filteredConversations();
    Array.from(el.conversationList.children).forEach((row, index) => {
      const conversation = visible[index];
      if (!conversation || !generations.has(conversation.id)) return;
      const titleWrap = row.querySelector(".conversation-title-wrap");
      if (!titleWrap) return;
      const indicator = document.createElement("span");
      indicator.className = "project-count";
      indicator.textContent = "✦";
      indicator.title = "Generation in progress";
      titleWrap.appendChild(indicator);
    });
  };

  function attachmentSnapshot(attachmentIds) {
    return attachmentIds
      .map((attachmentId) => findAttachment(attachmentId))
      .filter(Boolean);
  }

  async function refreshConversationIfActive(conversationId) {
    if (state.activeConversationId !== conversationId) return;

    // Do not throw away files the user may have attached for their next message
    // while the current generation was still running.
    const pending = [...state.pendingAttachments];
    await originalOpenConversation(conversationId);
    state.pendingAttachments = pending;
    renderPendingFiles();
    syncBusyState();
    renderGenerationIndicator();
  }

  async function saveStoppedPartial(generation) {
    if (!generation.answerText.trim()) return;
    try {
      await api(
        `/api/conversations/${generation.conversationId}/messages/partial`,
        {
          method: "POST",
          body: JSON.stringify({
            content: generation.answerText,
            model_id: generation.modelId,
            use_web_search: generation.useWebSearch,
            research_depth: "thorough",
            reasoning_summary: generation.reasoningText,
            activities: generation.activities,
            duration_seconds: (performance.now() - generation.startedAt) / 1000,
          }),
        }
      );
    } catch (error) {
      console.error("Could not save partial response", error);
    }
  }

  sendMessageWithContent = async function reliableSendMessageWithContent(
    content,
    attachmentIds = [],
    { clearComposer = false } = {}
  ) {
    const cleaned = content.trim();
    if (!cleaned) return;

    const conversationId = await ensureConversation();
    if (generations.has(conversationId)) {
      showToast("This chat already has a generation in progress");
      return;
    }

    const controller = new AbortController();
    const generation = {
      conversationId,
      controller,
      attachments: attachmentSnapshot(attachmentIds),
      answerText: "",
      reasoningText: "",
      activities: [],
      activitySet: new Set(),
      statusLabel: "Thinking",
      startedAt: performance.now(),
      streaming: null,
      optimisticElement: null,
      done: false,
      modelId: el.model.value,
      useWebSearch: el.webSearch.checked,
    };
    generations.set(conversationId, generation);
    syncBusyState();
    renderConversations();

    const optimistic = {
      role: "user",
      content: cleaned,
      metadata: { attachment_ids: attachmentIds },
    };

    if (state.activeConversationId === conversationId) {
      if (el.messages.querySelector(".welcome")) el.messages.innerHTML = "";
      generation.optimisticElement = messageElement(optimistic, true);
      el.messages.appendChild(generation.optimisticElement);
      generation.streaming = addStreamingAssistant();
      generation.streaming.startedAt = generation.startedAt;
    }

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
          let streaming = null;
          if (state.activeConversationId === conversationId) {
            streaming = generationStreamingUi(generation);
          }

          switch (event.type) {
            case "started":
              if (
                generation.optimisticElement &&
                document.body.contains(generation.optimisticElement)
              ) {
                generation.optimisticElement.replaceWith(
                  messageElement(event.user_message)
                );
              }
              generation.optimisticElement = null;
              break;

            case "status":
              generation.statusLabel = event.label || "Thinking";
              if (streaming) updateStreamingStatus(streaming, generation.statusLabel);
              break;

            case "reasoning_delta":
              generation.reasoningText += event.delta || "";
              if (streaming) {
                appendStreamingReasoning(streaming, event.delta || "");
              }
              break;

            case "reasoning_done":
              generation.reasoningText = event.text || generation.reasoningText;
              if (streaming) setStreamingReasoning(streaming, generation.reasoningText);
              break;

            case "activity":
              if (event.label && !generation.activitySet.has(event.label)) {
                generation.activitySet.add(event.label);
                generation.activities.push(event.label);
              }
              generation.statusLabel = event.label || generation.statusLabel;
              if (streaming && event.label) {
                addStreamingActivity(streaming, event.label);
                updateStreamingStatus(streaming, event.label);
              }
              break;

            case "output_delta":
              generation.answerText += event.delta || "";
              generation.statusLabel = "Writing response";
              if (streaming) {
                streaming.answerText = generation.answerText;
                updateStreamingStatus(streaming, "Writing response");
                scheduleStreamingAnswerRender(streaming);
              }
              break;

            case "done":
              completed = true;
              generation.done = true;
              if (streaming && document.body.contains(streaming.row)) {
                finishStreamingAssistant(streaming);
              }
              break;

            case "error":
              serverFailed = true;
              serverFailureText = event.message || "The streamed request failed";
              generation.done = true;
              if (streaming && document.body.contains(streaming.row)) {
                finishStreamingAssistant(streaming);
              }
              break;

            default:
              break;
          }
        },
        controller.signal
      );

      if (serverFailed) {
        showToast(serverFailureText);
      } else if (!completed) {
        throw new Error("The response stream ended before completion");
      }

      await refreshConversationIfActive(conversationId);
      await loadWorkspace();
    } catch (error) {
      if (error.name === "AbortError") {
        await saveStoppedPartial(generation);
        generation.done = true;
        const streaming = generation.streaming;
        if (streaming && document.body.contains(streaming.row)) {
          finishStreamingAssistant(streaming);
        }
        showToast("Generation stopped");
        await new Promise((resolve) => setTimeout(resolve, 150));
        await refreshConversationIfActive(conversationId);
        await loadWorkspace();
      } else {
        generation.done = true;
        const streaming = generation.streaming;
        if (streaming && document.body.contains(streaming.row)) {
          finishStreamingAssistant(streaming);
        }
        showToast(error.message);
        await refreshConversationIfActive(conversationId);
        await loadWorkspace();
      }
    } finally {
      generations.delete(conversationId);
      renderConversations();
      syncBusyState();
      if (state.activeConversationId === conversationId) {
        el.messageInput.focus();
      }
    }
  };

  stopGeneration = async function reliableStopGeneration() {
    const conversationId = state.activeConversationId;
    const generation = conversationId ? generations.get(conversationId) : null;
    if (!generation) return;

    try {
      await api(`/api/conversations/${conversationId}/cancel`, {
        method: "POST",
      });
    } catch (error) {
      console.warn("Could not signal backend cancellation", error);
    } finally {
      generation.controller.abort();
    }
  };
})();
