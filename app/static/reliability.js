// Settings and attachment UX extensions for Foundry Chat.
// Loaded after app.js so these small overrides can preserve user tool settings
// and show submitted files without replacing the generation lifecycle.
(() => {
  const originalRefreshModelControls = refreshModelControls;
  const originalMessageElement = messageElement;

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

  // app.js registered the model listener first. This listener runs afterwards
  // and reapplies the user's intended settings after model defaults.
  el.model.addEventListener("change", applyDesiredSettings);
  el.codeInterpreter.addEventListener("change", rememberDesiredSettings);
  el.webSearch.addEventListener("change", rememberDesiredSettings);
  el.maxOutputTokens.addEventListener("input", rememberDesiredSettings);

  function findAttachment(attachmentId) {
    const conversationAttachments =
      state.activeConversation?.attachments || [];
    const active = conversationAttachments.find(
      (item) => item.id === attachmentId
    );
    if (active) return active;

    const pending = state.pendingAttachments.find(
      (item) => item.id === attachmentId
    );
    if (pending) return pending;

    for (const generation of state.generations.values()) {
      const attached = (generation.attachments || []).find(
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
        chip.textContent =
          `📎 ${attachment.original_name}${size ? ` · ${size}` : ""}`;
        chip.title = attachment.original_name;
      } else {
        chip.textContent = "📎 Attached file";
        chip.title = attachmentId;
      }
      files.appendChild(chip);
    });

    return files;
  }

  messageElement = function reliableMessageElement(
    message,
    temporary = false
  ) {
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
})();
