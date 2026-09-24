// Browser regression checks with a mock API. No Azure requests or credentials.
// Setup: npm install --no-save --package-lock=false playwright
//        npx playwright install chromium
// Run:   node --test tests/frontend_browser.test.cjs
// Optional: set PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH to an installed Chromium.
const { test, before, after, beforeEach, afterEach } = require("node:test");
const assert = require("node:assert/strict");
const { createServer } = require("node:http");
const { readFile } = require("node:fs/promises");
const { resolve } = require("node:path");
const { chromium } = require("playwright");

let server, browser, baseURL, context, page, fixture;
const staticRoot = resolve(__dirname, "../app/static");
const model = {
  id: "test-model", label: "Test model", reasoning_efforts: ["auto"],
  default_reasoning_effort: "auto", verbosity_options: ["medium"],
  default_verbosity: "medium", supports_code_interpreter: true,
  default_code_interpreter: true, supports_web_search: true,
  default_web_search: false, default_max_output_tokens: 16384,
};

function conversation(id) {
  return { id, title: `Chat ${id}`, model_id: model.id, messages: [],
    attachments: [], project_files: [], project_id: null, project: null };
}

before(async () => {
  server = createServer(async (req, res) => {
    const files = { "/": "index.html", "/static/app.js": "app.js",
      "/static/reliability.js": "reliability.js", "/static/styles.css": "styles.css" };
    const name = files[new URL(req.url, "http://localhost").pathname];
    if (!name) { res.writeHead(404); res.end(); return; }
    const type = name.endsWith(".js") ? "text/javascript"
      : name.endsWith(".css") ? "text/css" : "text/html";
    res.writeHead(200, { "Content-Type": type });
    res.end(await readFile(resolve(staticRoot, name)));
  });
  await new Promise((done) => server.listen(0, "127.0.0.1", done));
  baseURL = `http://127.0.0.1:${server.address().port}`;
  browser = await chromium.launch({
    executablePath: process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH || undefined,
    args: ["--no-sandbox", "--disable-dev-shm-usage"],
  });
});

after(async () => {
  await browser?.close();
  if (server) await new Promise((done) => server.close(done));
});

beforeEach(async () => {
  fixture = { conversations: [conversation("a"), conversation("b")], uploads: [],
    sent: [], created: 0, uploadStatus: 200, nextUploadGate: null, errors: [] };
  context = await browser.newContext({ viewport: { width: 1280, height: 900 },
    permissions: ["clipboard-read", "clipboard-write"] });
  page = await context.newPage();
  page.on("pageerror", (error) => fixture.errors.push(error.message));
  await page.route("https://cdn.jsdelivr.net/**", (route) => route.fulfill({ body: "" }));
  await page.route("**/api/**", async (route) => {
    const request = route.request();
    const path = new URL(request.url()).pathname;
    const json = (body, status = 200) => route.fulfill({ status, json: body });
    if (path === "/api/catalog") return json({ title: "Test chat", models: [model], default_model: model.id });
    if (path === "/api/projects") return json([]);
    if (path === "/api/conversations") {
      if (request.method() === "POST") {
        const created = conversation(`new-${++fixture.created}`);
        fixture.conversations.unshift(created);
        return json(created);
      }
      return json(fixture.conversations);
    }
    const id = path.split("/")[3];
    const chat = fixture.conversations.find((item) => item.id === id);
    if (path.endsWith("/attachments")) {
      const form = await new Response(request.postDataBuffer(), {
        headers: { "Content-Type": request.headers()["content-type"] },
      }).formData();
      const files = await Promise.all(form.getAll("files").map(async (file) => ({
        name: file.name, type: file.type, size: file.size, bytes: [...new Uint8Array(await file.arrayBuffer())],
      })));
      const gate = fixture.nextUploadGate;
      fixture.nextUploadGate = null;
      const status = fixture.uploadStatus;
      const number = fixture.uploads.push({ conversationId: id, files });
      if (gate) await gate;
      if (status !== 200) return json({ detail: "File too large" }, status);
      const attachments = files.map((file, index) => ({ id: `upload-${number}-${index}`,
        original_name: file.name, size_bytes: file.size }));
      chat.attachments.push(...attachments);
      return json(attachments);
    }
    if (path.endsWith("/messages/start")) {
      fixture.sent.push({ conversationId: id, ...request.postDataJSON() });
      // Also exercise recovery of the draft if starting a response fails.
      return json({ detail: "Simulated generation failure" }, 503);
    }
    if (path.endsWith("/read")) return json({});
    if (chat) return json(chat);
    return json({ detail: `Unexpected request: ${path}` }, 404);
  });
  await page.goto(baseURL);
  await page.locator(".conversation-item.active").waitFor();
});

afterEach(async () => {
  await context?.close();
  assert.deepEqual(fixture.errors, [], "No uncaught JavaScript errors");
});

async function until(check) {
  const deadline = Date.now() + 5000;
  while (!check()) {
    if (Date.now() > deadline) assert.fail("Timed out waiting for mock API request");
    await new Promise((done) => setTimeout(done, 10));
  }
}

async function render(markdown, generatedFiles = []) {
  await page.evaluate(({ markdown, generatedFiles }) => {
    renderMessages([{ id: "answer", role: "assistant", content: markdown,
      metadata: { generated_files: generatedFiles } }]);
  }, { markdown, generatedFiles });
}

async function transfer(kind, files, selector = "#message-input") {
  return page.evaluate(({ kind, files, selector }) => {
    const data = new DataTransfer();
    files.forEach((file) => data.items.add(new File([
      file.bytes ? new Uint8Array(file.bytes) : (file.text || "example"),
    ], file.name, { type: file.type || "text/plain" })));
    const event = kind === "paste"
      ? new ClipboardEvent("paste", { clipboardData: data, bubbles: true, cancelable: true })
      : new DragEvent(kind, { dataTransfer: data, bubbles: true, cancelable: true });
    document.querySelector(selector).dispatchEvent(event);
    // Real browser drag/clipboard stores are unavailable after the event.
    data.items.clear();
    return event.defaultPrevented;
  }, { kind, files, selector });
}

test("tables have aligned headers, inline formatting, escaped pipes, and safe links", async () => {
  await render([
    "Results:", "| **Name** | Value | Notes |", "| :--- | ---: | :---: |",
    "| [Docs](https://example.com) | `a\\|b` | x\\|y |",
    "| <img src=x onerror=alert(1)> | 12 | [unsafe](javascript:alert) |",
  ].join("\n"));
  const table = page.locator(".markdown-table");
  assert.deepEqual(await table.locator("th").allTextContents(), ["Name", "Value", "Notes"]);
  assert.deepEqual(await table.locator("th").evaluateAll((cells) => cells.map((cell) => cell.style.textAlign)),
    ["left", "right", "center"]);
  assert.equal(await table.locator("th strong").count(), 1);
  assert.equal(await table.locator("td code").textContent(), "a|b");
  assert.equal(await table.locator("td").nth(2).textContent(), "x|y");
  assert.equal(await table.locator("a").getAttribute("href"), "https://example.com");
  assert.equal(await table.locator("img, script").count(), 0);
  assert.equal(await table.locator("[href^='javascript:']").count(), 0);
});

test("tables accept optional edge pipes and ragged rows and end before other blocks", async () => {
  await render("A | B\n- | -:\n| one |\n| two | three | discarded |\n# Next | section");
  assert.deepEqual(await page.locator("tbody tr").evaluateAll((rows) =>
    rows.map((row) => [...row.cells].map((cell) => cell.textContent))), [["one", ""], ["two", "three"]]);
  assert.equal(await page.locator("h1").textContent(), "Next | section");
  await render("| Only |\n| --- |\n| cell |");
  assert.equal(await page.locator("table td").textContent(), "cell");
});

test("plain pipe text, invalid tables, and fenced code are not rendered as tables", async () => {
  await render("a | b\nnot a delimiter\n\n| A | B |\n| --- |\n\n```md\n| A | B |\n| --- | --- |\n| 1 | 2 |\n```");
  assert.equal(await page.locator("table").count(), 0);
  assert.match(await page.locator("pre code").textContent(), /\| 1 \| 2 \|/);
  for (const fence of ["```", "~~~"]) {
    await render(`${fence}md\n| A | B |\n| --- | --- |\n| 1 | 2 |`);
    assert.equal(await page.locator("table").count(), 0, "Unfinished code fences stay literal during streaming");
    assert.match(await page.locator("pre code").textContent(), /\| 1 \| 2 \|/);
  }
});

test("streamed table text gains a table once the delimiter arrives", async () => {
  await page.evaluate(() => {
    window.testStreaming = addStreamingAssistant();
    testStreaming.answerText = "| A | B |\n| --- |";
    scheduleStreamingAnswerRender(testStreaming);
  });
  await page.waitForFunction(() => !testStreaming.renderQueued);
  assert.equal(await page.locator("table").count(), 0);
  await page.evaluate(() => {
    testStreaming.answerText += " --- |\n| one | two |";
    scheduleStreamingAnswerRender(testStreaming);
  });
  await page.locator("table").waitFor();
  assert.deepEqual(await page.locator("table td").allTextContents(), ["one", "two"]);
  await page.evaluate(() => clearInterval(testStreaming.timer));
});

test("wide tables scroll within the message on a narrow screen", async () => {
  await page.setViewportSize({ width: 390, height: 844 });
  const header = Array.from({ length: 8 }, (_, i) => `Column ${i}`).join(" | ");
  await render(`${header}\n${Array(8).fill("---").join(" | ")}\n${Array(8).fill("content").join(" | ")}`);
  const sizes = await page.locator(".markdown-table-wrapper").evaluate((wrapper) => ({
    width: wrapper.clientWidth, scroll: wrapper.scrollWidth,
    page: document.documentElement.scrollWidth, viewport: window.innerWidth,
  }));
  assert.ok(sizes.scroll > sizes.width, "The table has its own horizontal scrollbar");
  assert.ok(sizes.page <= sizes.viewport, "The table does not widen the page");
  await page.locator(".markdown-table-wrapper").focus();
  await page.keyboard.press("ArrowRight");
  await page.waitForFunction(() => document.querySelector(".markdown-table-wrapper").scrollLeft > 0);
});

test("pasting multiple files captures bytes once and preserves existing prompt text", async () => {
  await page.locator("#message-input").fill("Analyze these files");
  const files = [{ name: "report.csv", text: "a,b\n1,2", type: "text/csv" },
    { name: "screenshot.png", bytes: [137, 80, 78, 71, 0, 255], type: "image/png" }];
  assert.equal(await transfer("paste", files), true);
  await page.waitForFunction(() => document.querySelectorAll(".pending-file").length === 2);
  assert.equal(fixture.uploads.length, 1);
  assert.deepEqual(fixture.uploads[0].files.map((file) => file.name), ["report.csv", "screenshot.png"]);
  assert.deepEqual(fixture.uploads[0].files[1].bytes, files[1].bytes);
  assert.equal(await page.locator("#message-input").inputValue(), "Analyze these files");
});

test("ordinary text paste still uses the native textarea behavior", async () => {
  await page.evaluate(() => navigator.clipboard.writeText("ordinary\npasted text"));
  await page.locator("#message-input").focus();
  await page.keyboard.press("Control+V");
  assert.equal(await page.locator("#message-input").inputValue(), "ordinary\npasted text");
  assert.equal(fixture.uploads.length, 0);
});

test("a screenshot pasted through the browser clipboard becomes a PNG attachment", async () => {
  const png = await page.screenshot({ clip: { x: 0, y: 0, width: 2, height: 2 } });
  await page.evaluate(async (bytes) => {
    await navigator.clipboard.write([new ClipboardItem({
      "image/png": new Blob([new Uint8Array(bytes)], { type: "image/png" }),
    })]);
  }, [...png]);
  await page.locator("#message-input").focus();
  await page.keyboard.press("Control+V");
  await page.locator(".pending-file").waitFor();
  assert.equal(fixture.uploads.length, 1);
  assert.equal(fixture.uploads[0].files[0].type, "image/png");
  assert.deepEqual(fixture.uploads[0].files[0].bytes.slice(0, 8), [137, 80, 78, 71, 13, 10, 26, 10]);
});

test("files exposed only through clipboard items can be pasted", async () => {
  await page.evaluate(() => {
    const event = new Event("paste", { bubbles: true, cancelable: true });
    Object.defineProperty(event, "clipboardData", { value: { files: [], items: [
      { kind: "file", getAsFile: () => new File(["fallback"], "notes.txt", { type: "text/plain" }) },
    ] } });
    document.querySelector("#message-input").dispatchEvent(event);
  });
  await page.locator(".pending-file").waitFor();
  assert.equal(fixture.uploads[0].files[0].name, "notes.txt");
});

test("file drops show a target, upload multiple files, and never navigate away", async () => {
  // During dragover, browsers expose the file type but protect the file bytes.
  const prevented = await page.evaluate(() => {
    const event = new Event("dragover", { bubbles: true, cancelable: true });
    Object.defineProperty(event, "dataTransfer", { value: { types: ["Files"], files: [], items: [] } });
    document.querySelector("#messages").dispatchEvent(event);
    return event.defaultPrevented;
  });
  assert.equal(prevented, true);
  assert.equal(await page.locator("#file-drop-overlay").isVisible(), true);
  assert.equal(await transfer("drop", [{ name: "one.txt" }, { name: "two.pdf", type: "application/pdf" }], "#messages"), true);
  await page.waitForFunction(() => document.querySelectorAll(".pending-file").length === 2);
  assert.equal(await page.locator("#file-drop-overlay").isVisible(), false);
  assert.equal(fixture.uploads.length, 1);
  assert.equal(fixture.uploads[0].files.length, 2);
  assert.equal(page.url(), `${baseURL}/`);
});

test("drops outside chat are blocked, text drags stay native, and canceled drags clear the target", async () => {
  assert.equal(await transfer("drop", [{ name: "outside.txt" }], ".sidebar"), true);
  assert.equal(fixture.uploads.length, 0);
  const textPrevented = await page.evaluate(() => {
    const data = new DataTransfer();
    data.setData("text/plain", "dragged text");
    const event = new DragEvent("drop", { dataTransfer: data, bubbles: true, cancelable: true });
    document.querySelector("#message-input").dispatchEvent(event);
    return event.defaultPrevented;
  });
  assert.equal(textPrevented, false);
  await transfer("dragenter", [{ name: "one.txt" }]);
  assert.equal(await page.locator("#file-drop-overlay").isVisible(), true);
  await page.keyboard.press("Escape");
  assert.equal(await page.locator("#file-drop-overlay").isVisible(), false);
});

test("folders are rejected instead of uploading a misleading empty file", async () => {
  await page.evaluate(() => {
    const event = new Event("drop", { bubbles: true, cancelable: true });
    Object.defineProperty(event, "dataTransfer", { value: { types: ["Files"], files: [], items: [
      { kind: "file", webkitGetAsEntry: () => ({ isDirectory: true }) },
    ] } });
    document.querySelector("#messages").dispatchEvent(event);
  });
  assert.equal(fixture.uploads.length, 0);
  assert.match(await page.locator("#toast").textContent(), /zip the folder/);
});

test("the picker still works and Enter cannot send before an upload finishes", async () => {
  let finish;
  fixture.nextUploadGate = new Promise((resolve) => { finish = resolve; });
  await page.locator("#message-input").fill("Review this");
  await page.locator("#file-input").setInputFiles({ name: "chosen.txt", mimeType: "text/plain", buffer: Buffer.from("picker bytes") });
  await until(() => fixture.uploads.length === 1);
  assert.equal(await page.locator("#send-button").isDisabled(), true);
  assert.match(await page.locator("#attachment-status").textContent(), /Uploading 1 file/);
  await page.locator("#message-input").press("Enter");
  assert.equal(fixture.sent.length, 0);
  finish();
  await page.locator(".pending-file").waitFor();
  assert.equal(await page.locator("#send-button").isEnabled(), true);
  await page.locator("#send-button").click();
  await until(() => fixture.sent.length === 1);
  assert.deepEqual(fixture.sent[0].attachment_ids, ["upload-1-0"]);
  await page.waitForFunction(() => document.querySelector("#message-input").value === "Review this");
  assert.equal(await page.locator(".pending-file").count(), 1, "Failed sends restore attached files");
});

test("uploads stay in their original chat when the user switches chats", async () => {
  let finish;
  fixture.nextUploadGate = new Promise((resolve) => { finish = resolve; });
  await transfer("paste", [{ name: "chat-a.txt" }]);
  await until(() => fixture.uploads.length === 1);
  await page.locator(".conversation-item").filter({ hasText: "Chat b" }).click();
  assert.equal(await page.locator("#send-button").isEnabled(), true);
  await transfer("drop", [{ name: "chat-b.txt" }]);
  await page.locator(".pending-file").waitFor();
  finish();
  await page.waitForFunction(() => state.attachmentUploads.size === 0);
  assert.deepEqual(fixture.uploads.map((upload) => upload.conversationId), ["a", "b"]);
  assert.deepEqual(await page.locator(".pending-file span").allTextContents(), ["chat-b.txt"]);
  await page.locator(".conversation-item").filter({ hasText: "Chat a" }).click();
  assert.deepEqual(await page.locator(".pending-file span").allTextContents(), ["chat-a.txt"]);
  await page.locator(".pending-file button").click();
  await page.locator(".conversation-item").filter({ hasText: "Chat b" }).click();
  await page.locator(".conversation-item").filter({ hasText: "Chat a" }).click();
  assert.equal(await page.locator(".pending-file").count(), 0, "Removed files do not reappear");
});

test("rapid paste and drop on the welcome screen share one newly created chat", async () => {
  fixture.conversations = [];
  await page.reload();
  await page.locator(".welcome").waitFor();
  await page.waitForFunction(() => state.catalog !== null);
  await page.evaluate(() => {
    for (const kind of ["paste", "drop"]) {
      const data = new DataTransfer();
      data.items.add(new File([kind], `${kind}.txt`, { type: "text/plain" }));
      const event = kind === "paste"
        ? new ClipboardEvent(kind, { clipboardData: data, bubbles: true, cancelable: true })
        : new DragEvent(kind, { dataTransfer: data, bubbles: true, cancelable: true });
      document.querySelector("#message-input").dispatchEvent(event);
      data.items.clear();
    }
  });
  await page.waitForFunction(() => document.querySelectorAll(".pending-file").length === 2);
  assert.equal(fixture.created, 1);
  assert.equal(fixture.uploads.length, 2);
  assert.deepEqual(fixture.uploads.map((upload) => upload.conversationId), ["new-1", "new-1"]);
  assert.deepEqual(fixture.uploads.map((upload) => Buffer.from(upload.files[0].bytes).toString()).sort(), ["drop", "paste"]);
});

test("an upload failure clears busy state and allows a successful retry", async () => {
  fixture.uploadStatus = 413;
  await transfer("paste", [{ name: "too-big.txt" }]);
  await page.waitForFunction(() => document.querySelector("#toast").textContent === "File too large");
  assert.equal(await page.locator("#send-button").isEnabled(), true);
  assert.equal(await page.locator("#attachment-status").isVisible(), false);
  assert.equal(await page.locator(".pending-file").count(), 0);
  fixture.uploadStatus = 200;
  await transfer("paste", [{ name: "retry.txt" }]);
  await page.locator(".pending-file").waitFor();
  assert.equal(await page.locator(".pending-file span").textContent(), "retry.txt");
});
