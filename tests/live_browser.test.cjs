// Actual browser -> actual FastAPI -> real OpenAI SDK -> simulated Azure.
// Run with the same Playwright setup as frontend_browser.test.cjs.
const { test } = require('node:test');
const assert = require('node:assert/strict');
const { spawn } = require('node:child_process');
const { mkdtemp, readFile, rm } = require('node:fs/promises');
const { tmpdir } = require('node:os');
const { join, resolve } = require('node:path');
const { createServer } = require('node:net');
const { chromium } = require('playwright');

async function freePort() {
  const s = createServer();
  await new Promise(r => s.listen(0, '127.0.0.1', r));
  const port = s.address().port;
  await new Promise(r => s.close(r));
  return port;
}
async function until(check) {
  const deadline = Date.now() + 15000;
  while (Date.now() < deadline) {
    try { if (await check()) return; } catch (_) {}
    await new Promise(r => setTimeout(r, 100));
  }
  throw new Error('Timed out waiting for fixture');
}

test('real downloads, interrupted Azure stream, and active-process restart retain the chat and files', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'chat-live-'));
  const port = await freePort();
  const base = `http://127.0.0.1:${port}`;
  let server, browser, logs = '';
  async function launch(stall = false) {
    server = spawn(process.env.PYTHON || 'python', [resolve(__dirname, 'browser_api_fixture.py'), String(port)], {
      cwd: resolve(__dirname, '..'), env: { ...process.env, APP_DATA_DIR: directory, FIXTURE_STALL: stall ? '1' : '0' },
      stdio: ['ignore', 'pipe', 'pipe'],
    });
    server.stdout.on('data', chunk => logs += chunk);
    server.stderr.on('data', chunk => logs += chunk);
    await until(async () => (await fetch(`${base}/api/health`)).ok);
  }
  async function stop() {
    if (!server || server.exitCode !== null) return;
    const exited = new Promise(r => server.once('exit', r));
    server.kill('SIGKILL'); // Deliberate crash of our isolated test process.
    await exited;
  }
  try {
    await launch();
    browser = await chromium.launch({ executablePath: process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH || undefined,
      args: process.env.PLAYWRIGHT_CHROMIUM_ARGS ? JSON.parse(process.env.PLAYWRIGHT_CHROMIUM_ARGS) : ['--no-sandbox'] });
    const page = await browser.newPage();
    const errors = [];
    page.on('pageerror', error => errors.push(error.message));
    await page.route('https://cdn.jsdelivr.net/**', route => route.fulfill({ body: '' }));
    await page.goto(base);
    const link = page.locator('.message-download-link').first();
    await link.waitFor();
    let pending = page.waitForEvent('download');
    await link.click();
    const original = await pending;
    const zip = await readFile(await original.path());
    assert.equal(zip.subarray(0, 4).toString('hex'), '504b0304');
    assert.equal(original.suggestedFilename(), 'report.zip');
    let requests = await (await fetch(`${base}/test/requests`)).json();
    assert.ok(requests.length > 0 && requests.every(r => r.method === 'GET'));
    await page.locator('#new-chat').click();
    await page.locator('#messages .welcome').waitFor();
    await page.locator('#message-input').fill('Generate a file after a simulated stream error');
    await page.locator('#message-input').press('Enter');
    await page.locator('.message-download-link').waitFor();
    await until(async () => !(await page.locator('.streaming-message').count()));
    requests = await (await fetch(`${base}/test/requests`)).json();
    assert.equal(requests.filter(r => r.method === 'POST' && r.path.endsWith('/responses')).length, 1);
    assert.ok(requests.some(r => r.method === 'GET' && r.path.endsWith('/responses/resp-running')));
    pending = page.waitForEvent('download');
    await page.locator('.message-download-link').click();
    assert.deepEqual(await readFile(await (await pending).path()), zip);

    // A crash in a new active request must attach to its saved Azure ID.
    await stop();
    await launch(true);
    await page.reload();
    await page.locator('#new-chat').click();
    await page.locator('#messages .welcome').waitFor();
    await page.locator('#message-input').fill('Keep working while the app restarts');
    await page.locator('#message-input').press('Enter');
    await until(async () => {
      const rows = await (await fetch(`${base}/test/checkpoint`)).json();
      return rows.some(r => r.response_id === 'resp-running' && r.files > 0);
    });
    const cid = await page.evaluate(() => state.activeConversationId);
    await stop();
    await launch();
    // Leave this browser tab open: its reconnect path must finish the same chat.
    await until(async () => {
      const chat = await (await fetch(`${base}/api/conversations/${cid}`)).json();
      return chat.messages.some(m => m.role === 'assistant');
    });
    await until(async () => await page.locator('.message-download-link').count() > 0 && !await page.locator('.streaming-message').count());
    const chat = await (await fetch(`${base}/api/conversations/${cid}`)).json();
    assert.equal(chat.messages.filter(m => m.role === 'assistant').length, 1);
    assert.equal(chat.messages.at(-1).metadata.provider_status, 'completed');
    requests = await (await fetch(`${base}/test/requests`)).json();
    assert.equal(requests.filter(r => r.method === 'POST' && r.path.endsWith('/responses')).length, 0);
    pending = page.waitForEvent('download');
    await page.locator('.message-download-link').click();
    assert.deepEqual(await readFile(await (await pending).path()), zip);
    assert.deepEqual(errors, []);
  } catch (error) {
    throw new Error(`${error.stack}\nFixture logs:\n${logs}`, { cause: error });
  } finally {
    await browser?.close();
    await stop();
    await rm(directory, { recursive: true, force: true });
  }
});
