export class BrowserModelClient {
  constructor({ onProgress, onStatus } = {}) {
    this.worker = new Worker(new URL('./model-worker.js', import.meta.url), { type: 'module' });
    this.pending = new Map();
    this.onProgress = onProgress ?? (() => {});
    this.onStatus = onStatus ?? (() => {});
    this.worker.addEventListener('message', (event) => this.handleMessage(event.data));
  }

  handleMessage(message) {
    if (message.type === 'progress') {
      this.onProgress(message.progress);
      return;
    }
    if (message.type === 'status') {
      this.onStatus(message.message);
      return;
    }

    const pending = this.pending.get(message.id);
    if (!pending) return;

    if (message.type === 'loaded') {
      this.pending.delete(message.id);
      pending.resolve(message.metadata);
    } else if (message.type === 'result') {
      this.pending.delete(message.id);
      pending.resolve(message.result);
    } else if (message.type === 'error') {
      this.pending.delete(message.id);
      pending.reject(new Error(message.error.message));
    }
  }

  request(action, payload) {
    const id = crypto.randomUUID();
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
      this.worker.postMessage({ id, action, payload });
    });
  }

  load(runtime) {
    return this.request('load', runtime);
  }

  generate(prompt, settings) {
    return this.request('generate', { prompt, settings });
  }

  dispose() {
    this.worker.terminate();
    this.pending.clear();
  }
}
