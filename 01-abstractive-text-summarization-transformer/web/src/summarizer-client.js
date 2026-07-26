export class SummarizerClient {
  constructor({ onProgress, onStatus } = {}) {
    this.onProgress = onProgress ?? (() => {});
    this.onStatus = onStatus ?? (() => {});
    this.worker = null;
    this.requests = new Map();
    this.nextRequestId = 1;
    this.runtimePreference = null;
  }

  ensureWorker() {
    if (this.worker) return;

    this.worker = new Worker(new URL('./model-worker.js', import.meta.url), { type: 'module' });
    this.worker.addEventListener('message', (event) => this.handleMessage(event.data));
    this.worker.addEventListener('error', (event) => {
      const error = new Error(event.message || 'The model worker stopped unexpectedly.');
      for (const pending of this.requests.values()) pending.reject(error);
      this.requests.clear();
      this.onStatus({ state: 'error', message: error.message });
    });
  }

  handleMessage(message) {
    if (message.type === 'progress') {
      this.onProgress(message.payload);
      return;
    }
    if (message.type === 'status') {
      this.onStatus(message.payload);
      return;
    }

    const pending = this.requests.get(message.requestId);
    if (!pending) return;

    if (message.type === 'result') {
      pending.resolve(message.payload);
      this.requests.delete(message.requestId);
    } else if (message.type === 'error') {
      pending.reject(new Error(message.payload?.message || 'Inference failed.'));
      this.requests.delete(message.requestId);
    }
  }

  request(action, payload = {}) {
    this.ensureWorker();
    const requestId = this.nextRequestId++;

    return new Promise((resolve, reject) => {
      this.requests.set(requestId, { resolve, reject });
      this.worker.postMessage({ requestId, action, payload });
    });
  }

  async load(runtimePreference = 'auto') {
    this.runtimePreference = runtimePreference;
    return this.request('load', { runtimePreference });
  }

  async summarize(payload) {
    return this.request('summarize', payload);
  }

  async compareBeams(payload) {
    return this.request('compare-beams', payload);
  }

  reset() {
    if (this.worker) this.worker.terminate();
    this.worker = null;
    this.requests.clear();
    this.runtimePreference = null;
  }
}
