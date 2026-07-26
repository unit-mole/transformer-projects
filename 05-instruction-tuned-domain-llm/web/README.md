---
title: ML Data Science Instruction Tuned Assistant
emoji: 🤖
colorFrom: indigo
colorTo: cyan
sdk: static
app_build_command: npm run build
app_file: dist/index.html
pinned: false
license: mit
---

# ML & Data Science Instruction-Tuned Assistant

A browser-based Transformer demonstration for Project 05. The application uses Transformers.js and ONNX Runtime Web to run a FLAN-T5 text-to-text generation model directly in the visitor's browser.

## Real browser inference

No Python inference server or hidden API is required. The default demo loads the browser-compatible `Xenova/flan-t5-small` base model. After the Project 05 LoRA adapter is trained, merged, exported to ONNX, and uploaded to the Hub, visitors can load that custom model repository from the interface.

## Responsible use

This educational portfolio demo can generate incorrect, outdated, biased, incomplete, or hallucinated explanations. It is not intended for legal, medical, financial, immigration, safety-critical, or official decision-making. Do not enter private, confidential, proprietary, copyrighted, or personally identifiable information.

## GitHub project

Replace the link in `src/config.js` with the final GitHub repository URL after publishing.
