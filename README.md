# HZ's AI Notes

> **Deep Dive into LLM Systems & Engineering**
> From Training to Inference & Agents.

This repository hosts my personal knowledge base and engineering notes on Large Language Models, built with [VitePress](https://vitepress.dev/).

## 📚 Content Structure

The documentation is organized into three main pillars:

1.  **AI Training** (`/training/`)
    *   SFT (Supervised Fine-Tuning)
    *   LoRA / QLoRA
    *   RLHF / Alignment
2.  **LLM Inference** (`/`)
    *   Generation Parameters & Sampling
    *   KV Cache Optimizations (GQA, MQA, PagedAttention)
    *   Parallelism Strategies (TP, PP, SP)
    *   SGLang & Serving Techniques
3.  **AI Agents** (`/agent/`)
    *   Planning & Reasoning
    *   Tool Use & Memory

## 🛠️ Tech Stack

*   **Framework**: [VitePress](https://vitepress.dev/)
*   **Diagrams**: [Mermaid.js](https://mermaid.js.org/) (via `vitepress-plugin-mermaid`)
*   **Math**: MathJax (via native VitePress support)

## 🚀 Local Development

### 1. Install Dependencies
```bash
npm install
```

### 2. Run Dev Server
```bash
npm run docs:dev
```
The site will differ at `http://localhost:5173`.

### 3. Build for Production
```bash
npm run docs:build
```

## 📦 Deployment

This project is configured to automatically deploy to **GitHub Pages** via GitHub Actions.
See `.github/workflows/deploy.yml` for the workflow configuration.
