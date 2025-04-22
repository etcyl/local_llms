
# Local Coding LLM Chat

## Overview
This project implements a local coding assistant powered by multiple Large Language Models (LLMs), including Claude, CodeGemma, CodeLlama, Deepseek Coder, Mistral, and Phi. It features a Streamlit frontend, supports real-time interaction, web search integration, and detailed hardware monitoring.

---

## Prerequisites
- Windows Subsystem for Linux (WSL)
- Python 3.8+ installed on WSL
- CUDA-compatible GPU (optional, recommended for performance)

---

## Installation

### Step 1: Open WSL Terminal

Open WSL terminal (Ubuntu recommended).

### Step 2: Clone Repository

```bash
git clone <your-repo-url>
cd <repo-name>
```

### Step 3: Create Virtual Environment

You can use either Python's built-in `venv` or Anaconda/Miniconda:

**Using venv:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**Using Conda:**
```bash
conda create -n local_llm python=3.10
conda activate local_llm
```

### Step 4: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Make sure the `requirements.txt` file includes:
```text
streamlit
requests
beautifulsoup4
psutil
torch
```

---

## Running the App

### Step 1: Start LLM Backend Servers

Ensure that your LLM backend servers (e.g., Ollama or Claude) are running locally:

- Ollama (for CodeGemma, CodeLlama, Deepseek Coder, Mistral, Phi):
```bash
ollama serve
```

- Claude (if using):
Follow Claude's setup documentation to run locally.

### Step 2: Run Streamlit App

Activate your virtual environment first:

```bash
source venv/bin/activate # or conda activate local_llm
```

Run Streamlit:
```bash
streamlit run app.py
```

The app will launch in your default web browser.

---

## Features
- Real-time coding assistance
- Multiple LLM models with adjustable settings (temperature, tokens)
- Web search integration for additional context
- Interactive chat interface with logging and autosave functionality
- System monitoring (CPU/GPU/memory usage)

---

## Hardware Recommendations
- GPU: NVIDIA RTX series recommended for optimal performance
- RAM: 16GB+ recommended
- CPU: Modern Intel or AMD CPU

---

## Troubleshooting

### Web Search Not Working
- Ensure internet connectivity from WSL (`ping google.com`).
- Check firewall settings on Windows to allow WSL network access.

### GPU Not Detected
- Ensure CUDA drivers are correctly installed in WSL (via WSLg or WSL2 with CUDA support).
- Run `torch.cuda.is_available()` in Python to confirm GPU availability.

---
