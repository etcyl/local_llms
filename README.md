
# Local Coding LLM Chat

<!-- Demo Videos -->
<iframe width="560" height="315" src="https://www.youtube.com/embed/Fh-Pue_GVPM?si=jQUgdoB3OzTPboFo" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

<iframe width="560" height="315" src="https://www.youtube.com/embed/C3C3QQGjuoo" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

## Overview
This project provides a local coding assistant powered by multiple Large Language Models (LLMs), including:
- Claude
- CodeGemma
- CodeLlama
- Deepseek Coder
- Mistral
- Phi

**Key features:**
- Streamlit-based frontend for interactive chat
- Real-time coding assistance
- Web search integration for contextual references
- Hardware monitoring (CPU, GPU, memory)
- Adjustable model settings (temperature, token limits)

## Prerequisites
- **Operating System:** Windows Subsystem for Linux (WSL) (Ubuntu recommended)  
- **Python:** 3.8 or higher  
- **GPU (optional):** CUDA-compatible (NVIDIA) for improved performance

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/etcyl/local_llms.git
   cd local_llms
   ```

2. **Set up a Python environment:**
   - _Using venv:_
     ```bash
     python3 -m venv venv
     source venv/bin/activate
     ```
   - _Using Conda:_
     ```bash
     conda create -n local_llm python=3.10
     conda activate local_llm
     ```

3. **Install dependencies:**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
   > Ensure `requirements.txt` includes:
   > ```txt
   > streamlit
   > requests
   > beautifulsoup4
   > psutil
   > torch
   > ```

## Running the Application

1. **Start backend servers:**
   - **Ollama:**  
     ```bash
     ollama serve
     ```
   - **Claude:**  
     Follow Claude's local setup guide.

2. **Launch Streamlit app:**
   ```bash
   streamlit run app.py
   ```
   The app will open in your default browser at `http://localhost:8501`.

## Features
- Real-time coding support with multiple LLMs
- Web search integration for enhanced context
- Interactive chat with autosave and logging
- System resource monitoring dashboard

## Hardware Recommendations
- **GPU:** NVIDIA RTX series  
- **RAM:** 16 GB or more  
- **CPU:** Modern Intel or AMD processor

## Troubleshooting

### Web Search Issues
- Verify internet access from WSL:
  ```bash
  ping google.com
  ```
- Check Windows firewall settings for WSL network.

### GPU Not Detected
- Confirm CUDA drivers in WSL2:
  ```python
  import torch
  print(torch.cuda.is_available())
  ```

## Contributing
_Use this template to add guidelines for contributing:_

1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/awesome-feature`).
3. Commit your changes (`git commit -m "Add awesome feature"`).
4. Push to the branch (`git push origin feature/awesome-feature`).
5. Open a pull request.

## License
_Specify the project's license here (e.g., MIT, Apache 2.0)._

## Acknowledgments
- [Ollama](https://ollama.com)
- [Anthropic Claude](https://www.anthropic.com/product/claude)
- [Streamlit](https://streamlit.io)

## Additional Information
_Template for extra sections:_
- **Security:** Describe security considerations.
- **Performance Tuning:** Tips to optimize performance.
- **FAQ:** Frequently Asked Questions and answers.
- **Contact:** Maintainer's contact information.
