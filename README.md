
# Local Coding LLM Chat

[![Demo 1](https://img.youtube.com/vi/Fh-Pue_GVPM/hqdefault.jpg)](https://youtu.be/Fh-Pue_GVPM?si=jQUgdoB3OzTPboFo)
[![Demo 2](https://img.youtube.com/vi/C3C3QQGjuoo/hqdefault.jpg)](https://www.youtube.com/watch?v=C3C3QQGjuoo)

## Overview
This project provides a local coding assistant powered by multiple Large Language Models (LLMs), including:
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
Use this template to add guidelines for contributing:
1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/`).
3. Commit your changes (`git commit -m "Add feature"`).
4. Push to the branch (`git push origin feature/`).
5. Open a pull request.

## License
Specify the project's license here (e.g., MIT, Apache 2.0).

## Acknowledgments
- [Ollama](https://ollama.com)
- [Anthropic Claude](https://www.anthropic.com/product/claude)
- [Streamlit](https://streamlit.io)

## Performance Tuning
Tips to optimize application performance:
- Use a CUDA-compatible GPU for model inference  
- Adjust Streamlit's `server.maxMessageSize` for large payloads  
- Profile and optimize Python code with `cProfile` or `line_profiler`  

## FAQ
**Q:** Can I run this without a GPU?  
**A:** Yes, but performance will be limited to CPU speeds.

**Q:** How do I add a new LLM?  
**A:** Update `MODEL_INFOS` in `app.py` and ensure the model server is running locally.
