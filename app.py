#!/usr/bin/env python3
"""
Streamlit Coding-LLM Chat – v2.4
------------------------------------------------
New
• Timer now starts the moment the user clicks **Send**.
• Elapsed time is stored and rendered under each model reply in history:
  “Model took X s to reply.”
Fixes
• Internal chat-history tuples expanded to carry the elapsed-seconds value
  while remaining backward-compatible with existing sessions.
• Live status bar still shows the running timer during generation.
"""

import streamlit as st
import requests, json, time, os, gc, atexit, threading, queue
from datetime import datetime
from bs4 import BeautifulSoup
import pandas as pd
import psutil, torch

# ---------- Optional JAX ----------
try:
    import jax, jax.numpy as jnp        # noqa: F401
    HAS_JAX = True
except ImportError:
    HAS_JAX = False

# ---------- CONFIG ----------
OLLAMA_ENDPOINT  = "http://localhost:11434/api/generate"
CLAUDE_ENDPOINT  = "http://localhost:8000/api/claude"
MAX_CPU_PCT, MAX_MEM_FRAC, MAX_GPU_FRAC = 85.0, 0.85, 0.85

MODEL_INFOS = {
    "codegemma":      "CodeGemma is a compact and lightweight model designed for fast code synthesis, built on Google's Gemma architecture.",
    "codellama":      "CodeLlama is Meta's code generation model, fine-tuned on code-specific tasks and known for its efficiency.",
    "deepseek-coder": "Deepseek Coder is a powerful code model trained with billions of tokens, designed for multi-language code generation and reasoning.",
    "mistral":        "Mistral is a general-purpose small LLM that performs well across many tasks, including code and math reasoning.",
    "phi":            "Phi is a lightweight transformer model developed by Microsoft with impressive performance on code and QA tasks."
}

DEFAULT_PARAMS = {
    "codegemma":      (0.8, 800),
    "codellama":      (0.6, 1024),
    "deepseek-coder": (0.7, 1024),
    "mistral":        (0.9, 1024),
    "phi":            (0.75, 900)
}

ALL_AVAILABLE_MODELS = sorted(MODEL_INFOS.keys())
LANGUAGES            = sorted(["C","C#","C++","Go","Java","JavaScript","Python","Rust"])

st.set_page_config(page_title="💬 Local Coding LLM Chat", layout="wide")

# ---------- SAFE RERUN ----------
def _safe_rerun():
    if hasattr(st, "rerun"):
        st.rerun()
    else:  # Streamlit < 1.31
        st.experimental_rerun()

# ---------- SAFETY ----------
def _gpu_usage_frac() -> float:
    if not torch.cuda.is_available():
        return 0.0
    props = torch.cuda.get_device_properties(0)
    return torch.cuda.memory_allocated(0) / props.total_memory

def _safe() -> bool:
    mem = psutil.virtual_memory().percent / 100.0
    return (psutil.cpu_percent(0.25) <= MAX_CPU_PCT
            and mem <= MAX_MEM_FRAC
            and _gpu_usage_frac() <= MAX_GPU_FRAC)

if HAS_JAX:
    @jax.jit
    def _adj_tokens(max_t, cpu_pct):
        import jax.numpy as jnp
        scale = jnp.clip(1.0 - cpu_pct/100.0, 0.3, 1.0)
        return jnp.floor(max_t*scale).astype(int)
else:
    def _adj_tokens(max_t, cpu_pct):
        from math import floor
        return floor(max_t*max(0.3, 1.0 - cpu_pct/100.0))

def _throttle():
    while not _safe():
        time.sleep(0.5)

def _cleanup():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

atexit.register(_cleanup)

# ---------- SESSION ----------
st.session_state.setdefault(
    "system_prompt",
    "You are a code assistant. Return idiomatic, well-formatted code in the specified language. Explain nothing unless asked."
)
st.session_state.setdefault("chat_history", [])          # [(q, a, qts, mdl, ats, elapsed)]
st.session_state.setdefault("temperature", DEFAULT_PARAMS["mistral"][0])
st.session_state.setdefault("max_tokens",  DEFAULT_PARAMS["mistral"][1])
st.session_state.setdefault("is_generating", False)
st.session_state.setdefault("gen_start_time", None)      # ⏱️ NEW
st.session_state.setdefault("hw_history", {"t":[], "cpu":[], "mem":[], "gpu":[]})
st.session_state.setdefault("autosave", False)
st.session_state.setdefault("enable_web_search", False)
st.session_state.setdefault("user_prompt", "")

# ---------- Connectivity ----------
internet_online = True
try:
    requests.get("https://www.google.com", timeout=2)
except Exception:
    internet_online = False

# ---------- Helpers ----------
def format_out(txt: str, mdl: str) -> str:
    if txt.strip().startswith("```"):
        return txt
    lang = "python" if mdl.split("-")[0] in {"deepseek", "code", "phi", "mistral"} else "text"
    return f"```{lang}\n{txt.strip()}\n```"

def stream_response(endpoint: str, payload: dict):
    try:
        r = requests.post(endpoint, json=payload, stream=True, timeout=600)
        if r.status_code != 200:
            yield f"[HTTP {r.status_code}] {r.text}"; return
        buf = ""
        for ln in r.iter_lines():
            if not ln:
                continue
            buf += json.loads(ln.decode()).get("response", "")
            yield buf
    except Exception as e:
        yield f"[EXCEPTION] {e}"

def web_search(query: str, num_results: int = 5):
    results = []; headers = {"User-Agent": "Mozilla/5.0"}
    url = f"https://www.google.com/search?q={query.replace(' ','+')}+site:stackoverflow"
    try:
        soup = BeautifulSoup(requests.get(url, headers=headers, timeout=5).text, "html.parser")
        for g in soup.select("div.tF2Cxc")[:num_results]:
            a = g.select_one("div.yuRUbf a"); s = g.select_one("div.VwiC3b")
            if a and s:
                results.append({
                    "snippet": s.get_text(' ', strip=True)[:500],
                    "link":    a['href']
                })
    except Exception:
        pass
    return results

# ---------- STYLE ----------
st.markdown("""
<style>
.bubble-user,.bubble-bot{padding:10px;margin:8px 0;border-radius:10px;max-width:80%;}
.bubble-user{background:#dcf8c6;} .bubble-bot{background:#e1ecf4;margin-left:40px;}
.avatar{font-size:24px;margin-right:8px;vertical-align:middle;}
</style>
""", unsafe_allow_html=True)

# ---------- SIDEBAR ----------
with st.sidebar:
    st.title("💬 Local Coding LLM Chat")

    with st.expander("💻 Hardware", expanded=True):
        st.markdown(f"**Internet:** {'🟢 Online' if internet_online else '🔴 Offline'}")
        st.markdown(f"**JAX Acceleration:** {'🟢 Active' if HAS_JAX else '🔴 Not available'}")
        cpu_pct = psutil.cpu_percent(0.2); mem_pct = psutil.virtual_memory().percent
        gpu_pct = _gpu_usage_frac() * 100
        hist = st.session_state.hw_history
        hist["t"].append(datetime.now())
        hist["cpu"].append(cpu_pct); hist["mem"].append(mem_pct); hist["gpu"].append(gpu_pct)
        if len(hist["t"]) > 60:
            for k in hist:
                hist[k] = hist[k][-60:]
        df = pd.DataFrame({"CPU %": hist["cpu"],
                           "RAM %": hist["mem"],
                           "GPU %": hist["gpu"]},
                          index=hist["t"])
        st.line_chart(df, use_container_width=True, height=150)

    with st.expander("🧐 Model Settings", expanded=True):
        mdl  = st.selectbox("Model", ALL_AVAILABLE_MODELS)
        lang = st.selectbox("Language", LANGUAGES)
        t0, m0 = DEFAULT_PARAMS[mdl]
        st.session_state.temperature = st.slider("Temperature", 0.0, 1.5, t0, 0.05,
                                                 help="Randomness control.")
        st.session_state.max_tokens  = st.slider("Max Tokens", 100, 4000, m0, 50,
                                                 help="Generation upper bound.")

    with st.expander("⚙️ Prompt / Options"):
        st.text_area("System Prompt",
                     st.session_state.system_prompt,
                     key="system_prompt", height=120)
        st.checkbox("Autosave", key="autosave")
        st.checkbox("Enable Web Search", key="enable_web_search",
                    disabled=not internet_online)

# ---------- CHAT HISTORY ----------
chat_area = st.container()
for rec in st.session_state.chat_history:
    # Back-compat if older tuples lack elapsed-time
    if len(rec) == 6:
        q, a, qt, md, at, elapsed = rec
    else:
        q, a, qt, md, at = rec
        elapsed = None
    chat_area.markdown(
        f"<div class='bubble-user'><span class='avatar'>👤</span><b>You</b><br>({qt})<br>{q}</div>",
        unsafe_allow_html=True)
    chat_area.markdown(
        f"<div class='bubble-bot'><span class='avatar'>🤖</span><b>{md}</b><br>({at})</div>",
        unsafe_allow_html=True)
    chat_area.markdown(format_out(a, md))
    if elapsed is not None:
        chat_area.markdown(f"*Model took&nbsp;{elapsed:.1f}&nbsp;s to reply.*")

# ---------- INPUT & BUTTONS ----------
st.divider()
input_ph = st.empty()
btn_col  = st.columns([1, 1, 4])

def _recall_last():
    if st.session_state.chat_history:
        st.session_state.user_prompt = st.session_state.chat_history[-1][0]

user_prompt = input_ph.text_area("Your Prompt",
                                 key="user_prompt", height=120)

send_clicked = btn_col[0].button("📤 Send",
                                 disabled=st.session_state.is_generating)
btn_col[1].button("↺ Last Query",
                  disabled=st.session_state.is_generating,
                  on_click=_recall_last)
status_ph = btn_col[2].empty()

# ---------- MAIN LOGIC ----------
if send_clicked and user_prompt.strip():
    st.session_state.is_generating  = True
    st.session_state.gen_start_time = time.time()     # ⏱️ start timer immediately
    _safe_rerun()

if st.session_state.is_generating:
    _throttle()
    cpu_now = psutil.cpu_percent(0.1)
    dyn_tok = max(100, int(_adj_tokens(st.session_state.max_tokens, cpu_now)))

    query = st.session_state.user_prompt.strip()
    prompt_parts = []
    if st.session_state.enable_web_search and internet_online:
        ctx = web_search(query)
        if ctx:
            prompt_parts.append("Web Context:\n" + "\n\n".join(c["snippet"] for c in ctx))
    prompt_parts.append(f"User Query:\n{query}")
    full_prompt = st.session_state.system_prompt + "\n\n".join(prompt_parts)

    endpoint = CLAUDE_ENDPOINT if "claude" in mdl else OLLAMA_ENDPOINT
    payload  = {"model": mdl, "prompt": full_prompt, "stream": True,
                "temperature": st.session_state.temperature,
                "options": {"num_predict": dyn_tok}}

    qts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    chat_area.markdown(
        f"<div class='bubble-user'><span class='avatar'>👤</span><b>You</b><br>({qts})<br>{query}</div>",
        unsafe_allow_html=True)

    head_ph, cont_ph = chat_area.empty(), chat_area.empty()
    q = queue.Queue()
    threading.Thread(
        target=lambda: [q.put(x) for x in stream_response(endpoint, payload)] + [q.put(None)],
        daemon=True
    ).start()

    buf = ""
    while True:
        chunk = q.get()
        if chunk is None:
            break
        buf = chunk
        ats = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        head_ph.markdown(
            f"<div class='bubble-bot'><span class='avatar'>🤖</span><b>{mdl}</b><br>({ats})</div>",
            unsafe_allow_html=True)
        cont_ph.markdown(format_out(buf, mdl))
        elapsed = time.time() - st.session_state.gen_start_time
        status_ph.info(f"⏱ {elapsed:.1f}s | CPU {psutil.cpu_percent(0):.0f}%")

    # -------- persist history with elapsed-time --------
    total_elapsed = time.time() - st.session_state.gen_start_time
    st.session_state.chat_history.append((query, buf, qts, mdl, ats, total_elapsed))
    st.session_state.is_generating  = False
    st.session_state.gen_start_time = None
    _cleanup()
    _safe_rerun()
