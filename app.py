import streamlit as st
import requests
import json
import time
import os
import torch
import psutil
from datetime import datetime
from bs4 import BeautifulSoup

# -----------------------------
# CONFIG
# -----------------------------
OLLAMA_ENDPOINT = "http://localhost:11434/api/generate"
CLAUDE_ENDPOINT  = "http://localhost:8000/api/claude"

MODEL_INFOS = {
    ##"anthropic-claude": "Claude from Anthropic is known for safety‑first design and strong performance in structured and creative tasks, including code generation.",
    "codegemma":       "CodeGemma is a compact and lightweight model designed for fast code synthesis, built on Google's Gemma architecture.",
    "codellama":       "CodeLlama is Meta's code generation model, fine‑tuned on code‑specific tasks and known for its efficiency.",
    "deepseek-coder":  "Deepseek Coder is a powerful code model trained with billions of tokens, designed for multi‑language code generation and reasoning.",
    "mistral":         "Mistral is a general‑purpose small LLM that performs well across many tasks, including code and math reasoning.",
    "phi":             "Phi is a lightweight transformer model developed by Microsoft with impressive performance on code and QA tasks."
}

DEFAULT_PARAMS = {
    ##"anthropic-claude": (0.5, 1500),
    "codegemma":        (0.8, 800),
    "codellama":        (0.6, 1024),
    "deepseek-coder":   (0.7, 1024),
    "mistral":          (0.9, 1024),
    "phi":              (0.75, 900)
}

ALL_AVAILABLE_MODELS = sorted(MODEL_INFOS.keys())
LANGUAGES           = sorted(["C", "C#", "C++", "Go", "Java", "JavaScript", "Python", "Rust"])

st.set_page_config(page_title="💬 Local Coding LLM Chat", layout="wide")

# -----------------------------
# INTERNET CONNECTIVITY CHECK
# -----------------------------
try:
    requests.get("https://www.google.com", timeout=2)
    internet_online = True
except:
    internet_online = False

# -----------------------------
# SESSION STATE INIT
# -----------------------------
st.session_state.setdefault("chat_history", [])
st.session_state.setdefault("system_prompt",
    "You are a code assistant. Return idiomatic, well‑formatted code in the specified language. Explain nothing unless asked."
)
st.session_state.setdefault("autosave", False)
st.session_state.setdefault("temperature", DEFAULT_PARAMS["mistral"][0])
st.session_state.setdefault("max_tokens", DEFAULT_PARAMS["mistral"][1])
st.session_state.setdefault("message_duration", 0.0)
st.session_state.setdefault("enable_web_search", False)

# -----------------------------
# HELPERS
# -----------------------------
def format_output(text, model):
    t = text.strip()
    if t.startswith("```"):
        return t
    base = model.lower().split("-")[0]
    lang = "python" if base in ["deepseek","code","phi","mistral"] else "text"
    return f"```{lang}\n{t}\n```"


def stream_response(endpoint, payload):
    try:
        resp = requests.post(endpoint, json=payload, stream=True, timeout=600)
        if resp.status_code != 200:
            yield f"**[HTTP ERROR]** {resp.status_code}: {resp.text}"
            return
        buf = ""
        for line in resp.iter_lines():
            if not line:
                continue
            try:
                piece = json.loads(line.decode("utf-8")).get("response", "")
            except:
                continue
            buf += piece
            yield buf
    except Exception as e:
        yield f"**[EXCEPTION]** {e}"

# Function perform web search and extract snippets + links
# Improved Web Search Function (Google and Stack Overflow)
import requests
from bs4 import BeautifulSoup

# Improved Web Search Function (Google and Stack Overflow)
def web_search(query, num_results=5):
    results = []
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
    }

    google_url = f'https://www.google.com/search?q={query.replace(" ", "+")}+site:stackoverflow'
    
    try:
        google_resp = requests.get(google_url, headers=headers, timeout=5)
        google_soup = BeautifulSoup(google_resp.text, 'html.parser')
        for g in google_soup.select('div.tF2Cxc')[:num_results]:
            link_tag = g.select_one('div.yuRUbf a')
            snippet_tag = g.select_one('div.VwiC3b')
            if link_tag and snippet_tag:
                link = link_tag['href']
                snippet = snippet_tag.get_text(separator=' ', strip=True)
                results.append({"snippet": snippet[:500], "link": link})
    except Exception as e:
        print(f"Google search error: {e}")

    # Stack Overflow API fallback
    if len(results) < num_results:
        try:
            params = {
                "order": "desc",
                "sort": "relevance",
                "q": query,
                "site": "stackoverflow",
                "pagesize": num_results - len(results),
                "filter": "withbody"
            }
            resp = requests.get("https://api.stackexchange.com/2.3/search/advanced", params=params, timeout=5)
            data = resp.json()
            for item in data.get("items", []):
                link = item.get("link")
                body_html = item.get("body", "")
                if body_html and link:
                    soup = BeautifulSoup(body_html, "html.parser")
                    text = soup.get_text(separator=" ", strip=True)
                    results.append({"snippet": text[:500], "link": link})
        except Exception as e:
            print(f"Stack Overflow API error: {e}")

    return results

# -----------------------------
# STYLES
# -----------------------------
st.markdown("""
<style>
.bubble-user, .bubble-bot {
  padding: 10px; margin: 8px 0; border-radius: 10px; max-width: 80%;
}
.bubble-user { background: #dcf8c6; }
.bubble-bot  { background: #e1ecf4; margin-left: 40px; }
.avatar { font-size: 24px; margin-right: 8px; vertical-align: middle; }
.timer { font-size: 0.9em; color: #555; margin-bottom: 8px; }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# SIDEBAR
# -----------------------------
with st.sidebar:
    st.title("💬 Local Coding LLM Chat")
    with st.expander("💻 Hardware Settings", expanded=True):
        st.markdown(f"**Internet:** {'🟢 Online' if internet_online else '🔴 Offline'}")
        try:
            cpu = os.popen("lscpu | grep 'Model name'").read().split(":")[1].strip()
            gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None"
            mem = psutil.virtual_memory()
            st.metric("CPU Usage %", f"{psutil.cpu_percent(1):.1f}%")
            st.metric("Memory GB", f"{mem.used/1024**3:.1f}/{mem.total/1024**3:.1f}")
            st.metric("GPU Available", "✅ Yes" if torch.cuda.is_available() else "❌ No")
            st.markdown(f"**CPU Model:** {cpu}")
            st.markdown(f"**GPU Model:** {gpu}")
        except Exception as e:
            st.error(f"Hardware Info Error: {e}")

    with st.expander("🧐 Model Settings", expanded=True):
        selected_model    = st.selectbox("Model", ALL_AVAILABLE_MODELS)
        st.markdown(f"**Info:** {MODEL_INFOS[selected_model]}")
        selected_language = st.selectbox("Language", LANGUAGES)
        t0, m0            = DEFAULT_PARAMS[selected_model]
        st.session_state.temperature = st.slider("Temperature", 0.0, 1.5, t0, 0.05)
        st.session_state.max_tokens  = st.slider("Max Tokens", 100, 4000, m0, 50)

    with st.expander("⚙️ System Settings", expanded=True):
        st.checkbox("Autosave Replies", key="autosave")
        st.text_area("Edit System Prompt", st.session_state.system_prompt, height=150, key="system_prompt")
        log = "\n\n".join(
            f"You ({u_ts}):\n{q}\n\nBot ({a_ts}):\n{a}"
            for q, a, u_ts, m, a_ts in st.session_state.chat_history
        )
        st.download_button("Download Chat Log", data=log, file_name="chat_history.txt")
        if internet_online:
            st.checkbox("Enable Web Search Context", key="enable_web_search", help="Include relevant web search snippets as context.")

# -----------------------------
# UI LOADING SCREEN
# -----------------------------
def show_loading_screen():
    placeholder = st.empty()
    progress = placeholder.progress(0)
    for pct in range(101):
        remaining = 100 - pct
        placeholder.markdown(f"### 🐢 Loading... **{pct}%** done — only **{remaining}%** to go! 🦄")
        placeholder.image("https://media.giphy.com/media/3oEjI6SIIHBdRxXI40/giphy.gif", width=150)
        progress.progress(pct)
        time.sleep(0.02)
    placeholder.empty()

show_loading_screen()

# -----------------------------
# MAIN CONTENT
# -----------------------------
chat_area = st.container()
for q, a, q_ts, m, a_ts in st.session_state.chat_history:
    chat_area.markdown(
        f"<div class='bubble-user'>"
        f"<span class='avatar'>👤</span><b>You</b><br>({q_ts})<br>{q}"
        f"</div>", unsafe_allow_html=True
    )
    chat_area.markdown(
        f"<div class='bubble-bot'>"
        f"<span class='avatar'>🤖</span><b>{m}</b><br>({a_ts})"
        f"</div>", unsafe_allow_html=True
    )
    chat_area.markdown(format_output(a, m))

# -----------------------------
# LIVE STATUS INDICATOR
# -----------------------------
status_ph = st.empty()

# -----------------------------
# INPUT & STREAMING
# -----------------------------
st.divider()
with st.form("chat_form", clear_on_submit=True):
    user_input = st.text_area("Your Prompt", height=100)
    submitted  = st.form_submit_button("Send")

# Main submission handler
if submitted and user_input.strip():
    q_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    chat_area.markdown(
        f"<div class='bubble-user'>"
        f"<span class='avatar'>👤</span><b>You</b><br>({q_ts})<br>{user_input}"
        f"</div>", unsafe_allow_html=True
    )
    status_ph.info("🤔 Thinking...")
    header_ph = chat_area.empty()
    content_ph = chat_area.empty()

    endpoint = CLAUDE_ENDPOINT if "claude" in selected_model else OLLAMA_ENDPOINT
    citations = []

    context_text = ""
    if st.session_state.get("enable_web_search", False) and internet_online:
        results = web_search(user_input)
        if results:
            print(results)
            context_chunks, citations = [], []
            for res in results:
                snippet, link = res["snippet"], res["link"]
                context_chunks.append(snippet[:500])
                citations.append(link)
            context_chunks = context_chunks[:5]
            context_text = "\n\n".join(f"Context {i+1}:\n{chunk}" for i, chunk in enumerate(context_chunks))
        else:
            st.warning("⚠️ No web results found or failed to fetch results.")

    if context_text:
        prompt = f"{st.session_state.system_prompt}\n\nWeb Search Context:\n{context_text}\n\nUser Query:\n{user_input}"
    else:
        prompt = f"{st.session_state.system_prompt}\n\nUser Query:\n{user_input}"

    payload = {
        "model": selected_model,
        "prompt": prompt,
        "stream": True,
        "temperature": st.session_state.temperature,
        "options": {"num_predict": st.session_state.max_tokens}
    }

    buf = ""
    start = time.time()
    a_ts = q_ts
    for chunk in stream_response(endpoint, payload):
        buf = chunk
        a_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        header_ph.markdown(
            f"<div class='bubble-bot'>"
            f"<span class='avatar'>🤖</span><b>{selected_model}</b><br>({a_ts})"
            f"</div>", unsafe_allow_html=True
        )
        content_ph.markdown(format_output(buf, selected_model))
        elapsed = time.time() - start
        status_ph.markdown(f"Generating... {elapsed:.2f}s elapsed")

    status_ph.empty()

    # Show citations explicitly
    if citations:
        citation_md = "\n".join(f"- [{url}]({url})" for url in citations)
        chat_area.markdown(f"**Citations:**\n{citation_md}", unsafe_allow_html=True)
    elif st.session_state.get("enable_web_search", False):
        chat_area.markdown("**Citations:**\n_No citations were found._", unsafe_allow_html=True)

    st.session_state.chat_history.append((user_input, buf, q_ts, selected_model, a_ts))
    st.session_state.message_duration = time.time() - start
    st.info(f"Model response time took {st.session_state.message_duration:.2f} seconds")

    if st.session_state.autosave:
        try:
            with open("chat_autosave.txt", "w") as f:
                f.write("\n\n".join(
                    f"You ({u_ts}):\n{q}\n\nBot ({a_ts}):\n{a}"
                    for q, a, u_ts, m, a_ts in st.session_state.chat_history
                ))
        except Exception as e:
            st.error(f"Autosave Error: {e}")