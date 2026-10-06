"""
ResearchMind - Streamlit UI for the Multi-Agent Research System.

Run with (inside your activated .venv):
    python -m streamlit run app.py

It re-uses the agents from agents.py and mirrors the 4 steps of pipeline.py
(search -> reader -> writer -> critic), but shows live progress per agent in
the browser instead of printing to the terminal.
"""

import html
import re
import time

import streamlit as st

# Load API keys from .env (agents.py may already do this; harmless twice)
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from agents import (
    build_reader_agent,
    build_search_agent,
    critic_chain,
    writer_chain,
)

# --------------------------------------------------------------------------- #
# Config - change the branding here
# --------------------------------------------------------------------------- #
APP_NAME_LEFT = "Research"   # shown in off-white
APP_NAME_RIGHT = "Mind"      # shown in orange

AGENTS = [
    ("01", "Search Agent", "Gathers recent web information"),
    ("02", "Reader Agent", "Scrapes & extracts deep content"),
    ("03", "Writer Agent", "Drafts a structured research report"),
    ("04", "Critic Agent", "Reviews the report and gives feedback"),
]

EXAMPLES = {
    "Quantum computing": "Quantum computing breakthroughs in 2026",
    "AI in healthcare": "How AI is transforming healthcare diagnostics",
    "Climate tech": "Latest innovations in climate technology and carbon capture",
}

st.set_page_config(
    page_title=f"{APP_NAME_LEFT}{APP_NAME_RIGHT}",
    page_icon="⚡",
    layout="wide",
)

# --------------------------------------------------------------------------- #
# Styling
# --------------------------------------------------------------------------- #
CSS = """<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=DM+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

:root {
  --bg: #0a0a0d;
  --panel: #111116;
  --panel-2: #17171e;
  --border: #25252e;
  --text: #ecebe6;
  --muted: #9a9aa5;
  --accent: #ff8a2a;
  --accent-2: #ff5a1f;
  --green: #34d399;
  --red: #f87171;
}

.stApp {
  background:
    radial-gradient(900px 420px at 50% -80px, rgba(255, 122, 26, 0.16), transparent 70%),
    var(--bg);
  color: var(--text);
  font-family: 'DM Sans', sans-serif;
}
[data-testid="stHeader"] { background: transparent; }
#MainMenu, footer { visibility: hidden; }
.block-container, [data-testid="stMainBlockContainer"] {
  max-width: 1180px;
  padding-top: 3.5rem;
  padding-bottom: 5rem;
}

/* ---------- Hero ---------- */
.eyebrow {
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.72rem;
  letter-spacing: 0.3em;
  text-transform: uppercase;
  color: var(--accent);
  text-align: center;
  margin-bottom: 1rem;
}
.hero-title {
  font-family: 'Syne', sans-serif;
  font-weight: 800;
  font-size: clamp(3rem, 9vw, 6.4rem);
  line-height: 1;
  letter-spacing: -0.02em;
  text-align: center;
  color: var(--text);
}
.hero-title span { color: var(--accent); }
.hero-sub {
  max-width: 560px;
  margin: 1.5rem auto 0 auto;
  text-align: center;
  color: var(--muted);
  font-size: 0.98rem;
  line-height: 1.6;
}
.hero-divider {
  height: 1px;
  margin: 2.4rem 0 1.4rem 0;
  background: linear-gradient(90deg, transparent, var(--border), transparent);
}

/* ---------- Labels ---------- */
.field-label {
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.72rem;
  letter-spacing: 0.2em;
  text-transform: uppercase;
  color: var(--accent);
  padding-bottom: 0.6rem;
}
.try-label {
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.7rem;
  letter-spacing: 0.2em;
  text-transform: uppercase;
  color: var(--muted);
  padding-top: 0.6rem;
  padding-bottom: 0.9rem;
}
.note {
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.78rem;
  color: var(--muted);
}

/* ---------- Input + buttons ---------- */
[data-testid="stForm"] { border: none; padding: 0; background: transparent; }

[data-testid="stTextInput"] [data-baseweb="input"] {
  background: var(--panel-2);
  border: 1px solid var(--border);
  border-radius: 12px;
}
[data-testid="stTextInput"] [data-baseweb="input"]:focus-within {
  border-color: var(--accent);
  box-shadow: 0 0 0 1px var(--accent);
}
[data-testid="stTextInput"] input {
  background: transparent;
  color: var(--text);
  font-size: 1rem;
  padding: 0.85rem 1rem;
}
[data-testid="stTextInput"] input::placeholder { color: #6b6b76; }

/* Primary run button */
[data-testid="stFormSubmitButton"],
[data-testid="stFormSubmitButton"] button { width: 100% !important; }
[data-testid="stElementContainer"]:has([data-testid="stFormSubmitButton"]) { width: 100% !important; }
.element-container:has([data-testid="stFormSubmitButton"]) { width: 100% !important; }
[data-testid="stFormSubmitButton"] button {
  background: linear-gradient(90deg, var(--accent), var(--accent-2));
  border: none;
  border-radius: 12px;
  padding: 0.85rem 1rem;
  box-shadow: 0 8px 30px rgba(255, 106, 26, 0.25);
  transition: filter 0.15s ease, transform 0.15s ease;
}
[data-testid="stFormSubmitButton"] button p {
  color: #fff;
  font-weight: 700;
  font-size: 1rem;
}
[data-testid="stFormSubmitButton"] button:hover {
  filter: brightness(1.08);
  transform: translateY(-1px);
}

/* Example chips */
[data-testid="stButton"],
[data-testid="stButton"] button { width: 100% !important; }
[data-testid="stElementContainer"]:has([data-testid="stButton"]) { width: 100% !important; }
.element-container:has([data-testid="stButton"]) { width: 100% !important; }
[data-testid="stButton"] button {
  background: var(--panel-2);
  border: 1px solid var(--border);
  border-radius: 999px;
  padding: 0.25rem 0.6rem;
  min-height: 0;
}
[data-testid="stButton"] button p {
  color: var(--muted);
  font-size: 0.78rem;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
[data-testid="stButton"] button:hover { border-color: var(--accent); }
[data-testid="stButton"] button:hover p { color: var(--accent); }

/* Download buttons */
[data-testid="stDownloadButton"] button {
  background: transparent;
  border: 1px solid var(--accent);
  border-radius: 10px;
}
[data-testid="stDownloadButton"] button p { color: var(--accent); font-weight: 600; }
[data-testid="stDownloadButton"] button:hover { background: rgba(255, 138, 42, 0.1); }

/* ---------- Pipeline panel ---------- */
.panel-title {
  font-family: 'Syne', sans-serif;
  font-weight: 700;
  font-size: 1.45rem;
  color: var(--text);
  margin-bottom: 0.2rem;
}
.step {
  background: var(--panel);
  border: 1px solid var(--border);
  border-left: 3px solid var(--border);
  border-radius: 14px;
  padding: 1rem 1.2rem;
  margin-bottom: 0.8rem;
  transition: border-color 0.2s ease;
}
.step-head { display: flex; align-items: center; gap: 0.8rem; }
.step-num {
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.72rem;
  color: var(--accent);
}
.step-name { font-weight: 700; font-size: 1rem; color: var(--text); flex: 1; }
.step-badge {
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.68rem;
  letter-spacing: 0.1em;
  color: var(--muted);
}
.step-desc { margin-top: 0.3rem; font-size: 0.85rem; color: var(--muted); }

.step.running { border-left-color: var(--accent); }
.step.running .step-badge { color: var(--accent); animation: pulse 1.2s ease-in-out infinite; }
.step.done { border-left-color: var(--green); }
.step.done .step-badge { color: var(--green); }
.step.error { border-left-color: var(--red); }
.step.error .step-badge { color: var(--red); }

@keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.4; } }

/* ---------- Results ---------- */
.results-head { margin: 1.5rem 0 1rem 0; }
.results-title {
  font-family: 'Syne', sans-serif;
  font-weight: 700;
  font-size: 1.8rem;
  color: var(--text);
  margin-top: 0.3rem;
}
.stats { display: flex; gap: 0.8rem; margin-bottom: 1.2rem; flex-wrap: wrap; }
.stat {
  flex: 1;
  min-width: 140px;
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: 14px;
  padding: 0.9rem 1.1rem;
}
.stat-label {
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.66rem;
  letter-spacing: 0.18em;
  text-transform: uppercase;
  color: var(--muted);
}
.stat-value {
  font-family: 'Syne', sans-serif;
  font-weight: 700;
  font-size: 1.5rem;
  color: var(--accent);
  margin-top: 0.15rem;
}

/* Tabs */
[data-testid="stTabs"] button[role="tab"] p {
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.74rem;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--muted);
}
[data-testid="stTabs"] button[role="tab"][aria-selected="true"] p { color: var(--accent); }
[data-baseweb="tab-highlight"] { background-color: var(--accent); }
[data-baseweb="tab-border"] { background-color: var(--border); }

/* Markdown content (report etc.) */
.stMarkdown, .stMarkdown p, .stMarkdown li { color: var(--text); }
.stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
  font-family: 'Syne', sans-serif;
  color: var(--text);
}
.stMarkdown a { color: var(--accent); }
</style>"""

st.markdown(CSS, unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def to_text(content) -> str:
    """Normalise whatever an agent/chain returns into a plain string."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if hasattr(content, "content"):  # LangChain message objects
        return to_text(content.content)
    if isinstance(content, list):  # list of content blocks
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "\n".join(p for p in parts if p)
    return str(content)


def slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug[:50] or "report"


def set_topic(value: str) -> None:
    """Callback for the example chips."""
    st.session_state["topic_input"] = value


def new_steps() -> list:
    return [{"state": "idle", "secs": None} for _ in AGENTS]


def render_pipeline(slot, steps: list) -> None:
    labels = {
        "idle": "WAITING",
        "running": "● RUNNING",
        "done": "✓ DONE",
        "error": "✕ FAILED",
    }
    cards = []
    for (num, name, desc), step in zip(AGENTS, steps):
        label = labels[step["state"]]
        if step["state"] == "done" and step["secs"] is not None:
            label += f" · {step['secs']:.0f}s"
        cards.append(
            f'<div class="step {step["state"]}">'
            f'<div class="step-head"><span class="step-num">{num}</span>'
            f'<span class="step-name">{name}</span>'
            f'<span class="step-badge">{label}</span></div>'
            f'<div class="step-desc">{desc}</div></div>'
        )
    slot.markdown("".join(cards), unsafe_allow_html=True)


def mark(slot, steps: list, index: int, state: str, secs: float | None = None) -> None:
    steps[index] = {"state": state, "secs": secs}
    st.session_state["steps"] = steps
    render_pipeline(slot, steps)


# --------------------------------------------------------------------------- #
# Pipeline (same logic as pipeline.py, but reports progress to the UI)
# --------------------------------------------------------------------------- #
def run_pipeline(topic: str, slot):
    """Runs the 4 steps. Returns (state, timings) or (None, None) on failure."""
    steps = new_steps()
    st.session_state["steps"] = steps
    render_pipeline(slot, steps)

    state: dict = {}
    timings: dict = {}
    current = 0

    try:
        # Step 1 - search agent
        current = 0
        mark(slot, steps, current, "running")
        t0 = time.perf_counter()
        search_agent = build_search_agent()
        search_result = search_agent.invoke(
            {
                "messages": [
                    (
                        "user",
                        f"Find recent,reliable and detailed information about: {topic}",
                    )
                ]
            }
        )
        state["search_results"] = to_text(search_result["messages"][-1].content)
        timings["Search"] = time.perf_counter() - t0
        mark(slot, steps, current, "done", timings["Search"])

        # Step 2 - reader agent
        current = 1
        mark(slot, steps, current, "running")
        t0 = time.perf_counter()
        reader_agent = build_reader_agent()
        reader_result = reader_agent.invoke(
            {
                "messages": [
                    (
                        "user",
                        f"Based on the following search results about '{topic}', "
                        f"pick the most relevant URL and scrape it for deeper content.\n\n"
                        f"Search Results:\n{state['search_results'][:1000]}",
                    )
                ]
            }
        )
        state["scraped_content"] = to_text(reader_result["messages"][-1].content)
        timings["Reader"] = time.perf_counter() - t0
        mark(slot, steps, current, "done", timings["Reader"])

        # Step 3 - writer chain
        current = 2
        mark(slot, steps, current, "running")
        t0 = time.perf_counter()
        research_combined = (
            f"SEARCH RESULTS : \n {state['search_results']} \n\n"
            f"DETAILED SCRAPED CONTENT : \n {state['scraped_content']}"
        )
        state["report"] = to_text(
            writer_chain.invoke({"topic": topic, "research": research_combined})
        )
        timings["Writer"] = time.perf_counter() - t0
        mark(slot, steps, current, "done", timings["Writer"])

        # Step 4 - critic chain
        current = 3
        mark(slot, steps, current, "running")
        t0 = time.perf_counter()
        state["feedback"] = to_text(critic_chain.invoke({"report": state["report"]}))
        timings["Critic"] = time.perf_counter() - t0
        mark(slot, steps, current, "done", timings["Critic"])

        return state, timings

    except Exception as e:  # noqa: BLE001 - show any failure in the UI
        mark(slot, steps, current, "error")
        st.error(f"{AGENTS[current][1]} failed: {e}")
        st.exception(e)
        return None, None


# --------------------------------------------------------------------------- #
# Results view
# --------------------------------------------------------------------------- #
def render_results(topic: str, state: dict, timings: dict) -> None:
    total = sum(timings.values())
    words = len(state["report"].split())

    st.markdown(
        '<div class="hero-divider"></div>'
        '<div class="results-head">'
        '<div class="field-label" style="padding-bottom:0">Results</div>'
        f'<div class="results-title">{html.escape(topic)}</div></div>'
        '<div class="stats">'
        f'<div class="stat"><div class="stat-label">Total time</div><div class="stat-value">{total:.0f}s</div></div>'
        f'<div class="stat"><div class="stat-label">Report length</div><div class="stat-value">{words} words</div></div>'
        '<div class="stat"><div class="stat-label">Agents run</div><div class="stat-value">4</div></div>'
        "</div>",
        unsafe_allow_html=True,
    )

    tab_report, tab_feedback, tab_search, tab_scraped = st.tabs(
        ["Report", "Critic feedback", "Search results", "Scraped content"]
    )
    slug = slugify(topic)

    with tab_report:
        st.markdown(state["report"])
        st.download_button(
            "⬇ Download report (.md)",
            data=state["report"],
            file_name=f"{slug}-report.md",
            mime="text/markdown",
        )

    with tab_feedback:
        st.markdown(state["feedback"])
        full_export = (
            f"# {topic}\n\n"
            f"## Report\n\n{state['report']}\n\n"
            f"## Critic Feedback\n\n{state['feedback']}\n"
        )
        st.download_button(
            "⬇ Download report + feedback (.md)",
            data=full_export,
            file_name=f"{slug}-full.md",
            mime="text/markdown",
        )

    with tab_search:
        st.markdown(state["search_results"])

    with tab_scraped:
        st.markdown(state["scraped_content"])


# --------------------------------------------------------------------------- #
# Page
# --------------------------------------------------------------------------- #
if "steps" not in st.session_state:
    st.session_state["steps"] = new_steps()
if "topic_input" not in st.session_state:
    st.session_state["topic_input"] = ""

# Hero
st.markdown(
    '<div class="eyebrow">Multi-Agent AI System</div>'
    f'<div class="hero-title">{APP_NAME_LEFT}<span>{APP_NAME_RIGHT}</span></div>'
    '<div class="hero-sub">Four specialized AI agents collaborate — searching, scraping, '
    "writing, and critiquing — to deliver a polished research report on any topic.</div>"
    '<div class="hero-divider"></div>',
    unsafe_allow_html=True,
)

left, right = st.columns([1.1, 1], gap="large")

with left:
    st.markdown('<div class="field-label">Research topic</div>', unsafe_allow_html=True)
    with st.form("research_form", border=False):
        topic_input = st.text_input(
            "Research topic",
            key="topic_input",
            placeholder="e.g. Quantum computing breakthroughs in 2026",
            label_visibility="collapsed",
        )
        submitted = st.form_submit_button("⚡ Run Research Pipeline")

    st.markdown('<div class="try-label">Try →</div>', unsafe_allow_html=True)
    chip_cols = st.columns(len(EXAMPLES))
    for col, (label, full_topic) in zip(chip_cols, EXAMPLES.items()):
        col.button(label, on_click=set_topic, args=(full_topic,), key=f"chip_{label}")

    status_slot = st.empty()

with right:
    st.markdown('<div class="panel-title">Pipeline</div>', unsafe_allow_html=True)
    pipeline_slot = st.empty()

render_pipeline(pipeline_slot, st.session_state["steps"])

if submitted:
    topic = topic_input.strip()
    if not topic:
        status_slot.warning("Please enter a research topic first.")
    else:
        for key in ("result", "timings", "research_topic"):
            st.session_state.pop(key, None)
        status_slot.markdown(
            '<div class="note">Agents are working — this usually takes a minute or two.</div>',
            unsafe_allow_html=True,
        )
        result, timings = run_pipeline(topic, pipeline_slot)
        status_slot.empty()
        if result:
            st.session_state["result"] = result
            st.session_state["timings"] = timings
            st.session_state["research_topic"] = topic

# Results live in session_state so they survive reruns (e.g. clicking download)
if "result" in st.session_state:
    render_results(
        st.session_state["research_topic"],
        st.session_state["result"],
        st.session_state["timings"],
    )