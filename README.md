# ResearchMind: Multi-Agent Research System

A multi-agent AI pipeline that turns a single research topic into a structured, sourced report. Four specialised agents work in sequence: one **searches** the web, one **reads** a source in depth, one **writes** the report, and one **critiques** it. You can run it from the terminal or through a Streamlit web app.

## How it works

```
Topic ─► Search Agent ─► Reader Agent ─► Writer ─► Critic ─► Report + Feedback
         (Tavily)        (scrapes a      (drafts     (scores
                          top URL)        report)     the report)
```

| # | Agent | What it does |
|---|-------|--------------|
| 1 | **Search Agent** | Uses the Tavily search API to find recent, reliable sources (top 3 results with title, URL and snippet). |
| 2 | **Reader Agent** | Picks the most relevant URL from the search results and scrapes the page for deeper content. |
| 3 | **Writer** | Combines the search results and scraped content into a report with an Introduction, Key Findings (at least 3 points), Conclusion and Sources. |
| 4 | **Critic** | Reviews the report and returns a score out of 10, strengths, areas to improve and a one-line verdict. |

The search and reader steps are tool-using agents built with LangChain's `create_agent`. The writer and critic are prompt chains (`prompt | llm | output parser`).

## Features

- Four-stage pipeline with a clear separation of roles
- Runs on [Groq](https://groq.com/) with `openai/gpt-oss-120b` by default (switchable through an environment variable)
- Reasoning effort tuned per step (low for search, reader and critic; medium for the writer) to stay within Groq's tokens-per-minute limits
- Automatic retries (`max_retries=6`) so a rate-limited call is retried on its own instead of restarting the whole run
- Streamlit UI with live per-agent progress and timings, example topic chips, and tabs for the report, critic feedback, raw search results and scraped content
- Download the report, or the report plus critic feedback, as Markdown

## Project structure

```
.
├── agents.py            # LLM setup, search/reader agents, writer and critic chains
├── tools.py             # web_search (Tavily) and scrape_url (requests + BeautifulSoup)
├── pipeline.py          # CLI entry point: runs the four steps and prints each result
├── app.py               # Streamlit UI (ResearchMind)
├── .streamlit/
│   └── config.toml      # Dark theme with orange accent
├── requirement.txt      # Python dependencies
└── .gitignore
```

## Getting started

### Prerequisites

- Python 3.10 or newer
- A [Groq API key](https://console.groq.com/)
- A [Tavily API key](https://tavily.com/)

### Installation

```bash
git clone https://github.com/CodaptorShafa/multi-agent-research-system.git
cd multi-agent-research-system

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirement.txt
pip install langchain-groq streamlit
```

> `langchain-groq` and `streamlit` are imported by the code, so install them alongside the requirements file if they aren't already present. The code also uses `create_agent`, which needs a recent LangChain release (1.x).

### Configuration

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key
TAVILY_API_KEY=your_tavily_api_key

# Optional: use a different Groq model
# GROQ_MODEL=openai/gpt-oss-120b
```

The `.env` file is listed in `.gitignore`, so your keys won't be committed.

## Usage

### Web app

```bash
python -m streamlit run app.py
```

Enter a topic (or click one of the example chips), press **Run Research Pipeline**, and watch each agent move from waiting to running to done. A run usually takes a minute or two. When it finishes, the results are split across four tabs: Report, Critic feedback, Search results and Scraped content.

### Command line

```bash
python pipeline.py
```

You'll be prompted for a topic, and each step's output is printed to the terminal as it completes.

### Use it from your own code

```python
from pipeline import run_research_pipeline

state = run_research_pipeline("Quantum computing breakthroughs in 2026")

print(state["report"])
print(state["feedback"])
```

The returned `state` dictionary contains `search_results`, `scraped_content`, `report` and `feedback`.

## Configuration notes

- **Model:** set `GROQ_MODEL` in `.env` to try another Groq model. The `reasoning_effort` setting is only applied to `gpt-oss` models.
- **Branding:** the app name shown in the UI is controlled by `APP_NAME_LEFT` and `APP_NAME_RIGHT` at the top of `app.py`.
- **Theme:** colours are set in `.streamlit/config.toml` and the CSS block in `app.py`.

## Limitations

- The reader agent scrapes a single page per run, and only the first 3,000 characters of its text are used.
- Search returns 3 results with short snippets, and only the first 1,000 characters of the search output are passed to the reader agent.
- Scraping uses plain HTTP requests, so pages that rely on JavaScript to render their content, or that block scrapers, may return little or nothing.
- Report quality depends on the sources found; the critic's score is an LLM judgement, not a fact-check.

## Tech stack

- [LangChain](https://www.langchain.com/) for agents, prompts and chains
- [Groq](https://groq.com/) for LLM inference
- [Tavily](https://tavily.com/) for web search
- [Requests](https://requests.readthedocs.io/) and [Beautiful Soup](https://www.crummy.com/software/BeautifulSoup/) for scraping
- [Streamlit](https://streamlit.io/) for the web interface

## Ideas for future work

- Scrape and summarise several sources instead of one
- Loop the critic's feedback back into the writer for a revised draft
- Stream agent output live in the UI

## License

No license has been specified yet. Add a `LICENSE` file to choose one.
