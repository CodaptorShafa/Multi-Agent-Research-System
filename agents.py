import os

from langchain.agents import create_agent
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from tools import web_search,scrape_url
from dotenv import load_dotenv

load_dotenv()

#model setup
# To try another Groq model without touching code, add GROQ_MODEL=<model id> to your .env
MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")


def make_llm(reasoning_effort: str) -> ChatGroq:
    """Build the Groq model.

    gpt-oss models "think" before answering, and those hidden reasoning tokens count
    toward Groq's tokens-per-minute (TPM) limit, so we keep the effort low/medium.
    max_retries lets the Groq client wait out a 429 and retry just that one call
    (instead of the whole agent run starting over).
    """
    kwargs = {}
    if "gpt-oss" in MODEL:  # reasoning_effort is only valid for reasoning models
        kwargs["reasoning_effort"] = reasoning_effort
    return ChatGroq(model=MODEL, temperature=0, max_retries=6, **kwargs)


llm = make_llm("low")            # search agent, reader agent, critic
writer_llm = make_llm("medium")  # the report writer gets a bit more thinking

#first agent
def build_reader_agent():
    return create_agent(
        model=llm,
        tools=[scrape_url],
        system_prompt="""
You are a research reader agent.

Your job:
1. Select the most relevant URL from the search results.
2. Use scrape_url on that URL.
3. Return the useful information extracted from the page.

Do not perform additional web searches.
"""
    )
#Second Agent
def build_search_agent():
    return create_agent(
        model=llm,
        tools=[web_search],
        system_prompt="""
You are a web research agent.

Search for recent and reliable information.
Use the web_search tool.
Return concise search results with titles, URLs, and snippets.
"""
    )
#writer chain
writer_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert research writer. Write clear, structured and insightful reports."),
    ("human", """Write a detailed research report on the topic below.

Topic: {topic}

Research Gathered:
{research}

Structure the report as:
- Introduction
- Key Findings (minimum 3 well-explained points)
- Conclusion
- Sources (list all URLs found in the research)

Be detailed, factual and professional."""),
])
writer_chain = writer_prompt | writer_llm | StrOutputParser()
#critic chain

critic_prompt = ChatPromptTemplate.from_messages([
     ("system", "You are a sharp and constructive research critic. Be honest and specific."),
    ("human", """Review the research report below and evaluate it strictly.

Report:
{report}

Respond in this exact format:

Score: X/10

Strengths:
- ...
- ...

Areas to Improve:
- ...
- ...

One line verdict:
..."""),
])
critic_chain = critic_prompt | llm | StrOutputParser()