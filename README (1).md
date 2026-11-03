# AI-Powered Chatbot Assistant

A conversational support chatbot in pure Python (standard library only). It answers common questions instantly from an FAQ using TF-IDF retrieval, and sends everything else to the OpenAI API with a grounded, prompt-engineered request.

## Features

- **Automated FAQ handling**: TF-IDF + cosine similarity implemented from scratch (no scikit-learn).
- **OpenAI API integration**: Chat Completions via `urllib`, with timeouts, retry and exponential backoff.
- **Prompt engineering**: role and rules system prompt, plus relevant FAQ entries injected as context to reduce hallucination.
- **Conversation memory**: sliding window of recent turns (`max_turns`).
- **Graceful degradation**: no API key or API failure -> falls back to FAQ-only answers.
- **CLI commands**: `/help`, `/history`, `/reset`, `/save`, `/quit`.
- **Unit tests** with a fake LLM client (no network or API cost).

## How it works

```
user message
    |
    v
TF-IDF FAQ search --score >= 0.60--> answer from FAQ        (fast, free)
    |
    | else
    v
build prompt = system rules + related FAQ context + chat history + question
    |
    v
OpenAI API --success--> LLM answer
    |
    | fail / no key
    v
weak FAQ match (>= 0.25) --> FAQ answer, otherwise polite fallback
```

## Setup

Requires Python 3.9+. No `pip install` needed.

```bash
git clone https://github.com/SuruthiSankar/ai-chatbot-assistant.git
cd ai-chatbot-assistant

# Optional: enable LLM mode
export OPENAI_API_KEY="sk-..."        # Windows PowerShell: $env:OPENAI_API_KEY="sk-..."
export OPENAI_MODEL="gpt-4o-mini"     # optional

python chatbot.py
```

## Run tests

```bash
python -m unittest -v
```

## Customize

- Edit `faq.json` with your own question/answer pairs.
- Tune thresholds and memory size in the `Settings` dataclass in `chatbot.py`.
- Change the bot's personality in `SYSTEM_PROMPT`.

## Project structure

```
chatbot.py        # preprocessing, TF-IDF, OpenAI client, routing, CLI
faq.json          # knowledge base
test_chatbot.py   # unit tests
```

## Possible improvements

- Streaming responses, a web UI (Flask/Django), embeddings-based retrieval, intent classification.

## License

MIT
