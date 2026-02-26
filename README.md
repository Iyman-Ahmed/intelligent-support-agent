---
title: Intelligent Customer Support Agent
emoji: 🤖
colorFrom: purple
colorTo: blue
sdk: gradio
sdk_version: "5.0.0"
app_file: app.py
pinned: false
license: mit
short_description: AI support agent with tool use and escalation
---

# Intelligent Customer Support Agent

A production-ready AI customer support agent built with **Google Gemini 2.0 Flash (free tier)**, FastAPI, and a full service stack. Handles multi-turn conversations via chat and email, calls real APIs via tool use, and automatically escalates complex issues to human agents.

> **100% free to run** — uses Gemini 2.0 Flash free tier (1,500 requests/day).
> Get your free API key at [aistudio.google.com](https://aistudio.google.com/apikey).

---

## Quick Start

```bash
# 1. Clone and install
pip install -r requirements.txt

# 2. Configure environment
cp .env.example .env
# → Add your GEMINI_API_KEY (free at aistudio.google.com)

# 3. Run locally
python app/main.py
# → Server at http://localhost:8000
# → API docs at http://localhost:8000/docs
```

## Test a conversation

```bash
curl -X POST http://localhost:8000/chat/message \
  -H "Content-Type: application/json" \
  -H "X-API-Key: dev-secret-key" \
  -d '{"message": "Hi, I need help with my refund"}'
```

---

## Architecture

```
Client (Chat / Email)
    ↓
FastAPI Gateway  (auth, rate limiting, session management)
    ↓
Agent Core  (Claude claude-sonnet-4-6 agentic loop)
    ↓ tool_use
Tool Dispatcher  ──→  Knowledge Base (semantic search)
                  ──→  Customer DB (account lookup)
                  ──→  Ticket Service (create/update tickets)
                  ──→  Escalation Engine (human handoff)
                  ──→  Email Service (outbound replies)
```

---

## Project Structure

```
app/
├── main.py               # FastAPI app + startup
├── config.py             # Environment settings
├── session_store.py      # Redis-backed session persistence
├── agent/
│   ├── core.py           # Agentic loop (tool use + fallback)
│   ├── tools.py          # Tool schemas + dispatcher
│   ├── escalation.py     # Rule-based + LLM escalation
│   └── prompts.py        # System prompts
├── api/
│   ├── chat.py           # Chat + WebSocket endpoints
│   ├── email.py          # Email webhook endpoints
│   └── tickets.py        # Ticket CRUD
├── services/
│   ├── knowledge_base.py # Vector search (Pinecone / pgvector / mock)
│   ├── customer_db.py    # Customer lookup
│   ├── ticket_service.py # Ticket management
│   └── email_service.py  # SendGrid / Mailgun
├── models/
│   ├── conversation.py   # Session, message, status models
│   └── customer.py       # Customer, order, ticket models
└── monitoring/
    ├── metrics.py        # KPI tracking (FCR, response time, CSAT)
    └── tracing.py        # OpenTelemetry distributed tracing
```

---

## Key Features

- **Multi-turn conversations** with rolling context window + history compression
- **7 built-in tools**: knowledge base search, customer lookup, ticket creation, order status, escalation, email reply
- **Dual-model fallback**: Sonnet → Haiku → static response
- **Smart escalation**: legal keywords, VIP flag, sentiment score, turn limit
- **Real-time streaming** via SSE + WebSocket
- **Email channel**: inbound parse (SendGrid / Mailgun) + outbound replies
- **Metrics**: Prometheus-compatible /metrics endpoint tracking FCR, response time, CSAT

---

## Deployment

### Modal
```bash
modal deploy deploy/modal_app.py
```

### Railway
```bash
# Connect your repo to Railway and it will auto-deploy using railway.json
railway up
```

### Docker
```bash
docker build -t support-agent .
docker run -p 8000:8000 --env-file .env support-agent
```

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | /chat/message | Send a message, get agent reply |
| POST | /chat/stream | Streaming SSE response |
| WS | /chat/ws/{session_id} | Real-time WebSocket chat |
| GET | /chat/conversation/{id} | Fetch conversation summary |
| POST | /email/inbound/sendgrid | SendGrid inbound webhook |
| POST | /email/inbound/mailgun | Mailgun inbound webhook |
| GET | /tickets/ | List open tickets |
| POST | /tickets/ | Create a ticket |
| PATCH | /tickets/{id} | Update a ticket |
| GET | /metrics | Prometheus metrics |
| GET | /metrics/summary | Human-readable KPI summary |
| GET | /health | Health check |

---

## Running Tests

```bash
pytest tests/ -v
```

## Load Testing

```bash
python scripts/load_test.py --users 20 --url http://localhost:8000
```
# intelligent-support-agent
