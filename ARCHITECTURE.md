# Intelligent Customer Support Agent — System Architecture

---

## 1. High-Level Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          CLIENT LAYER                                   │
│         Chat Widget  │  Email Inbox  │  Slack / WhatsApp (optional)     │
└──────────────────────────────┬──────────────────────────────────────────┘
                               │  WebSocket / REST / Webhook
┌──────────────────────────────▼──────────────────────────────────────────┐
│                        API GATEWAY / ROUTER                             │
│                  (FastAPI — hosted on Modal / Railway)                  │
│     Session Manager  │  Rate Limiter  │  Auth Middleware                │
└──────────────────────────────┬──────────────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────────────┐
│                      AI AGENT CORE  (Claude claude-sonnet-4-6)                  │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │  Conversation Manager                                           │    │
│  │   • Maintains multi-turn context (message history per session)  │    │
│  │   • Handles system prompt + persona injection                   │    │
│  │   • Tracks conversation state (open / pending / escalated)      │    │
│  └──────────────────────────┬──────────────────────────────────────┘    │
│                             │ tool_use blocks                           │
│  ┌──────────────────────────▼──────────────────────────────────────┐    │
│  │  Tool / Function Dispatcher                                     │    │
│  │   • search_knowledge_base(query)                                │    │
│  │   • lookup_customer(email | customer_id)                        │    │
│  │   • create_ticket(subject, description, priority)               │    │
│  │   • update_ticket(ticket_id, status, note)                      │    │
│  │   • escalate_to_human(reason, urgency)                          │    │
│  │   • check_order_status(order_id)                                │    │
│  │   • send_email_reply(to, subject, body)                         │    │
│  └──────────────────────────┬──────────────────────────────────────┘    │
│                             │                                           │
│  ┌──────────────────────────▼──────────────────────────────────────┐    │
│  │  Escalation Engine                                              │    │
│  │   • Rule-based triggers (sentiment < threshold, VIP customer,   │    │
│  │     legal keywords, repeated contacts)                          │    │
│  │   • LLM-based confidence scoring                                │    │
│  │   • Routes to human queue with full context summary             │    │
│  └─────────────────────────────────────────────────────────────────┘    │
└─────────────────────────┬───────────────────────────────────────────────┘
                          │  API calls
        ┌─────────────────┼──────────────────────┐
        │                 │                      │
┌───────▼──────┐  ┌───────▼──────┐  ┌───────────▼─────────┐
│  Ticket API  │  │ Knowledge DB │  │   Customer DB        │
│  (Zendesk /  │  │ (Pinecone /  │  │  (PostgreSQL /       │
│   Freshdesk  │  │  Weaviate /  │  │   Supabase)          │
│   or custom) │  │  pgvector)   │  │                      │
└──────────────┘  └──────────────┘  └─────────────────────┘
```

---

## 2. Component Breakdown

### 2.1 Client Layer
| Channel | Protocol | Notes |
|---|---|---|
| Chat Widget | WebSocket (real-time) | Streaming responses via SSE |
| Email | Webhook (inbound parse) | SendGrid / Mailgun inbound parse |
| Slack / WhatsApp | Webhook | Optional Phase 2 |

### 2.2 API Gateway (FastAPI)
- **Endpoints:**
  - `POST /chat/message` — receive user message, return agent reply
  - `POST /email/inbound` — webhook for inbound email
  - `GET /conversation/{session_id}` — fetch conversation history
  - `POST /ticket/webhook` — receive ticket status updates
  - `GET /health` — uptime monitoring endpoint

- **Middleware:**
  - JWT / API key authentication
  - Per-user rate limiting (Redis)
  - Request logging & tracing (OpenTelemetry)

### 2.3 AI Agent Core
- **Model:** `claude-sonnet-4-6` (primary), `claude-haiku-4-5` (fast triage)
- **Pattern:** Agentic loop with tool use — agent runs until `end_turn` or `max_tokens`
- **System Prompt Sections:**
  - Role & persona definition
  - Escalation criteria
  - Tone & response length guidelines
  - Tool usage instructions
- **Context Window Management:**
  - Rolling window (last N turns)
  - Summarization of old turns to preserve context cheaply

### 2.4 Tools / Function Calls
```python
tools = [
    {
        "name": "search_knowledge_base",
        "description": "Search internal documentation and FAQs to answer customer questions",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Natural language search query"}
            },
            "required": ["query"]
        }
    },
    {
        "name": "lookup_customer",
        "description": "Look up customer account, subscription, and history by email or ID",
        "input_schema": {
            "type": "object",
            "properties": {
                "identifier": {"type": "string"},
                "identifier_type": {"type": "string", "enum": ["email", "customer_id"]}
            },
            "required": ["identifier", "identifier_type"]
        }
    },
    {
        "name": "create_ticket",
        "description": "Create a support ticket for tracking this issue",
        "input_schema": {
            "type": "object",
            "properties": {
                "subject": {"type": "string"},
                "description": {"type": "string"},
                "priority": {"type": "string", "enum": ["low", "medium", "high", "urgent"]},
                "category": {"type": "string"}
            },
            "required": ["subject", "description", "priority"]
        }
    },
    {
        "name": "escalate_to_human",
        "description": "Route this conversation to a human agent with a context summary",
        "input_schema": {
            "type": "object",
            "properties": {
                "reason": {"type": "string"},
                "urgency": {"type": "string", "enum": ["normal", "high", "critical"]},
                "summary": {"type": "string", "description": "Brief summary of the issue for the human agent"}
            },
            "required": ["reason", "urgency", "summary"]
        }
    },
    {
        "name": "check_order_status",
        "description": "Fetch real-time order status from the order management system",
        "input_schema": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string"}
            },
            "required": ["order_id"]
        }
    }
]
```

### 2.5 Escalation Engine
**Automatic escalation triggers:**
- Sentiment score below threshold (using Claude classification)
- Keywords: legal, lawsuit, fraud, refund > $X
- VIP / enterprise customer flag
- Issue unresolved after N turns
- Explicit customer request for human

**Escalation output:**
- Conversation transcript
- AI-generated context summary
- Customer tier & history
- Suggested resolution path

### 2.6 Data Layer
| Store | Purpose | Technology |
|---|---|---|
| Conversation Store | Session history, state | Redis (hot) + PostgreSQL (cold) |
| Knowledge Base | FAQs, docs, policies | pgvector / Pinecone |
| Customer DB | Account, orders, subscriptions | PostgreSQL / Supabase |
| Ticket Store | Support tickets | Zendesk / Freshdesk / custom |
| Metrics Store | KPIs, latency, resolution rate | TimescaleDB / Grafana |

---

## 3. Request Lifecycle (Step-by-Step)

```
1. Customer sends message (chat or email)
        ↓
2. API Gateway authenticates, creates/resumes session
        ↓
3. Session Manager loads conversation history from Redis
        ↓
4. Agent Core sends message + history to Claude API
        ↓
5. Claude returns response:
    a. Text only  →  stream back to customer
    b. Tool use   →  Dispatcher calls appropriate API
                      →  Result injected back into context
                      →  Claude generates final response
        ↓
6. Escalation Engine evaluates response & conversation state
    a. No escalation  →  response sent to customer
    b. Escalation     →  routed to human queue + notification
        ↓
7. Conversation state & metrics saved
        ↓
8. Response streamed to customer
```

---

## 4. Error Handling & Fallback Strategy

```
Primary:   Claude claude-sonnet-4-6  (full agent with tools)
    ↓ (timeout / API error)
Fallback 1: Claude claude-haiku-4-5  (lightweight, fast response)
    ↓ (both unavailable)
Fallback 2: Static response + ticket creation ("We'll get back to you")
    ↓ (catastrophic failure)
Fallback 3: Human escalation immediately with error context
```

**Tool failure handling:**
- Each tool call wrapped in try/except with structured error returns
- Agent instructed to gracefully inform customer if data is unavailable
- Retry with exponential backoff for transient API failures (max 3 retries)

---

## 5. Monitoring & Observability

### KPIs to Track
| Metric | Description | Target |
|---|---|---|
| First Contact Resolution (FCR) | Issues resolved in 1 conversation | >70% |
| Average Response Time | Time from message to reply | <3 seconds |
| Escalation Rate | % of conversations sent to human | <30% |
| CSAT Score | Customer satisfaction (1–5) | >4.2 |
| Tool Success Rate | % of tool calls that succeed | >98% |
| Hallucination Rate | Agent responses contradicting KB | <1% |

### Tooling
- **Logging:** Structured JSON logs → Datadog / Grafana Loki
- **Tracing:** OpenTelemetry spans across API → Agent → Tools
- **Alerting:** PagerDuty / Slack alerts for error spikes
- **Dashboard:** Grafana or custom admin panel

---

## 6. Deployment Architecture

```
┌──────────────────────────────────────────────────────────┐
│                    Modal / Railway                        │
│                                                          │
│  ┌────────────────┐    ┌────────────────┐                │
│  │  FastAPI App   │    │  Worker Queue  │                │
│  │  (web server)  │    │  (email batch) │                │
│  └────────────────┘    └────────────────┘                │
│                                                          │
│  ┌────────────────┐    ┌────────────────┐                │
│  │  Redis Cache   │    │  PostgreSQL    │                │
│  │  (sessions)    │    │  (persistent)  │                │
│  └────────────────┘    └────────────────┘                │
└──────────────────────────────────────────────────────────┘
          │                        │
    Anthropic API            External APIs
   (Claude models)       (Zendesk, SendGrid, etc.)
```

**Modal-specific:** Use `@modal.web_endpoint` for API + `@modal.function` for async workers
**Railway-specific:** Dockerfile + Railway services for app, Redis, Postgres

---

## 7. Project Folder Structure

```
intelligent-support-agent/
├── app/
│   ├── main.py                  # FastAPI entrypoint
│   ├── agent/
│   │   ├── core.py              # Claude agent loop
│   │   ├── tools.py             # Tool definitions & handlers
│   │   ├── escalation.py        # Escalation logic
│   │   └── prompts.py           # System prompts
│   ├── api/
│   │   ├── chat.py              # Chat endpoints
│   │   ├── email.py             # Email webhook endpoints
│   │   └── tickets.py           # Ticket management endpoints
│   ├── services/
│   │   ├── knowledge_base.py    # Vector search service
│   │   ├── customer_db.py       # Customer lookup service
│   │   ├── ticket_service.py    # Ticket CRUD
│   │   └── email_service.py     # Email send/receive
│   ├── models/
│   │   ├── conversation.py      # Conversation & message schemas
│   │   └── customer.py          # Customer data schemas
│   └── monitoring/
│       ├── metrics.py           # Custom KPI tracking
│       └── tracing.py           # OpenTelemetry setup
├── tests/
│   ├── test_agent.py
│   ├── test_tools.py
│   └── test_escalation.py
├── scripts/
│   ├── seed_knowledge_base.py   # Load docs into vector DB
│   └── load_test.py             # Simulate concurrent conversations
├── deploy/
│   ├── modal_app.py             # Modal deployment config
│   └── railway.json             # Railway deployment config
├── .env.example
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## 8. Tech Stack Summary

| Layer | Technology | Why |
|---|---|---|
| LLM | Anthropic Claude claude-sonnet-4-6 / claude-haiku-4-5 | Tool use, streaming, multi-turn |
| Backend | FastAPI + Python 3.11 | Async, fast, typed |
| Vector DB | pgvector or Pinecone | Semantic knowledge search |
| Session Cache | Redis | Fast session retrieval |
| Database | PostgreSQL / Supabase | Customer & ticket storage |
| Email | SendGrid / Mailgun | Inbound parse + outbound send |
| Ticket System | Zendesk API or custom | Issue tracking |
| Deployment | Modal or Railway | Serverless / managed hosting |
| Monitoring | Grafana + Prometheus | Metrics dashboards |
| Tracing | OpenTelemetry | End-to-end request tracing |

---

*Generated: 2026-02-24 | Phase 1 Scope: Chat + Email + Core Tools + Escalation*
