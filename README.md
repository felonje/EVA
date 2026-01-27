# EVA — Voice-first assistant prototype

This repo contains the EVA prototype:
- Backend: FastAPI proxy that dispatches to provider adapters (Gemini / Grok)
- Local agent: Python push-to-talk agent (microphone → STT → backend → TTS)
- Providers are adapters under backend/providers/ and are configurable via env vars

Important
- Do NOT commit API keys. Use environment variables or a secrets manager.
- The prototype requires you to create a GitHub repo `felonje/EVA` first if you want me to push.
- By default image generation is routed to a separate `/v1/image` endpoint.

Quick start
1. Create & activate venv:
   python3 -m venv .venv && source .venv/bin/activate
2. Install:
   pip install -r requirements.txt
3. Copy `.env.example` to `.env` and fill in values (no API keys in repo)
4. Start backend:
   uvicorn backend.app:app --reload --host 127.0.0.1 --port 8000
5. Run agent:
   python agent/main.py

Configuration
- PROVIDER: "mock" | "gemini" | "grok"
- PROVIDER_MODEL or provider-specific env vars (GEMINI_MODEL, GROK_MODEL)
- PROVIDER_API_KEY / GEMINI_API_KEY / GROK_API_KEY
