<p align="center">
  <img src="https://img.shields.io/badge/NVIDIA-Nemotron_Ultra-76B900?style=for-the-badge&logo=nvidia&logoColor=white" alt="Nemotron">
  <img src="https://img.shields.io/badge/Nebius-AI_Cloud-4285F4?style=for-the-badge" alt="Nebius">
  <img src="https://img.shields.io/badge/Post--Quantum-Ready-7C3AED?style=for-the-badge" alt="PQ">
  <img src="https://img.shields.io/badge/NIST_SP_800--90B-Compliant-00897B?style=for-the-badge" alt="NIST">
  <img src="https://img.shields.io/badge/Patent-IDF--B_Filed-E65100?style=for-the-badge" alt="Patent">
  <img src="https://img.shields.io/badge/License-MIT-blue?style=for-the-badge" alt="MIT">
</p>

<h1 align="center">CryptoGuard AI</h1>

<h3 align="center">AI-Driven Cryptographic Security Operations Center<br>with NIST SP 800-90B Entropy Pipeline &amp; Blockchain Anchoring</h3>

<p align="center">
  An intelligent security platform that scans code for cryptographic vulnerabilities, runs NIST-compliant entropy analysis, extracts near-uniform random bits via the Leftover Hash Lemma, and anchors audit trails on a Merkle-tree ledger — powered by <strong>NVIDIA Nemotron</strong> on <strong>Nebius AI Cloud</strong>.
</p>

<p align="center">
  <a href="#quick-start">Quick Start</a> &bull;
  <a href="#architecture">Architecture</a> &bull;
  <a href="#features">Features</a> &bull;
  <a href="#api-reference">API</a> &bull;
  <a href="#patent-claims">Patent Claims</a> &bull;
  <a href="#deployment">Deployment</a>
</p>

---

## What It Does

CryptoGuard AI is a unified cryptographic security platform spanning three layers:

| Layer | What | How |
|-------|------|-----|
| **AI Security Agent** | Code scanning, entropy assessment, post-quantum guidance | Nemotron Ultra 253B + Nano 8B via Nebius |
| **COSMO Entropy Pipeline** | Real-time entropy ingestion, extraction, conditioning, anchoring | Redis Streams microservices + HKDF-SHA3-256 |
| **Immutable Audit Trail** | Tamper-evident logging of all entropy operations | SHA3-256 Merkle trees + hash-chained JSONL ledger |

### Core Capabilities

- **10-class vulnerability taxonomy** — hardcoded secrets, weak hashes (MD5/SHA-1), insecure RNG, deprecated ciphers (DES/RC4), ECB mode, improper padding, timing-unsafe comparisons, nonce reuse, insufficient key lengths, missing AEAD
- **Hybrid scanning** — deterministic regex rules always run; LLM deep analysis augments when cloud mode is available
- **NIST SP 800-90B entropy analysis** — Shannon entropy, min-entropy (MCV estimator with 99% CI), chi-squared uniformity, serial correlation
- **LHL entropy extraction** — Leftover Hash Lemma via SHAKE256 XOF for near-uniform random bit generation
- **HKDF-SHA3-256 conditioning** — entropy pool with XOR mixing and standards-compliant key derivation
- **RL-optimised model routing** — PPO/DPO framework routes requests to Ultra, Nano, NeVA, or local mode based on complexity
- **Post-quantum guidance** — recommends CRYSTALS-Kyber (ML-KEM), CRYSTALS-Dilithium (ML-DSA), and FALCON

---

## Architecture

```
                         ┌────────────────────────────────────────────────────────┐
                         │                  CryptoGuard AI Gateway                │
                         │                                                        │
  Users ──── Frontend ───┤  /api/scan    Hybrid rule + LLM vulnerability scanner  │
             (SPA)       │  /api/entropy NIST 800-90B analysis + LHL extraction   │
                         │  /api/chat    Crypto security advisor (Ultra/Local)    │
                         │  /api/pipeline Pipeline control + ledger queries       │
                         │  /ws/pipeline  Real-time stream monitoring             │
                         │                                                        │
                         │  Middleware: rate limiting (token bucket) + telemetry   │
                         │  RL Engine:  PPO routing policy + reward functions      │
                         └───────┬──────────────────────────┬─────────────────────┘
                                 │                          │
                    ┌────────────▼────────────┐   ┌────────▼────────────────────┐
                    │    Nebius AI Cloud       │   │   COSMO Entropy Pipeline    │
                    │                         │   │                             │
                    │  Nemotron Ultra 253B    │   │  Ingestion (MQTT/gRPC)      │
                    │  Nemotron Nano 8B       │   │       ▼                     │
                    │  NeVA (multimodal)      │   │  Extraction (LHL/SHAKE256)  │
                    └─────────────────────────┘   │       ▼                     │
                                                  │  Mixing Pool (HKDF-SHA3)   │
                    ┌─────────────────────────┐   │       ▼                     │
                    │     cosmo_core lib       │   │  Anchoring (Merkle + JSONL)│
                    │                         │   │                             │
                    │  entropy / extraction   │   │  Transport: Redis Streams   │
                    │  conditioning / nist    │   └─────────────────────────────┘
                    │  taxonomy / anchoring   │
                    │  fingerprint            │
                    └─────────────────────────┘
```

### Tech Stack

| Layer | Technology |
|-------|-----------|
| **AI Models** | NVIDIA Nemotron Ultra 253B, Nano 8B, NeVA (multimodal) |
| **AI Infrastructure** | Nebius AI Cloud / Token Factory (OpenAI-compatible API) |
| **Core Library** | Python 3.11+, NumPy, SciPy, `cryptography` (HKDF-SHA3-256) |
| **Gateway** | FastAPI, Pydantic v2, pydantic-settings, uvicorn |
| **Pipeline** | Redis Streams, async microservices, MQTT/gRPC ingestion |
| **Frontend** | Vanilla HTML/CSS/JS, Web Crypto API, glassmorphism UI |
| **Deployment** | Docker Compose, Kubernetes (k8s manifests), GitHub Pages |

---

## Quick Start

### Option 1: Static Demo (no API key)

```bash
python serve.py
# Opens http://127.0.0.1:8080 with the demo dashboard
```

### Option 2: Standalone Backend

```bash
pip install -r requirements.txt
export NEBIUS_API_KEY=your_key_here        # or set in .env
python app.py
# API at http://localhost:8000, dashboard at http://localhost:8000/
```

### Option 3: Production Gateway

```bash
pip install -e ".[dev]"                    # installs cosmo_core + gateway
cp .env.example .env                       # configure API keys
uvicorn gateway.main:app --host 0.0.0.0 --port 8000
```

### Option 4: Full Stack with Docker Compose

```bash
cp .env.example .env                       # fill in CG_NEBIUS_API_KEY
docker compose up --build
# Gateway:   http://localhost:8000
# Redis:     localhost:6379
# Pipeline:  ingestion → extraction → mixing-pool → anchoring
```

---

## API Reference

### Health Check
```
GET /api/health
```
Returns system status, available capabilities, AI models, and connection state.

### Code Scanner (Patent Claim 2)
```
POST /api/scan
Content-Type: application/json

{
  "code": "import hashlib\nmd5 = hashlib.md5(password)",
  "language": "python"
}
```
Returns severity-ranked findings with CWE references, line numbers, and fix recommendations. Hybrid approach: deterministic rules always run, LLM augments in cloud mode.

### Entropy Analysis (Patent Claims 3-5)
```
POST /api/entropy
Content-Type: application/json

{
  "data": "a3f2c1..."   // hex-encoded bytes (min 16 bytes)
}
```
Returns Shannon entropy, min-entropy, chi-squared, serial correlation, NIST SP 800-90B checks, LHL extraction results, and AI assessment.

### AI Chat
```
POST /api/chat
Content-Type: application/json

{
  "message": "How should I migrate from RSA-2048 to post-quantum?",
  "context": "Python backend using cryptography library"
}
```

### Pipeline (Patent Claim 6)
```
POST /api/pipeline/submit       # Submit timestamp events
GET  /api/pipeline/status       # Stream lengths and connection state
GET  /api/pipeline/ledger       # Query anchored ledger entries
WS   /ws/pipeline               # Real-time stream monitoring
```

---

## Patent Claims

This system implements an **Indian patent application (IDF-B filed)** covering seven claims:

| Claim | Title | Implementation |
|-------|-------|----------------|
| **1** | AI-driven cryptographic security analysis system | `src/gateway/` — unified FastAPI gateway with multi-model AI routing |
| **2** | 10-class vulnerability taxonomy with hybrid scanning | `src/cosmo_core/taxonomy.py` — regex rules + LLM deep analysis |
| **3** | NIST SP 800-90B entropy estimation with LHL extraction | `src/cosmo_core/extraction.py` — MCV estimator, SHAKE256 XOF |
| **4** | HKDF-SHA3-256 entropy conditioning pipeline | `src/cosmo_core/conditioning.py` — XOR mixing pool + HKDF |
| **5** | Multi-metric entropy quality assessment | `src/cosmo_core/entropy.py` — Shannon, min-entropy, chi-squared, serial correlation |
| **6** | Real-time entropy pipeline with Redis Streams | `project-cosmo/services/` — 4-stage microservice pipeline |
| **7** | Immutable audit trail with Merkle tree anchoring | `src/cosmo_core/anchoring.py` — SHA3-256 Merkle root + hash-chained JSONL |

---

## Project Structure

```
cryptoguard-ai/
├── src/
│   ├── cosmo_core/                  # Shared cryptographic core library
│   │   ├── entropy.py               #   Shannon, min-entropy, chi-squared, serial correlation
│   │   ├── extraction.py            #   LHL entropy extraction via SHAKE256 XOF
│   │   ├── conditioning.py          #   HKDF-SHA3-256 entropy pool
│   │   ├── anchoring.py             #   Merkle tree + hash-chained JSONL ledger
│   │   ├── taxonomy.py              #   10-class vulnerability taxonomy + regex scanner
│   │   ├── nist_800_90b.py          #   NIST SP 800-90B sanity checks
│   │   └── fingerprint.py           #   SHA3-512 fingerprinting
│   └── gateway/                     # Production API gateway
│       ├── main.py                  #   FastAPI app with lifespan management
│       ├── config.py                #   Pydantic settings (env-driven)
│       ├── dependencies.py          #   DI providers (Redis, AI client, RL)
│       ├── ai/                      #   AI backends
│       │   ├── nim_client.py        #     Nebius/Nemotron cloud client
│       │   ├── local_fallback.py    #     Offline knowledge-base client
│       │   └── neva_client.py       #     NeVA multimodal client
│       ├── routers/                 #   API endpoints
│       │   ├── scan.py              #     /api/scan — hybrid code scanner
│       │   ├── entropy.py           #     /api/entropy — NIST analysis
│       │   ├── chat.py              #     /api/chat — AI advisor
│       │   ├── pipeline.py          #     /api/pipeline — stream control
│       │   └── health.py            #     /api/health — system status
│       ├── ws/pipeline.py           #   WebSocket real-time pipeline events
│       ├── middleware/              #   Rate limiting + structured telemetry
│       ├── rl/                      #   Reinforcement learning
│       │   ├── optimizer.py         #     PPO policy gradient + DPO pairs
│       │   ├── reward.py            #     Security/latency/consistency rewards
│       │   └── feedback.py          #     Inference recording + Redis flush
│       └── models/                  #   Pydantic request/response schemas
│
├── project-cosmo/                   # COSMO entropy pipeline (microservices)
│   ├── services/
│   │   ├── ingestion/               #   MQTT/gRPC → Redis Streams
│   │   ├── extraction/              #   LHL entropy extraction
│   │   ├── mixing-pool/             #   HKDF-SHA3-256 conditioning
│   │   └── anchoring/               #   Merkle tree + ledger commit
│   ├── k8s/                         #   Kubernetes manifests
│   └── web/                         #   COSMO standalone dashboard
│
├── tests/
│   ├── test_cosmo_core/             # Unit tests: entropy, conditioning, taxonomy, etc.
│   └── test_gateway/                # Integration tests: health, scan, entropy, chat
│
├── web/                             # Frontend dashboard (SPA)
├── k8s/                             # Gateway Kubernetes manifests
├── patent-documents/                # IDF-B forms and course submission
├── hackathon-submission/            # Demo script and checklist
│
├── app.py                           # Standalone FastAPI backend (quick demo)
├── serve.py                         # Static file server (no-dependency demo)
├── docker-compose.yml               # Full stack: gateway + Redis + pipeline services
├── Dockerfile                       # Standalone container
├── Dockerfile.gateway               # Production gateway container
├── pyproject.toml                   # Package config + test settings
├── requirements.txt                 # Standalone dependencies
└── .env.example                     # Environment configuration template
```

---

## Configuration

All settings are driven by environment variables (prefix `CG_`). Copy `.env.example` to `.env`:

| Variable | Default | Description |
|----------|---------|-------------|
| `CG_NEBIUS_API_KEY` | — | Nebius AI API key (enables cloud mode) |
| `CG_NIM_BASE_URL` | `https://api.studio.nebius.com/v1/` | NIM-compatible endpoint |
| `CG_MODEL_ULTRA` | `nvidia/llama-3.1-nemotron-ultra-253b-v1` | Deep analysis model |
| `CG_MODEL_NANO` | `nvidia/llama-3.1-nemotron-nano-8b-v1` | Fast inference model |
| `CG_NEVA_ENDPOINT` | — | NeVA multimodal endpoint (optional) |
| `CG_REDIS_URL` | — | Redis URL (enables pipeline mode) |
| `CG_RL_FEEDBACK_ENABLED` | `false` | Enable RL feedback collection |
| `CG_RATE_LIMIT_RPM` | `60` | API rate limit (requests/minute) |
| `CG_LOG_LEVEL` | `INFO` | Logging verbosity |

**Modes:**
- **Cloud mode** — set `CG_NEBIUS_API_KEY` to enable Nemotron Ultra/Nano inference
- **Local mode** — runs without API key using cosmo_core deterministic analysis + knowledge base
- **Pipeline mode** — set `CG_REDIS_URL` to enable the real-time entropy pipeline

---

## Testing

```bash
pip install -e ".[dev]"
pytest                          # run all tests
pytest tests/test_cosmo_core/   # core library only
pytest tests/test_gateway/      # gateway integration only
```

Tests cover entropy metric accuracy, NIST SP 800-90B checks, conditioning correctness, taxonomy detection rates, and gateway endpoint behaviour.

---

## Deployment

### Kubernetes

```bash
kubectl apply -f k8s/gateway/                  # gateway deployment + service
kubectl apply -f project-cosmo/k8s/            # full COSMO pipeline
```

### GitHub Pages (frontend only)

The repository includes a GitHub Actions workflow (`.github/workflows/pages.yml`) that deploys the dashboard to GitHub Pages on push to `master`.

---

## Standards & References

- **NIST SP 800-90B** — Recommendation for the Entropy Sources Used for Random Bit Generation
- **NIST SP 800-90C** — Recommendation for Random Bit Generator (RBG) Constructions
- **NIST SP 800-22** — Statistical Test Suite for Random and Pseudorandom Number Generators
- **NIST SP 800-131A Rev 2** — Transitioning the Use of Cryptographic Algorithms and Key Lengths
- **Leftover Hash Lemma** — Impagliazzo, Levin, Luby (1989) — privacy amplification via universal hashing
- **HKDF (RFC 5869)** — HMAC-based Extract-and-Expand Key Derivation Function
- **CRYSTALS-Kyber (ML-KEM)** — NIST post-quantum key encapsulation (FIPS 203)
- **CRYSTALS-Dilithium (ML-DSA)** — NIST post-quantum digital signatures (FIPS 204)
- **OWASP Top 10 A02:2021** — Cryptographic Failures

---

## Hackathon

**Best Apps and Agents** — Nebius x NVIDIA Global AI Hackathon

- **Real problem**: cryptographic failures are the #2 risk in OWASP Top 10
- **Multi-model intelligence**: Nemotron Ultra for deep reasoning, Nano for fast checks, NeVA for image analysis
- **Research depth**: NIST SP 800-90B compliance, Leftover Hash Lemma extraction, Merkle-tree anchoring
- **Production-ready**: Docker Compose, Kubernetes manifests, RL-optimised routing, structured telemetry
- **Patent-backed**: seven claims covering the full system architecture

---

<p align="center">
  <strong>Swagata Banerjee</strong> &middot; VIT Vellore &middot; Nebius x NVIDIA Global AI Hackathon 2025
</p>
