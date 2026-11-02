<p align="center">
  <br>
  <img width="80" src="https://img.icons8.com/fluency/96/shield.png" alt="CryptoGuard AI">
  <br>
</p>

<h1 align="center">CryptoGuard AI</h1>

<p align="center">
  <strong>AI-Powered Cryptographic Security Operations Center</strong><br>
  Vulnerability scanning &bull; NIST 800-90B entropy analysis &bull; Post-quantum guidance &bull; Blockchain-anchored audit trails
</p>

<p align="center">
  <img src="https://img.shields.io/badge/NVIDIA-Nemotron_Ultra-76B900?style=flat-square&logo=nvidia&logoColor=white" alt="Nemotron">
  <img src="https://img.shields.io/badge/Nebius-AI_Cloud-4285F4?style=flat-square" alt="Nebius">
  <img src="https://img.shields.io/badge/NIST_SP_800--90B-Compliant-00897B?style=flat-square" alt="NIST">
  <img src="https://img.shields.io/badge/Post--Quantum-Ready-7C3AED?style=flat-square" alt="PQ">
  <img src="https://img.shields.io/badge/Patent-IDF--B_Filed-E65100?style=flat-square" alt="Patent">
  <img src="https://img.shields.io/badge/Python-3.11+-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/License-MIT-blue?style=flat-square" alt="MIT">
</p>

<p align="center">
  <a href="#-quick-start">Quick Start</a> &nbsp;&middot;&nbsp;
  <a href="#-how-it-works">How It Works</a> &nbsp;&middot;&nbsp;
  <a href="#-api-reference">API</a> &nbsp;&middot;&nbsp;
  <a href="#-patent-claims">Patent</a> &nbsp;&middot;&nbsp;
  <a href="#-deployment">Deploy</a>
</p>

---

> **Cryptographic failures are the #2 risk in the OWASP Top 10.** Most teams catch SQL injection but ship code with `hashlib.md5()`, hardcoded secrets, and ECB mode. CryptoGuard AI finds these — automatically, with severity rankings, fix recommendations, and post-quantum migration paths.

---

## Try It in 30 Seconds

```bash
python serve.py          # zero dependencies — opens the dashboard at localhost:8080
```

Or with full AI analysis:

```bash
pip install -r requirements.txt
export NEBIUS_API_KEY=your_key
python app.py            # API + dashboard at localhost:8000
```

---

## What It Does

<table>
<tr>
<td width="33%" valign="top">

### Scan Code
Paste any code snippet. Get severity-ranked vulnerability findings across **10 cryptographic weakness classes** — from hardcoded secrets to missing AEAD. Hybrid engine: deterministic regex rules always run, LLM deep analysis augments in cloud mode.

</td>
<td width="33%" valign="top">

### Analyze Entropy
Submit raw bytes. Get Shannon entropy, min-entropy, chi-squared, serial correlation, NIST SP 800-90B compliance checks, and **Leftover Hash Lemma extraction** of near-uniform random bits via SHAKE256.

</td>
<td width="33%" valign="top">

### Ask the AI
Chat with a cryptographic security expert. Get post-quantum migration plans, key management audits, algorithm recommendations — backed by Nemotron Ultra 253B or an offline knowledge base.

</td>
</tr>
</table>

### The Full Platform

| Layer | What | Powered By |
|:------|:-----|:-----------|
| **AI Security Agent** | Code scanning, entropy assessment, PQ guidance, image analysis | Nemotron Ultra 253B + Nano 8B + NeVA |
| **COSMO Entropy Pipeline** | Real-time ingestion &rarr; LHL extraction &rarr; HKDF conditioning &rarr; Merkle anchoring | Redis Streams microservices |
| **RL Routing Engine** | Learns which model to use for each request type | PPO policy gradient + DPO preference pairs |
| **Immutable Audit Trail** | Tamper-evident logging of every entropy operation | SHA3-256 Merkle trees + hash-chained JSONL |

---

## How It Works

```
                    ┌─────────────────────────────────────────────────┐
                    │              CryptoGuard AI Gateway              │
                    │                                                  │
 Browser ───────────┤  POST /api/scan      Hybrid vuln scanner        │
                    │  POST /api/entropy   NIST 800-90B + LHL         │
                    │  POST /api/chat      Crypto security advisor     │
                    │  WS   /ws/pipeline   Live pipeline monitor       │
                    │                                                  │
                    │  ┌──────────────┐  ┌───────────┐  ┌──────────┐ │
                    │  │ Rate Limiter │  │ Telemetry │  │ RL Router│ │
                    │  └──────────────┘  └───────────┘  └──────────┘ │
                    └─────────┬──────────────────────────┬────────────┘
                              │                          │
               ┌──────────────▼──────────────┐  ┌───────▼──────────────────────┐
               │      Nebius AI Cloud         │  │    COSMO Entropy Pipeline     │
               │                              │  │                               │
               │  Nemotron Ultra  (253B)      │  │  Ingestion ──► Extraction     │
               │  Nemotron Nano   (8B)        │  │                  │            │
               │  NeVA            (multimodal)│  │  Mixing Pool ◄───┘            │
               └──────────────────────────────┘  │       │                       │
                                                 │  Anchoring (Merkle + Ledger)  │
               ┌──────────────────────────────┐  │                               │
               │       cosmo_core library      │  │  Transport: Redis Streams     │
               │                              │  └───────────────────────────────┘
               │  entropy    extraction       │
               │  conditioning   anchoring    │
               │  taxonomy   nist_800_90b     │
               └──────────────────────────────┘
```

**Two operating modes:**

- **Cloud mode** — set `CG_NEBIUS_API_KEY` and get Nemotron-powered deep analysis
- **Local mode** — runs without any API key using deterministic cosmo_core analysis + built-in knowledge base (always works offline)

---

## API Reference

### `POST /api/scan` &mdash; Code Scanner

Hybrid vulnerability detection: deterministic rules + optional LLM deep analysis.

```json
// Request
{ "code": "import hashlib\nmd5 = hashlib.md5(password)", "language": "python" }

// Response
{
  "score": 70,
  "findings": [
    {
      "severity": "HIGH",
      "title": "Weak or broken hash function",
      "description": "Use of MD5, a cryptographically broken hash algorithm.",
      "line": 2,
      "fix": "Replace with SHA-256 or SHA-3. For password hashing use bcrypt/scrypt/Argon2.",
      "vuln_class": "WEAK_HASH"
    }
  ],
  "summary": "1 vulnerability detected (rule-based analysis)",
  "model_used": null
}
```

### `POST /api/entropy` &mdash; Entropy Analysis

NIST SP 800-90B metrics, LHL extraction, and AI assessment.

```json
// Request
{ "data": "a3f2c1b84e..." }

// Response
{
  "metrics": {
    "shannon_entropy": 7.8912,
    "min_entropy": 7.2341,
    "chi_squared": 261.44,
    "serial_correlation": 0.003211,
    "sample_size": 256,
    "quality_score": 98
  },
  "nist_checks": [
    { "name": "all_checks", "passed": true, "detail": "All NIST SP 800-90B sanity checks passed" }
  ],
  "analysis": "Entropy quality: 98%. Shannon entropy is 7.891 bits/byte ...",
  "mode": "local"
}
```

### `POST /api/chat` &mdash; AI Security Advisor

```json
// Request
{ "message": "How do I migrate from RSA-2048 to post-quantum?", "context": "Python backend" }

// Response
{ "response": "For post-quantum migration...", "model_used": "nemotron-ultra", "latency_ms": 1823.4 }
```

<details>
<summary><strong>More endpoints</strong></summary>

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/health` | System status, capabilities, connected models |
| `POST` | `/api/pipeline/submit` | Submit timestamp events to the entropy pipeline |
| `GET` | `/api/pipeline/status` | Redis stream lengths and connection state |
| `GET` | `/api/pipeline/ledger` | Query the last 50 anchored ledger entries |
| `WS` | `/ws/pipeline` | Real-time stream events via WebSocket |

</details>

---

## Patent Claims

This system implements an **Indian patent application (IDF-B filed)** covering seven independent claims:

| # | Claim | Source |
|:-:|:------|:-------|
| 1 | AI-driven cryptographic security analysis system with multi-model routing | `src/gateway/` |
| 2 | 10-class vulnerability taxonomy with hybrid rule + LLM scanning | `src/cosmo_core/taxonomy.py` |
| 3 | NIST SP 800-90B min-entropy estimation with LHL extraction via SHAKE256 | `src/cosmo_core/extraction.py` |
| 4 | HKDF-SHA3-256 entropy conditioning with XOR mixing pool | `src/cosmo_core/conditioning.py` |
| 5 | Multi-metric entropy quality assessment (Shannon, min, chi-sq, serial) | `src/cosmo_core/entropy.py` |
| 6 | Real-time 4-stage entropy pipeline over Redis Streams | `project-cosmo/services/` |
| 7 | Immutable audit trail with SHA3-256 Merkle trees + hash-chained ledger | `src/cosmo_core/anchoring.py` |

---

## Deployment

### Docker Compose (full stack)

```bash
cp .env.example .env          # fill in CG_NEBIUS_API_KEY
docker compose up --build
```

This starts: **gateway** (port 8000) + **Redis** + **ingestion** + **extraction** + **mixing-pool** + **anchoring**

### Production Gateway (standalone)

```bash
pip install -e .
uvicorn gateway.main:app --host 0.0.0.0 --port 8000
```

### Kubernetes

```bash
kubectl apply -f k8s/gateway/           # gateway deployment + service
kubectl apply -f project-cosmo/k8s/     # full COSMO pipeline with network policies
```

### GitHub Pages

Push to `master` and the dashboard auto-deploys via `.github/workflows/pages.yml`.

---

<details>
<summary><strong>Configuration</strong></summary>

All settings are env-driven with prefix `CG_`. Copy `.env.example` to `.env`:

| Variable | Default | Description |
|:---------|:--------|:------------|
| `CG_NEBIUS_API_KEY` | &mdash; | Nebius AI key (enables cloud mode) |
| `CG_NIM_BASE_URL` | `https://api.studio.nebius.com/v1/` | NIM-compatible endpoint |
| `CG_MODEL_ULTRA` | `nvidia/llama-3.1-nemotron-ultra-253b-v1` | Deep analysis model |
| `CG_MODEL_NANO` | `nvidia/llama-3.1-nemotron-nano-8b-v1` | Fast inference model |
| `CG_NEVA_ENDPOINT` | &mdash; | NeVA multimodal endpoint |
| `CG_REDIS_URL` | &mdash; | Redis URL (enables pipeline) |
| `CG_RL_FEEDBACK_ENABLED` | `false` | RL feedback collection |
| `CG_RATE_LIMIT_RPM` | `60` | Requests per minute per IP |
| `CG_LOG_LEVEL` | `INFO` | Logging verbosity |

</details>

<details>
<summary><strong>Project Structure</strong></summary>

```
cryptoguard-ai/
├── src/
│   ├── cosmo_core/                  # Shared cryptographic core library
│   │   ├── entropy.py               #   Shannon, min-entropy, chi-squared, serial correlation
│   │   ├── extraction.py            #   LHL extraction via SHAKE256 XOF
│   │   ├── conditioning.py          #   HKDF-SHA3-256 entropy pool
│   │   ├── anchoring.py             #   Merkle tree + hash-chained JSONL ledger
│   │   ├── taxonomy.py              #   10-class vulnerability taxonomy + regex scanner
│   │   ├── nist_800_90b.py          #   NIST SP 800-90B sanity checks
│   │   └── fingerprint.py           #   SHA3-512 fingerprinting
│   └── gateway/                     # Production API gateway
│       ├── main.py                  #   FastAPI app factory + lifespan
│       ├── config.py                #   Pydantic settings
│       ├── dependencies.py          #   DI providers
│       ├── ai/                      #   AI backends (NIM, local, NeVA)
│       ├── routers/                 #   API endpoints (scan, entropy, chat, pipeline, health)
│       ├── ws/                      #   WebSocket pipeline monitor
│       ├── middleware/              #   Rate limiting + structured telemetry
│       ├── rl/                      #   PPO optimizer, reward functions, feedback collection
│       └── models/                  #   Pydantic request/response schemas
│
├── project-cosmo/                   # COSMO entropy pipeline (4 microservices)
│   ├── services/{ingestion,extraction,mixing-pool,anchoring}/
│   ├── k8s/                         #   Kubernetes manifests + network policies
│   └── web/                         #   COSMO standalone dashboard
│
├── tests/
│   ├── test_cosmo_core/             # entropy, conditioning, taxonomy, NIST, statistics
│   └── test_gateway/                # health, scan, entropy, chat endpoint tests
│
├── web/                             # Frontend dashboard (SPA)
├── k8s/                             # Gateway Kubernetes manifests
├── patent-documents/                # IDF-B forms + course submission
├── hackathon-submission/            # Demo script + checklist
│
├── app.py                           # Standalone backend (quick demo)
├── serve.py                         # Zero-dependency static server
├── docker-compose.yml               # Full stack orchestration
├── Dockerfile / Dockerfile.gateway  # Container images
├── pyproject.toml                   # Package config + test settings
└── .env.example                     # Configuration template
```

</details>

<details>
<summary><strong>Testing</strong></summary>

```bash
pip install -e ".[dev]"
pytest                          # all tests
pytest tests/test_cosmo_core/   # core library
pytest tests/test_gateway/      # gateway endpoints
```

Coverage: entropy metric accuracy, NIST 800-90B check correctness, conditioning determinism, taxonomy detection rates, and gateway endpoint contracts.

</details>

<details>
<summary><strong>Tech Stack</strong></summary>

| Layer | Technology |
|:------|:-----------|
| AI Models | NVIDIA Nemotron Ultra 253B, Nano 8B, NeVA |
| AI Cloud | Nebius AI Cloud / Token Factory (OpenAI-compatible) |
| Core | Python 3.11+, NumPy, SciPy, `cryptography` |
| Gateway | FastAPI, Pydantic v2, pydantic-settings, uvicorn |
| Pipeline | Redis Streams, async microservices, MQTT/gRPC |
| Frontend | Vanilla HTML/CSS/JS, Web Crypto API |
| Deploy | Docker Compose, Kubernetes, GitHub Pages |

</details>

<details>
<summary><strong>Standards & References</strong></summary>

| Standard | Description |
|:---------|:------------|
| NIST SP 800-90B | Entropy source validation and min-entropy estimation |
| NIST SP 800-90C | Random bit generator constructions |
| NIST SP 800-22 | Statistical test suite for RNGs |
| NIST SP 800-131A Rev 2 | Cryptographic algorithm transition guidance |
| RFC 5869 | HKDF: HMAC-based Extract-and-Expand Key Derivation |
| FIPS 203 (ML-KEM) | CRYSTALS-Kyber post-quantum key encapsulation |
| FIPS 204 (ML-DSA) | CRYSTALS-Dilithium post-quantum digital signatures |
| OWASP Top 10 A02:2021 | Cryptographic Failures |
| Leftover Hash Lemma | Impagliazzo, Levin, Luby (1989) |

</details>

---

<p align="center">
  Built by <strong>Swagata Banerjee</strong> &middot; VIT Vellore<br>
  Nebius x NVIDIA Global AI Hackathon 2025
</p>
