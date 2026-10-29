<p align="center">
  <img src="https://img.shields.io/badge/NVIDIA-Nemotron_Ultra-76B900?style=for-the-badge&logo=nvidia&logoColor=white" alt="Nemotron">
  <img src="https://img.shields.io/badge/Nebius-AI_Cloud-4285F4?style=for-the-badge" alt="Nebius">
  <img src="https://img.shields.io/badge/Post--Quantum-Ready-7C3AED?style=for-the-badge" alt="PQ">
  <img src="https://img.shields.io/badge/NIST_SP_800--90B-Compliant-00897B?style=for-the-badge" alt="NIST">
  <img src="https://img.shields.io/badge/License-MIT-blue?style=for-the-badge" alt="MIT">
</p>

<h1 align="center">CryptoGuard AI</h1>

<h3 align="center">AI-Powered Cryptographic Security Operations Center</h3>

<p align="center">
  An intelligent security agent that scans code for cryptographic vulnerabilities, analyzes entropy quality, and provides post-quantum migration guidance — powered by <strong>NVIDIA Nemotron</strong> on <strong>Nebius AI Cloud</strong>.
</p>

---

## What It Does

CryptoGuard AI is a multi-capability security agent that helps developers find and fix cryptographic weaknesses in their code:

| Feature | Model | Description |
|---------|-------|-------------|
| **Code Scanner** | Nemotron Ultra 253B | Paste code → get severity-ranked vulnerability report with fix recommendations |
| **Entropy Lab** | Nemotron Nano 8B | Generate random bytes, compute Shannon/min-entropy/chi-squared, get AI quality assessment |
| **AI Chat** | Nemotron Ultra 253B | Ask anything about cryptography, post-quantum security, key management |
| **Dashboard** | — | Real-time stats, threat overview, quick actions |

### Key Capabilities

- Detects **10+ vulnerability classes**: hardcoded secrets, weak algorithms (MD5/SHA1/DES), insecure RNG, improper KDFs, ECB mode, timing attacks, insufficient key lengths
- **NIST SP 800-90B** entropy analysis with Shannon entropy, min-entropy (MCV), chi-squared, and serial correlation
- **Post-quantum guidance**: recommends CRYSTALS-Kyber, Dilithium, FALCON as replacements for RSA/ECDSA
- **Multi-model routing**: Ultra for deep analysis, Nano for fast entropy checks

---

## How We Built It

### Architecture

```
┌──────────────────────────────────────────────────────────┐
│                    CryptoGuard AI                         │
│                                                           │
│  ┌─────────┐  ┌───────────┐  ┌─────────┐  ┌──────────┐ │
│  │ Scanner  │  │ Entropy   │  │  Chat   │  │Dashboard │ │
│  │  Page    │  │   Lab     │  │  Agent  │  │ Overview │ │
│  └────┬─────┘  └─────┬─────┘  └────┬────┘  └──────────┘ │
│       │              │              │                     │
│  ┌────▼──────────────▼──────────────▼────┐               │
│  │         FastAPI Backend               │               │
│  │  /api/scan  /api/entropy  /api/chat   │               │
│  └────────────────┬──────────────────────┘               │
│                   │                                       │
└───────────────────┼───────────────────────────────────────┘
                    │
        ┌───────────▼───────────┐
        │   Nebius AI Cloud     │
        │                       │
        │  ┌─────────────────┐  │
        │  │ Nemotron Ultra  │  │  ← Code scanning, chat
        │  │    253B         │  │
        │  └─────────────────┘  │
        │  ┌─────────────────┐  │
        │  │ Nemotron Nano   │  │  ← Fast entropy analysis
        │  │    8B           │  │
        │  └─────────────────┘  │
        └───────────────────────┘
```

### Tech Stack

| Layer | Technology |
|-------|-----------|
| **AI Models** | NVIDIA Nemotron Ultra 253B (reasoning), Nemotron Nano 8B (fast inference) |
| **AI Infrastructure** | Nebius AI Cloud / Token Factory (OpenAI-compatible API) |
| **Backend** | Python, FastAPI, Pydantic |
| **Frontend** | Vanilla HTML/CSS/JS, Web Crypto API |
| **Crypto Analysis** | NIST SP 800-90B, Shannon entropy, min-entropy, chi-squared, serial correlation |
| **Deployment** | Docker, Nebius Serverless |

---

## Quick Start

### 1. Static Demo (no API key needed)

```bash
python serve.py
# Opens http://127.0.0.1:8080
# Dashboard works in demo mode with pre-computed responses
```

### 2. Full Backend with Nemotron

```bash
pip install -r requirements.txt
export NEBIUS_API_KEY=your_key_here
python app.py
# Server at http://0.0.0.0:8000
```

### 3. Docker

```bash
docker build -t cryptoguard-ai .
docker run -p 8000:8000 -e NEBIUS_API_KEY=your_key cryptoguard-ai
```

---

## Demo Flow

1. **Dashboard** — See the security operations overview
2. **Code Scanner** — Load the example vulnerable Python code → Scan → See 6 findings with severity badges and fix recommendations
3. **Entropy Lab** — Generate 256 random bytes → See Shannon entropy, min-entropy, chi-squared gauges → Read AI assessment
4. **AI Chat** — Ask "How do I make my code quantum-safe?" → Get detailed post-quantum migration plan

---

## Hackathon Track

**Best Apps and Agents** — Nebius x NVIDIA Global AI Hackathon

### Why This Wins

- **Real problem**: Cryptographic vulnerabilities are the #1 cause of data breaches
- **Multi-model intelligence**: Nemotron Ultra for deep reasoning, Nano for fast checks
- **Production-ready**: FastAPI backend, Docker deployment, Nebius Serverless compatible
- **Unique angle**: No other tool combines AI code scanning + entropy lab + PQ guidance
- **Research depth**: Built on NIST SP 800-90B standards, Leftover Hash Lemma, real entropy analysis

---

## Project Structure

```
cryptoguard-ai/
├── app.py              # FastAPI backend + Nemotron API integration
├── web/
│   └── index.html      # Full frontend — dashboard, scanner, entropy lab, chat
├── requirements.txt    # Python dependencies
├── Dockerfile          # Container deployment
├── serve.py            # Quick static server for demo mode
├── .env.example        # API key configuration
└── README.md
```

---

## Standards & References

- **NIST SP 800-90B** — Entropy source validation and min-entropy estimation
- **CRYSTALS-Kyber** (ML-KEM) — NIST post-quantum key encapsulation
- **CRYSTALS-Dilithium** (ML-DSA) — NIST post-quantum digital signatures
- **OWASP Cryptographic Failures** — A02:2021 in OWASP Top 10

---

<p align="center">
  <strong>Swagata Banerjee</strong> · VIT Vellore · Nebius x NVIDIA Global AI Hackathon 2026
</p>
