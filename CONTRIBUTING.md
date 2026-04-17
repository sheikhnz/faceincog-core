# Contributing to FaceIncog Core

Thanks for your interest in contributing! This project is a real-time AI face-swapping engine — contributions that improve performance, hardware support, mask types, or developer experience are all welcome.

---

## Table of Contents

- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Code Style](#code-style)
- [Submitting a Pull Request](#submitting-a-pull-request)
- [What to Contribute](#what-to-contribute)
- [What Not to Contribute](#what-not-to-contribute)

---

## Getting Started

1. **Fork** the repository and clone your fork locally
2. Create a new branch for your change:
   ```bash
   git checkout -b feat/your-feature-name
   ```
3. Make your changes, then open a Pull Request against `main`

---

## Development Setup

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install Ruff (linter + formatter)
pip install ruff

# Download AI models
python scripts/download_models.py
```

> The `buffalo_l` InsightFace models auto-download on first run into `~/.insightface/models/`. This takes ~1 min the first time.

---

## Code Style

This project uses **[Ruff](https://docs.astral.sh/ruff/)** for linting and formatting. All PRs must pass the CI lint check.

```bash
# Check for lint issues
ruff check .

# Auto-fix everything fixable
ruff check . --fix

# Format all files
ruff format .
```

The CI will automatically run both `ruff check` and `ruff format --check` on every PR. **PRs that fail lint will not be merged.**

Config lives in [`pyproject.toml`](./pyproject.toml).

---

## Submitting a Pull Request

- **Keep PRs focused** — one feature or fix per PR
- **Write a clear PR description** — explain what changed and why
- **Add or update comments** for any non-obvious logic, especially in the pipeline or mask code
- **Test your change** locally before opening a PR:
  - Desktop mode: `python main.py --mask demo_deepfake`
  - WebRTC mode: `python server.py --mask demo_deepfake`, then open `http://localhost:8080`
- Confirm lint passes: `ruff check . && ruff format --check .`

---

## What to Contribute

Great areas to contribute:

| Area | Examples |
|---|---|
| **Performance** | Faster frame processing, better queue management, batching |
| **Hardware support** | Improved CUDA/CoreML/ROCm detection, Apple Silicon optimizations |
| **New mask types** | GAN masks, style transfer, landmark-only masks |
| **WebRTC improvements** | Better ICE handling, multi-client support, adaptive bitrate |
| **Developer experience** | Better error messages, logging, CLI improvements |
| **Tests** | Unit tests for `parser.py`, `registry.py`, mask loading logic |
| **Docs** | Tutorials, architecture diagrams, configuration guides |
| **Docker** | ARM64 support, smaller image sizes, GPU auto-detection improvements |

---

## What Not to Contribute

Please **do not** open PRs for:

- Features that enable non-consensual use of someone's likeness
- Bypassing safety or ethical guardrails
- Proprietary model weights or copyrighted assets
- Massive refactors without prior discussion (open an issue first)

---

## Questions?

Open a [GitHub Issue](../../issues) and use the **Question** template. We're happy to help!
