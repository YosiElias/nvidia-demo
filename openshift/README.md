# NVIDIA Code Documentation Agents - RHOAI Deployment

This package contains the Red Hat OpenShift AI (RHOAI) deployment version of the [NVIDIA Code Documentation Agents demo](https://github.com/crewAIInc/nvidia-demo), showcasing how to run NVIDIA's blueprint on RHOAI with self-hosted NVIDIA NIM (NVIDIA Inference Microservices).

## Overview

This RHOAI adaptation demonstrates:
- Running CrewAI multi-agent workflows on OpenShift AI
- Deploying self-hosted NVIDIA NIM models (LLM + Embeddings)
- Configuring agents for internal cluster endpoints
- Automated code documentation generation using AI agents

## Quick Start

### Prerequisites
- RHOAI workbench access
- NVIDIA NIM enabled on your RHOAI cluster
- Self-hosted NIMs deployed (Llama 3.1 8B + NeMo Retriever E5)

### 3-Step Setup

1. **Set Environment Variables** (in RHOAI workbench settings):
   ```
   OPENAI_API_BASE=https://your-llama-endpoint.apps.cluster.example.com/v1
   OPENAI_API_KEY=your_service_account_token
   NVIDIA_EMBED_BASE_URL=https://your-embedding-endpoint.apps.cluster.example.com/v1
   NVIDIA_EMBED_TOKEN=your_embedding_token
   ```

2. **Open Notebook**:
   ```
   openshift/Building_Code_Documentation_Agents_RHOAI.ipynb
   ```

3. **Run All Cells**:
   - Auto-setup cells copy configs and change directory
   - Documentation agents analyze your codebase
   - Generated documentation files will be created in the `docs/` directory

## Technical Adaptations

This RHOAI version includes 6 key adaptations from the original demo (see [CHANGES.md](CHANGES.md) for technical details):

1. **SQLite Compatibility Patch** - Use pysqlite3-binary for SQLite 3.35+ support (RHOAI workbench images inherit SQLite 3.34 from UBI base image's system packages)

2. **Model Configuration** - Changed from Llama 3.3 70B (NVIDIA API Catalog) to Llama 3.1 8B (self-hosted NIM) due to GPU memory constraints in the tested cluster

3. **Environment Variables** - Added NVIDIA_EMBED_BASE_URL and NVIDIA_EMBED_TOKEN for self-hosted embedding endpoint authentication

4. **WebsiteSearchTool Configuration** - Changed from `provider="nvidia"` (uses NvidiaEmbedder with auto passage/query switching) to `provider="openai"` (uses OpenAI-compatible embedder) pointing to self-hosted NIM endpoints, requiring manual mode switching (details in notebook)

5. **switch_embedding_mode Function** - Helper function to switch between passage (indexing) and query (searching) modes for accurate retrieval

6. **Auto-Setup Cells** - Automatic configuration file copying and directory management for plug-and-play experience

**Note:** This demo uses Llama 3.1 8B, but other models can be used. See the "Using a Different Model" section in [SETUP.md](SETUP.md) for instructions.

## Documentation

- **[SETUP.md](SETUP.md)** - Complete deployment guide including NIM setup
- **[CHANGES.md](CHANGES.md)** - Technical details of all adaptations
- **[Original Demo](https://github.com/crewAIInc/nvidia-demo)** - NVIDIA API Catalog version

## Requirements

All dependencies are in the root `requirements.txt`.

**RHOAI-specific addition:**
- `pysqlite3-binary>=0.5.4` - Added for SQLite 3.35+ compatibility (RHOAI workbench images inherit SQLite 3.34 from UBI base image's system packages)

## Support

For RHOAI deployment assistance, please open an issue on this fork's repository.

---

**Note:** This is an RHOAI adaptation of the [official NVIDIA demo](https://github.com/crewAIInc/nvidia-demo). For the original version using NVIDIA API Catalog, see the main repository.
