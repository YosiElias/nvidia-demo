# Technical Adaptations for RHOAI Deployment

This document details the technical changes required to run the NVIDIA Code Documentation Agents demo on Red Hat OpenShift AI with self-hosted NVIDIA NIMs.

---

## 1. SQLite Compatibility Patch

**Problem:**
WebsiteSearchTool's vector storage fails with error:
```
SQLite version 3.35.0 or higher required
```

**Root Cause:**
RHOAI workbench images inherit SQLite 3.34 from UBI base image's system packages, but the vector database requires SQLite 3.35+ for its operations.

**Solution:**
Python's `sqlite3` module was redirected to use `pysqlite3-binary`, which bundles SQLite 3.51:

```python
# This cell was added at the very beginning of the notebook, before any other imports
import sys
__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
```

**Dependencies:**
- Added to `requirements.txt`: `pysqlite3-binary>=0.5.4`

---

## 2. Model Change: Llama 3.3 70B → Llama 3.1 8B

**Problem:**
Original demo uses `meta/llama-3.3-70b-instruct` from NVIDIA API Catalog, which requires significant GPU memory.

**Root Cause:**
The tested RHOAI cluster deployment has GPU memory constraints that cannot accommodate 70B parameter models.

**Solution:**
Model was changed to `meta/llama-3.1-8b-instruct` in agent configurations:

**File:** `config/documentation_agents.yaml`
```yaml
overview_writer:
  llm: openai/meta/llama-3.1-8b-instruct

documentation_reviewer:
  llm: openai/meta/llama-3.1-8b-instruct
```

**File:** `config/planner_agents.yaml`
```yaml
code_explorer:
  llm: openai/meta/llama-3.1-8b-instruct

documentation_planner:
  llm: openai/meta/llama-3.1-8b-instruct
```

**Note:**
- RHOAI supports larger models with adequate GPU resources. Llama 3.1 8B remains effective for code documentation tasks.
- The provider prefix changed from `nvidia/` (for NVIDIA API Catalog) to `openai/` (for self-hosted NIMs with OpenAI-compatible API)

---

## 3. New Environment Variables for Self-Hosted Embeddings

Self-hosted NeMo Retriever E5 embedding model requires separate endpoint URL and authentication token. Two new environment variables were added (set via RHOAI workbench settings).

**Environment Variables:**
- `NVIDIA_EMBED_BASE_URL` - Self-hosted embedding NIM endpoint (e.g., `https://nemo-e5-embedding-project.apps.cluster.example.com/v1`)
- `NVIDIA_EMBED_TOKEN` - Service account token for embedding authentication
- `OPENAI_API_BASE` - Self-hosted LLM NIM endpoint (existing)
- `OPENAI_API_KEY` - Service account token for LLM authentication (existing)

---

## 4. WebsiteSearchTool Configuration for Self-Hosted Embeddings

**Problem:**
WebsiteSearchTool needs to use self-hosted NVIDIA embedding NIM instead of NVIDIA API Catalog.

**Root Cause:**
Original demo uses `provider="nvidia"` which uses NvidiaEmbedder that automatically handles passage/query switching. However, NvidiaEmbedder doesn't support self-hosted endpoints. The alternative OpenAIEmbedder supports custom endpoints but doesn't support the `input_type` parameter for switching modes.

**Solution:**
WebsiteSearchTool was configured with OpenAI-compatible provider pointing to self-hosted NVIDIA NIM:

```python
from crewai_tools import WebsiteSearchTool

tool = WebsiteSearchTool(
    website="https://mermaid.js.org/intro/",
    config=dict(
        embedder=dict(
            provider="openai",  # Use OpenAI-compatible interface
            config=dict(
                model="nvidia/nv-embedqa-e5-v5-passage",  # NVIDIA embedding model (passage mode for indexing)
                api_base=os.environ.get("NVIDIA_EMBED_BASE_URL"),
                api_key=os.environ.get("NVIDIA_EMBED_TOKEN")
            )
        )
    )
)
```

**Key Points:**
- `provider="openai"` - Uses OpenAIEmbedder which supports custom endpoints (NVIDIA NIMs expose OpenAI-compatible API)
- `model="nvidia/nv-embedqa-e5-v5-passage"` - Uses passage mode for document indexing (NVIDIA supports mode suffix in model name as workaround for missing `input_type` parameter)
- Custom `api_base` and `api_key` point to self-hosted NIM

---

## 5. switch_embedding_mode Helper Function

**Problem:**
NVIDIA's `nv-embedqa-e5-v5` embedding model requires different modes:
- `passage` mode - For indexing/embedding documents
- `query` mode - For search/embedding queries

Using the wrong mode causes degraded retrieval accuracy.

**Root Cause:**
Using OpenAIEmbedder (instead of NvidiaEmbedder) is required because NvidiaEmbedder doesn't support self-hosted models. However, OpenAIEmbedder doesn't support the `input_type` parameter for switching between passage and query modes. NVIDIA supports using mode suffixes (`-passage`, `-query`) in the model name as a workaround, but this requires manually switching the embedding function between indexing and querying phases.

**Solution:**
Helper function was created to switch embedding modes by directly replacing the embedding function:

```python
def switch_embedding_mode(tool, mode: str):
    """
    Switch embedding model between 'query' and 'passage' modes.
    Use 'passage' for indexing, 'query' for searching.
    """
    from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

    if mode not in ["query", "passage"]:
        raise ValueError(f"Invalid mode '{mode}'. Must be 'query' or 'passage'.")

    embedding_function = OpenAIEmbeddingFunction(
        api_key=os.environ.get("NVIDIA_EMBED_TOKEN"),
        api_base=os.environ.get("NVIDIA_EMBED_BASE_URL"),
        model_name=f"nvidia/nv-embedqa-e5-v5-{mode}"
    )

    tool.adapter.embedchain_app.db.collection._embedding_function = embedding_function
    print(f"✓ Switched to '{mode}' mode (model: nvidia/nv-embedqa-e5-v5-{mode})")
```

**Usage:**
```python
# 1. Create tool with passage mode (for indexing)
tool = WebsiteSearchTool(...)  # configured with -passage model

# 2. After indexing, switch to query mode
switch_embedding_mode(tool, "query")

# 3. Now queries use correct query mode
```

**Technical Details:**
- Model name suffixes (`-passage`, `-query`) are NVIDIA's supported workaround for the missing `input_type` parameter in OpenAIEmbedder
- Direct manipulation of `collection._embedding_function` allows switching modes after collection creation
- Ensures correct modes: passage during indexing, query during searching

---

## 6. Notebook Auto-Setup Cells

**Problem:**
Notebook needs to be in `openshift/` directory for organization, but the workflow expects to run from project root.

**Root Cause:**
CrewAI searches for configuration files relative to the working directory, and the workflow expects `config/` directory, `docs/` directory, and `.env` file in the current path. If notebook runs from `openshift/`, it won't find these files or outputs will go to wrong locations.

**Solution:**
Auto-setup cells were added at the beginning of the notebook:

**Cell 1 - Copy RHOAI Configs:**
```python
import shutil
import os
from pathlib import Path

# Get current notebook directory (openshift/)
notebook_dir = Path(os.getcwd())

# Copy RHOAI configs from openshift/config/ to root config/
openshift_config = notebook_dir / 'config'
root_config = notebook_dir.parent / 'config'

print("Copying RHOAI configuration files...")
for config_file in openshift_config.glob('*.yaml'):
    dst = root_config / config_file.name
    shutil.copy(config_file, dst)
    print(f"  ✓ Copied {config_file.name}")
```

**Cell 2 - Change Working Directory:**
```python
# Change to project root so workflow finds config/, docs/, and .env in correct locations
project_root = notebook_dir.parent
os.chdir(project_root)
print(f"✓ Working directory: {os.getcwd()}")
```

**Result:**
- RHOAI agent configs copied to root `config/` directory
- Working directory changed to project root
- CrewAI finds configurations correctly
- Generated documentation goes to `docs/` directory in project root

---

## 7. Connection Retry for WebsiteSearchTool

**Problem:**
WebsiteSearchTool occasionally fails during Agent initialization with `ConnectionResetError: [Errno 104] Connection reset by peer` when scraping external websites.

**Solution:**
Added retry loop (3 attempts with 2s delay) around Agent initialization:

```python
import time

for attempt in range(3):
    try:
        overview_writer = Agent(config=agents_config['overview_writer'], tools=[...])
        break
    except ConnectionError:
        if attempt < 2:
            print(f"Connection failed, retrying... (attempt {attempt + 1}/3)")
            time.sleep(2)
        else:
            print("Connection failed after 3 attempts, continuing anyway...")
```

---

## Summary

These six adaptations enable the NVIDIA Code Documentation Agents demo to run on RHOAI with self-hosted NIMs:

1. **SQLite Patch** - Use bundled SQLite 3.51 via pysqlite3-binary
2. **Model Change** - Use Llama 3.1 8B for resource constraints
3. **Environment Variables** - Add NVIDIA_EMBED_BASE_URL and NVIDIA_EMBED_TOKEN
4. **WebsiteSearchTool** - Configure with self-hosted embedding endpoint
5. **switch_embedding_mode** - Ensure correct embedding modes for indexing/querying
6. **Auto-Setup Cells** - Copy configs and change directory automatically

All changes maintain compatibility with the original demo workflow while adapting to RHOAI's self-hosted deployment model.
