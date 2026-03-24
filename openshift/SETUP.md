# RHOAI Deployment Setup Guide

This guide walks you through deploying the NVIDIA Code Documentation Agents demo on Red Hat OpenShift AI with self-hosted NVIDIA NIM models.

## Prerequisites

Before starting, ensure you have:

- ✅ Red Hat OpenShift AI (RHOAI) cluster access
- ✅ NVIDIA NIM operator enabled on your RHOAI cluster
- ✅ Sufficient GPU resources (minimum 1x GPU for Llama 3.1 8B + 1x GPU for NeMo E5)
- ✅ Access to create workbenches and deploy models

---

## Part 1: Deploy NVIDIA NIMs

### 1.1 Deploy Llama 3.1 8B NIM (LLM)

**Step 1: Access RHOAI Model Serving**
1. Log in to your RHOAI dashboard
2. Navigate to **Data Science Projects** → Your Project
3. Click **Models** tab → Select **NVIDIA NIM** → **Deploy Model**

**Step 2: Configure NIM Deployment**
1. **Model Name**: `llama-3-1-8b` (or your preferred name)
2. **Model Selection**:
   - Search for `meta/llama-3.1-8b-instruct`
   - Select the model from the list
3. **Resources**:
   - **GPU**: 1 (minimum)
   - **Memory**: 16Gi (recommended)
   - **CPU**: 4 cores (recommended)
4. **Environment Variables**:
   - `NIM_MAX_INPUT_LENGTH`: `10000`
   - `NIM_MAX_TOTAL_TOKENS`: `14784`
   - `NIM_MAX_NEW_TOKENS`: `4096`
5. **Authentication**: Check the box **Require token authentication**
6. **Replicas**: 1
7. Click **Deploy**

**Step 3: Get Endpoint URL and Token**
1. Wait for deployment to complete (Status: **Running**)
2. Click on the deployed model
3. Copy the **Inference Endpoint** URL
   - Example: `https://llama-3-1-8b-yourproject.apps.cluster.example.com/v1`
   - **Save this URL** - you'll need it for `OPENAI_API_BASE`
4. Copy the **Token** from the project secret
   - **Save this token** - you'll need it for `OPENAI_API_KEY`

**Verification:**
```bash
# Test the endpoint
curl -X POST "https://llama-3-1-8b-yourproject.apps.cluster.example.com/v1/chat/completions" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta/llama-3.1-8b-instruct",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 50
  }'
```

**Using a Different Model:**

You can deploy other models (e.g., Llama 3.3 70B) if you have sufficient GPU resources. After deploying your chosen model:

1. Update model name in `openshift/config/documentation_agents.yaml`:
   ```yaml
   overview_writer:
     llm: openai/meta/llama-3.3-70b-instruct  # Change model name

   documentation_reviewer:
     llm: openai/meta/llama-3.3-70b-instruct  # Change model name
   ```

2. Update model name in `openshift/config/planner_agents.yaml`:
   ```yaml
   code_explorer:
     llm: openai/meta/llama-3.3-70b-instruct  # Change model name

   documentation_planner:
     llm: openai/meta/llama-3.3-70b-instruct  # Change model name
   ```

3. Update the notebook:
   - Update the "Agents Prompting Template" cell (adjust template for your model)
   - Update endpoint test cell: `LLM(model="openai/your-model-name")`

**Note:** Keep the `openai/` provider prefix for self-hosted NIMs on RHOAI. Only change the model name (e.g., `openai/meta/llama-3.3-70b-instruct`).

---

### 1.2 Deploy NeMo Retriever E5 NIM (Embeddings)

**Step 1: Deploy Embedding Model**
1. In RHOAI **Models** tab, select **NVIDIA NIM** → **Deploy Model**
2. **Model Name**: `nemo-e5-embedding` (or your preferred name)
3. **Model Selection**:
   - Search for `nvidia/nv-embedqa-e5-v5`
   - Select the embedding model
4. **Resources**:
   - **GPU**: 1 (minimum)
   - **Memory**: 16Gi (recommended)
   - **CPU**: 4 cores (recommended)
5. **Authentication**: Check the box **Require token authentication**
6. **Replicas**: 1
7. Click **Deploy**

**Step 2: Get Endpoint URL and Token**
1. Wait for deployment (Status: **Running**)
2. Click on the deployed model
3. Copy the **Inference Endpoint** URL
   - Example: `https://nemo-e5-embedding-yourproject.apps.cluster.example.com/v1`
   - **Save this URL** - you'll need it for `NVIDIA_EMBED_BASE_URL`
4. Copy the **Token** from the project secret
   - **Save this token** - you'll need it for `NVIDIA_EMBED_TOKEN`

**Verification:**
```bash
# Test the embedding endpoint
curl -X POST "https://nemo-e5-embedding-yourproject.apps.cluster.example.com/v1/embeddings" \
  -H "Authorization: Bearer YOUR_EMBED_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "input": "test embedding",
    "model": "nvidia/nv-embedqa-e5-v5-passage"
  }'
```

---

## Part 2: Configure Environment Variables

### Set in RHOAI Workbench

**When creating a new workbench:**
1. Navigate to **Data Science Projects** → Your Project
2. Click **Workbenches** → **Create Workbench**
3. Configure workbench settings:
   - **Name**: Your preferred name
   - **Image**: `Jupyter | Minimal | CPU | Python 3.12` (recommended, other Python images should work)
   - **Container size**: Small
   - **Accelerator**: None
4. Scroll to **Environment Variables** section
5. Click **Add Variable** and add each:

   | Name | Value |
   |------|-------|
   | `OPENAI_API_BASE` | `https://llama-3-1-8b-yourproject.apps.cluster.example.com/v1` |
   | `OPENAI_API_KEY` | `your_llm_token` |
   | `NVIDIA_EMBED_BASE_URL` | `https://nemo-e5-embedding-yourproject.apps.cluster.example.com/v1` |
   | `NVIDIA_EMBED_TOKEN` | `your_embedding_token` |

6. Click **Create Workbench**

---

## Part 3: Run the Notebook

### 3.1 Clone Repository

In your RHOAI workbench terminal:
```bash
cd /opt/app-root/src
git clone https://github.com/crewAIInc/nvidia-demo.git
```

### 3.2 Open the RHOAI Notebook

1. In JupyterLab file browser, navigate to:
   ```
   nvidia-demo/openshift/Building_Code_Documentation_Agents_RHOAI.ipynb
   ```

2. **Double-click** to open the notebook

### 3.3 Run Auto-Setup Cells

The first 3 cells handle automatic setup:

**Cell 1 - Markdown Header:**
- Explains what the auto-setup cells do
- Just informational, no need to modify

**Cell 2 - Copy RHOAI Configs:**
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

**Expected Output:**
```
Copying RHOAI configuration files...
  ✓ Copied documentation_agents.yaml
  ✓ Copied planner_agents.yaml
```

**Cell 3 - Change Working Directory:**
```python
# Change to project root so CrewAI finds config/ and outputs go to correct locations
project_root = notebook_dir.parent
os.chdir(project_root)
print(f"✓ Working directory: {os.getcwd()}")
```

**Expected Output:**
```
✓ Working directory: /opt/app-root/src/nvidia-demo
```

### 3.4 Run All Cells

Run all cells in the notebook. This will take a few minutes depending on your resources.

**Expected Output Files:**
```
docs/
├── project_overview.mdx
├── project_architecture_and_design.mdx
├── core_workflows_and_data_flows.mdx
├── high-level_component_interactions.mdx
├── getting_started_guide.mdx
└── plan.json
```

---

## Troubleshooting

### Issue: "SQLite version 3.35.0 or higher required"

**Solution:**
```bash
pip install --upgrade pysqlite3-binary
# Restart kernel and re-run SQLite patch cell
```

### Issue: "Connection refused" or "401 Unauthorized"

**Causes:**
- NIM model not deployed or not running
- Incorrect endpoint URL
- Invalid token

**Solution:**
1. Check model status in RHOAI dashboard (should be "Running")
2. Verify endpoint URLs (copy from RHOAI model details)
3. Verify your enpoint end with /v1
4. Regenerate tokens if needed

### Issue: Config files not found

**Symptoms:** `FileNotFoundError: config/file_name.yaml`

**Solution:**
1. Ensure you ran the auto-setup cells (Cell 2 and 3)
2. Verify working directory is project root: `print(os.getcwd())`
3. Check openshift/config/ has the YAML files: `ls openshift/config/`

### Issue: Embedding mode errors

**Symptoms:** Poor retrieval quality or embedding errors

**Solution:**
Ensure you're using the `switch_embedding_mode()` function after indexing:
```python
# After WebsiteSearchTool creates the index
switch_embedding_mode(tool, "query")
```

---

## Additional Resources

- **Technical Details**: See [CHANGES.md](CHANGES.md) for in-depth adaptation explanations
- **Original Demo**: [NVIDIA Demo Repository](https://github.com/crewAIInc/nvidia-demo)
- **NVIDIA NIM Docs**: [NVIDIA NIM Documentation](https://docs.nvidia.com/nim/)
- **RHOAI Docs**: [Red Hat OpenShift AI Documentation](https://access.redhat.com/documentation/en-us/red_hat_openshift_ai)

---

**Questions or Issues?** Refer to [CHANGES.md](CHANGES.md) for technical details or open an issue on this fork's repository for RHOAI deployment support.
