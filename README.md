# stat-genie

Evaluating and supplementing the stability of AI-performed data science.

## Overview

stat-genie is a framework for evaluating the stability and reliability of LLM-generated statistical analyses. It uses the [blade-bench](https://github.com/behavioral-data/blade) benchmark and extends it with perturbation experiments to test how consistent AI models are when analyzing datasets.

---

## Installation

### Prerequisites

- Python 3.10+
- [Poetry](https://python-poetry.org/) package manager
- Node.js (for Codex CLI)
- API keys for LLM providers (OpenAI, Azure OpenAI, etc.)

### Python & Poetry Setup

```bash
# Clone the repository
git clone <repo-url>
cd stat-genie

# Install Poetry (if not installed)
curl -sSL https://install.python-poetry.org | python3 -

# Install Python dependencies
poetry install
```

### Codex CLI Setup

Codex CLI is required for agentic experiments:

```bash
# Option 1: Install in project (already in package.json)
npm install

# Option 2: Install globally
npm install -g @openai/codex

# Option 3: Install via Homebrew (macOS)
brew install codex
```

---

## Configuration

### OpenAI API

Set your OpenAI API key:

```bash
export OPENAI_API_KEY="sk-..."

# Or create a .env file in the project root:
# OPENAI_API_KEY=sk-...
```

### Azure OpenAI with Entra ID Authentication

For Codex CLI + Azure OpenAI using Entra ID (Azure AD), the recommended path is the setup script under `agentic/experiments/scripts/`.

**Prerequisites:**

```bash
# 1. Install Azure CLI and login
az login

# 2. Install azure-identity Python package
pip install azure-identity

# 3. Ensure you have "Cognitive Services OpenAI User" role on your Azure OpenAI resource
```

**Recommended setup (Codex CLI):**

1. Export your Azure settings (deployment is the Azure *deployment* name, not model name):

```bash
export AZURE_RESOURCE_NAME="myopenai"         # e.g., "myopenai"
export AZURE_DEPLOYMENT_NAME="gpt-5.2-codex"  # e.g., "gpt-5.2-codex"
# Optional overrides:
# export AZURE_API_VERSION="2025-04-01-preview"
# export AZURE_WIRE_API="responses"
```

2. Source the setup script (creates `~/.codex/config.toml` and exports a token into your shell):

```bash
cd agentic/experiments
source scripts/setup-azure-codex.sh
```

3. Sanity check the Codex CLI profile:

```bash
npx codex --profile azure "Say hello from Azure"
```

4. For subsequent runs, refresh the token (tokens expire after ~1 hour):

```bash
source scripts/refresh-azure-token.sh
```

**Manual configuration (if you don’t want the script):**

Create `~/.codex/config.toml`:

```toml
model_provider = "azure"
model = "your-deployment-name"  # Must be Azure DEPLOYMENT name, not model name

[model_providers.azure]
name = "Azure OpenAI"
base_url = "https://YOUR_RESOURCE.openai.azure.com/openai"  # Must include /openai
query_params = { api-version = "2025-04-01-preview" }
wire_api = "responses"
env_key = "AZURE_OPENAI_API_KEY"

[profiles.azure]
model_provider = "azure"
model = "your-deployment-name"
```

Then get a token and export it (Azure CLI login required):

```bash
export AZURE_OPENAI_API_KEY="$(python3 -c '
from azure.identity import AzureCliCredential
cred = AzureCliCredential()
print(cred.get_token("https://cognitiveservices.azure.com/.default").token)
')"
```

### Other LLM Providers

Set API keys as environment variables or in a `.env` file:

```bash
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GROQ_API_KEY=gsk-...
TOGETHER_API_KEY=...
GEMINI_API_KEY=...
```

### LLM Configuration File (`config/llm_config.yml`)

Configure LLM providers and models:

```yaml
provider: openai
model: gpt-5-mini
providers:
  openai:
    name: OpenAI
    models:
      - name: gpt-5-mini
        model:
          model: gpt-5-mini
          api_key_env_name: OPENAI_API_KEY
```

Supported providers: `openai`, `azureopenai`, `anthropic`, `groq`, `mistral`, `together`, `gemini`, `huggingface`

---

## Project Structure

```
stat-genie/
├── blade/                    # Core blade-bench scripts
│   ├── run_gen_analyses.py   # Generate LLM analyses
│   └── run_get_eval.py       # Evaluate analyses
├── blade-demos/              # Demo scripts and outputs
├── config/                   # Configuration files
│   ├── llm_config.yml        # LLM provider config
│   └── llm_eval_config.yml   # Evaluation LLM config
├── experiments/
│   ├── scripts/              # Perturbation experiment scripts
│   │   ├── run_analysis.py   # Single analysis runner
│   │   ├── run_analysis.sh   # Shell wrapper
│   │   ├── run_analysis_master.sh    # Multi-dataset runner
│   │   ├── run_pairwise_eval.py      # Pairwise evaluation
│   │   └── run_pairwise_eval.sh      # Shell wrapper
│   └── outputs/              # Experiment outputs
├── agentic/experiments/      # Agentic (Codex) experiments
│   ├── scripts/              # Agentic experiment scripts
│   └── toy/                  # Toy example
├── src/stat_genie/           # Source code
│   └── blade_pipeline/
│       ├── additions/        # Custom additions
│       │   ├── perturbations/  # Perturbation implementations
│       │   └── eval/         # Evaluation utilities
│       ├── baselines/        # Baseline implementations
│       ├── datasets/         # Dataset files
│       └── llms/             # LLM utilities
├── pyproject.toml            # Poetry configuration
└── README.md                 # This file
```

---

## Available Datasets

| Dataset | Description |
|---------|-------------|
| `affairs` | Extramarital affairs study |
| `amtl` | AMTL dataset |
| `boxes` | Boxes experiment |
| `caschools` | California schools data |
| `compas` | COMPAS recidivism data |
| `crofoot` | Crofoot study |
| `fish` | Fish dataset |
| `hurricane` | Hurricane analysis |
| `mortgage` | Mortgage data |
| `panda_nuts` | Panda nuts experiment |
| `reading` | Reading study |
| `soccer` | Soccer data |
| `teachingratings` | Teaching ratings data |
| `toy` | Toy dataset for testing |

---

## Quick Start

Run a quick demo that generates analyses and evaluates them:

```bash
# From the project root
bash blade-demos/run_and_eval_agent.sh
```

This will:
1. Generate LLM analyses for the `hurricane` dataset
2. Evaluate the generated analyses

Output files are saved to `blade-demos/analysis_output/` and `blade-demos/eval_output/`.

---

## Core Workflows

### 1. Generate LLM Analyses

Use `blade/run_gen_analyses.py` to generate statistical analyses using an LLM:

```bash
poetry run python blade/run_gen_analyses.py \
    --run_dataset hurricane \
    --llm_config_path config/llm_config.yml \
    --llm_eval_config_path config/llm_eval_config.yml \
    --llm_provider openai \
    --llm_model gpt-5-mini \
    -n 10 \
    --output_dir outputs/my_analysis
```

**Options:**

| Flag | Description | Default |
|------|-------------|---------|
| `--run_dataset` | Dataset to analyze | Required |
| `-n, --num_runs` | Number of analysis runs | 10 |
| `--use_agent` | Use ReAct agent (vs base LM) | False |
| `--no_use_data_desc` | Disable data description in prompts | False |
| `--llm_provider` | LLM provider (openai, anthropic, etc.) | From config |
| `--llm_model` | Model name | From config |
| `--output_dir` | Output directory | Auto-generated |

**Output files:**
- `multirun_analyses.json` - Generated analyses (used for evaluation)
- `llm_analysis_*.py` - Generated Python code for each run
- `llm.log` - LLM prompt/response logs
- `run.log` - Execution logs
- `config.json` - Run configuration

### 2. Evaluate Generated Analyses

Use `blade/run_get_eval.py` to evaluate the generated analyses:

```bash
poetry run python blade/run_get_eval.py \
    --multirun_load_path outputs/my_analysis/multirun_analyses.json \
    --llm_eval_config_path config/llm_eval_config.yml \
    --output_dir outputs/my_eval
```

**Options:**

| Flag | Description |
|------|-------------|
| `--multirun_load_path` | Path to `multirun_analyses.json` |
| `--submission_load_path` | Alternative: path to submission file |
| `--output_dir` | Output directory |
| `--ks` | K values for diversity metrics (e.g., `'[3,5,10]'`) |

**Output files:**
- `eval_results.json` - Detailed evaluation results
- `eval_metrics.json` - Aggregated metrics
- `llm_history.json` - LLM evaluation logs

---

## Perturbation Experiments

The `experiments/scripts/` directory contains scripts for running perturbation experiments to evaluate LLM stability.

### Perturbation Types

| Type | Description |
|------|-------------|
| `noperturb` | No perturbation (baseline) |
| `anonymize` | Anonymize feature names |
| `shuffle_names` | Shuffle feature names |
| `add_features` | Add random features |
| `replace_with_rvs` | Replace data with random values |
| `positive_leading_statement` | Add positive framing to task |
| `negative_leading_statement` | Add negative framing to task |
| `replace_and_positive_statement` | Combined replacement + positive framing |

### Run Single Analysis with Perturbation

```bash
poetry run python experiments/scripts/run_analysis.py \
    --dataset caschools \
    --analysis-num 1 \
    --perturbation-type noperturb \
    --llm-provider openai \
    --llm-model gpt-5-mini \
    --num-runs 5
```

**Options:**

| Flag | Description | Default |
|------|-------------|---------|
| `--dataset` | Dataset name | Required |
| `--analysis-num` | Analysis number (1-8) | Required |
| `--perturbation-type` | Perturbation to apply | Required |
| `--llm-provider` | LLM provider | `openai` |
| `--llm-model` | Model name | `gpt-5-mini` |
| `--num-runs` | Number of runs | 3 |
| `--use-cache` | Enable LLM caching | False |
| `--use-agent` | Use agent mode | False |

### Run All Perturbations for a Dataset

```bash
# Run all 8 perturbation types for a dataset
bash experiments/scripts/run_analysis.sh caschools
```

### Run All Datasets (SLURM)

```bash
# Submit SLURM jobs for all datasets
bash experiments/scripts/run_analysis_master.sh
```

### Run Pairwise Evaluation

After running analyses, evaluate pairwise similarity across perturbations:

```bash
poetry run python experiments/scripts/run_pairwise_eval.py \
    --dataset caschools \
    --num-multiruns 5 \
    --llm-provider openai \
    --llm-model gpt-5-mini
```

Or use the shell script:

```bash
bash experiments/scripts/run_pairwise_eval.sh caschools
```

---

## Agentic Experiments

The `agentic/experiments/` directory contains scripts for running agentic (Codex) experiments.

### Run Codex Analysis Pipeline

**Local execution (no SLURM):**

```bash
cd agentic/experiments

# For Azure OpenAI: login first (tokens auto-refresh during long runs)
az login

bash scripts/run_codex_experiments_local.sh
```

**SLURM cluster submission:**

```bash
cd agentic/experiments
bash scripts/run_codex_experiments.sh
```

This runs:
1. Analysis generation with Codex agent
2. Extraction and aggregation
3. Pairwise evaluation

The local scripts automatically refresh Azure tokens every 30 minutes to prevent expiration during long runs.

### Run Single Agentic Analysis

```bash
# From agentic/experiments/
bash scripts/analysis.sh <dataset> <perturbation> <run_number> <agent_name>

# Example:
bash scripts/analysis.sh caschools noperturb 1 codex
```

### Extract and Aggregate Results

```bash
# From agentic/experiments/
bash scripts/run_extract_and_aggregate_all.sh
```

### Running Codex with Azure

```bash
cd agentic/experiments
source scripts/refresh-azure-token.sh
bash scripts/analysis.sh caschools noperturb 1 codex
```

---

## Running on HPC Clusters (SLURM)

The shell scripts support both local execution and SLURM job submission.

**Local execution (no SLURM):**

```bash
# Run directly with bash
bash experiments/scripts/run_analysis.sh caschools
bash experiments/scripts/run_analysis_master.sh
bash experiments/scripts/run_eval_master.sh
```

**SLURM cluster submission:**

```bash
# Submit as SLURM jobs (requires SLURM environment)
sbatch experiments/scripts/run_analysis.sh caschools
sbatch experiments/scripts/run_analysis_master.sh
sbatch experiments/scripts/run_eval_master.sh
```

Note: `sbatch` is only available on HPC clusters with SLURM installed. Use `bash` for local machines.

---

## Examples

See the `examples/` directory for Jupyter notebooks demonstrating various use cases:

- `examples/affairs/` - Affairs dataset analysis
- `examples/caschools/` - California schools analysis
- `examples/fish/` - Fish dataset analysis
- `examples/using_custom_prompts/` - Custom prompt examples
- `examples/using_gpt5/` - GPT-5 usage examples

---

## Troubleshooting

### Common Issues

1. **Poetry not found**: Install Poetry with `curl -sSL https://install.python-poetry.org | python3 -`

2. **API key not set**: Ensure `OPENAI_API_KEY` is exported or in `.env`

3. **Module not found**: Run `poetry install` to install dependencies

4. **SLURM errors**: Ensure you're submitting from the project root directory

5. **Azure token expired**: Re-run `source scripts/refresh-azure-token.sh`

### Logs

Check these log files for debugging:
- `llm.log` - LLM API calls and responses
- `run.log` - General execution logs
- `out/*.log` - SLURM job outputs (when using SLURM)
