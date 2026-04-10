# Sanity Checks for Agentic Data Science

Evaluating and supplementing the stability of AI-performed data science.

## Overview

We introduce a framework for evaluating the stability and reliability of LLM-generated statistical analyses. It uses the [blade-bench](https://github.com/behavioral-data/blade) benchmark and extends it with perturbation experiments to test how consistent AI models are when analyzing datasets.

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
├── blade/                    # Copy of the BLADE repo
│   ├── run_gen_analyses.py   # Generate LLM analyses
│   └── run_get_eval.py       # Evaluate analyses
├── config/                   # Configuration files
│   ├── llm_config.yml        # LLM provider config (legacy)
│   └── llm_eval_config.yml   # Evaluation LLM config (legacy)
├── experiments/              # Repository of non-agentic experiments using the BLADE harness
│   ├── scripts/              # Perturbation experiment scripts
│   │   ├── run_analysis.py   # Single analysis runner
│   │   ├── run_analysis.sh   # Shell wrapper
│   │   ├── run_analysis_master.sh    # Multi-dataset runner
│   │   ├── run_pairwise_eval.py      # Pairwise evaluation
│   │   └── run_pairwise_eval.sh      # Shell wrapper
│   └── outputs/              # Experiment outputs
├── agentic/                  # Agentic (Codex) experiments
│   ├── confidence_experiments/  # Experiments eliciting confidence scores from Codex
│   │   ├── scripts/          # Runner and aggregation scripts
│   │   ├── outputs/          # Per-run Codex outputs
│   │   ├── aggregated_results/ # Aggregated CSVs
│   │   └── insights/         # Analysis notebooks and figures
│   ├── scalar_experiments/   # Experiments eliciting scalar yes/no conclusions
│   │   ├── scripts/          # Runner and aggregation scripts
│   │   ├── outputs/          # Per-run Codex outputs
│   │   ├── aggregated_results/ # Aggregated CSVs
│   │   └── insights/         # Analysis notebooks and figures
│   └── human_experiments/    # Human baseline experiment results
├── src/stat_genie/           # Source code
│   └── blade_pipeline/       # Our additions/modifications to BLADE code
│       ├── additions/        # Custom additions to the blade pipeline
│       │   ├── analysis/     # Conclusion writing and model output extraction
│       │   ├── perturbations/  # Perturbation implementations
│       │   ├── eval/         # Evaluation utilities (extraction, judging)
│       │   └── prompt/       # Prompt construction utilities
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

## Core Workflows

Both experiment types (scalar and confidence) follow the same three-phase pattern: **run → fix → aggregate**. All commands should be run from the respective experiment directory.

### Scalar Experiments (`agentic/scalar_experiments/`)

Codex performs a full statistical analysis and writes a scalar yes/no conclusion to `conclusion.txt` in each run's output subdirectory.

**Phase 1 — Run analyses:**

```bash
cd agentic/scalar_experiments

# SLURM (recommended for full experiment):
sbatch scripts/analysis-runner.sh

# Local:
bash scripts/analysis-runner-local.sh

# Single run:
bash scripts/analysis.sh <dataset> <distribution> <perturbation> <run_number>
# e.g.:
bash scripts/analysis.sh caschools null none 1
bash scripts/analysis.sh caschools alt anonymize 3
```

For PVE (proportion of variance explained) experiments across a range of signal strengths:

```bash
sbatch scripts/pve-analysis-runner.sh   # SLURM
bash scripts/pve-runner-local.sh        # local
```

**Phase 2 — Fix broken conclusions:**

Some runs produce malformed `conclusion.txt` files, even though the agent is instructed to produce valid JSON output. Fix them before aggregating:

```bash
bash scripts/fix-conclusions.sh          # null/alt distributions
bash scripts/fix-conclusions.sh --pve    # pve distribution
```

**Phase 3 — Aggregate:**

```bash
bash scripts/aggregate_conclusions.sh       # null/alt → aggregated_results/aggregated_results.csv
bash scripts/aggregate_pve_conclusions.sh   # pve → aggregated_results/aggregated_pve_results.csv
```

**Optional — Calibration simulation:**

```bash
sbatch scripts/run_calibration.sh   # SLURM (32 CPUs, 64G RAM)
# Results saved to insights/calibration_results_*.npz
```

---

### Confidence Experiments (`agentic/confidence_experiments/`)

Codex is shown the output of a prior scalar analysis and asked to produce a calibrated confidence score.

**Phase 1 — Run confidence elicitations:**

```bash
cd agentic/confidence_experiments

# SLURM:
sbatch scripts/confidence-runner.sh

# Local:
bash scripts/analysis-runner-local.sh

# Single run:
bash scripts/confidence.sh <dataset> <distribution> <perturbation> <run_number>
# e.g.:
bash scripts/confidence.sh caschools null anonymize 1
```

**Phase 2 — Fix broken conclusions:**

```bash
bash scripts/fix-conclusions.sh --pve
```

**Phase 3 — Aggregate:**

```bash
bash scripts/aggregate_pve_conclusions.sh   # → aggregated_results/aggregated_pve_results.csv
```

**Optional — Calibration simulation:**

```bash
sbatch scripts/run_calibration.sh
# Results saved to insights/calibration_results_*.npz
```

---

### Output structure

Each run produces a subdirectory under `outputs/`:

```
outputs/<dataset>/<distribution>/<perturbation>/run<N>/
    AGENTS.md        # Codex prompt
    conclusion.txt   # Model's yes/no answer (scalar) or confidence score
    *.py             # Generated analysis code
    agent-analysis.out  # Raw Codex session log
```

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

The `agentic/` directory contains three experiment types, each in its own subdirectory. All experiments use Codex (via `npx codex exec`) and follow a similar structure: a runner script dispatches per-run jobs, each job sets up a subdirectory under `outputs/` and runs Codex against an `AGENTS.md` prompt, and aggregation scripts collect results into CSVs under `aggregated_results/`.

### Scalar Experiments (`agentic/scalar_experiments/`)

Codex performs a full statistical analysis and produces a scalar yes/no conclusion for each research question.

**Run a single analysis:**

```bash
cd agentic/scalar_experiments

# For Azure: refresh token first
source scripts/token-refresh-helper.sh

bash scripts/analysis.sh <dataset> <distribution> <perturbation> <run_number>

# Example:
bash scripts/analysis.sh caschools null none 1
```

`<distribution>` is `null`, `alt`, or `pve`. `<perturbation>` is `none` or one of the perturbation types (e.g. `anonymize`, `shuffle_names`, `add_features`, `positive_leading_statement`, `negative_leading_statement`).

**Run all analyses (SLURM):**

```bash
cd agentic/scalar_experiments
sbatch scripts/analysis-runner.sh

# Or locally:
bash scripts/analysis-runner-local.sh
```

**Aggregate results:**

```bash
cd agentic/scalar_experiments
bash scripts/aggregate_conclusions.sh
```

### Confidence Experiments (`agentic/confidence_experiments/`)

Codex is given the output of a prior scalar analysis and asked to produce a calibrated confidence score for its conclusion.

**Run a single confidence elicitation:**

```bash
cd agentic/confidence_experiments
bash scripts/confidence.sh <dataset> <distribution> <perturbation> <run_number>

# Example:
bash scripts/confidence.sh caschools null anonymize 1
```

**Run all (SLURM):**

```bash
cd agentic/confidence_experiments
sbatch scripts/confidence-runner.sh
```

**Aggregate results:**

```bash
cd agentic/confidence_experiments
bash scripts/aggregate_pve_conclusions.sh
```

### Human Experiments (`agentic/human_experiments/`)

Contains human baseline results (CSV) and analysis notebook (`insights.ipynb`) for comparison against the agentic experiments.

### Running Codex with Azure

For all experiment types, authenticate before running:

```bash
# Login (once per session)
az login

# Refresh token (tokens expire after ~1 hour)
source scripts/token-refresh-helper.sh
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
