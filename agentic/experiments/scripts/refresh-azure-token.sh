#!/bin/bash
#
# Quick script to refresh Azure Entra ID token for Codex.
# Source this before running Codex experiments:
#
#   source scripts/refresh-azure-token.sh
#
# Prerequisites:
#   - Azure CLI logged in (az login) OR other Azure credential configured
#   - pip install azure-identity

set -euo pipefail

if ! python3 -c "import azure.identity" 2>/dev/null; then
    echo "Error: azure-identity not installed. Run: pip install azure-identity"
    exit 1
fi

AZURE_OPENAI_API_KEY=$(python3 << 'PYTHON'
from azure.identity import DefaultAzureCredential
cred = DefaultAzureCredential()
print(cred.get_token("https://cognitiveservices.azure.com/.default").token)
PYTHON
)

export AZURE_OPENAI_API_KEY
echo "Azure token refreshed (length: ${#AZURE_OPENAI_API_KEY})"
