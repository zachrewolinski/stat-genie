#!/bin/bash
#
# Token refresh helper for Azure OpenAI.
# Source this file to get the refresh_azure_token_if_needed function.
#
# Usage:
#   source scripts/token-refresh-helper.sh
#   refresh_azure_token_if_needed
#

# Refresh Azure token using azure-identity
# NOTE: This function uses return 0 on errors to avoid killing shells with set -e.
# Errors are reported but won't exit the parent shell.
refresh_azure_token_if_needed() {
    echo "Refreshing Azure Entra ID token..."
    
    local token
    if python3 -c "import azure.identity" 2>/dev/null; then
        token=$(python3 << 'PYTHON'
from azure.identity import AzureCliCredential
import sys

scope = "https://cognitiveservices.azure.com/.default"

try:
    cred = AzureCliCredential()
    token = cred.get_token(scope).token
    print("USING_AZURE_CREDENTIAL:AzureCliCredential", file=sys.stderr)
    print(token)
except Exception as e:
    print(f"Error getting token: {e}", file=sys.stderr)
PYTHON
)
    else
        token=$(az account get-access-token \
            --resource https://cognitiveservices.azure.com/ \
            --query accessToken -o tsv 2>/dev/null || true)
        if [[ -n "$token" ]]; then
            echo "USING_AZURE_CREDENTIAL:AzureCLI" >&2
        else
            echo "Warning: azure-identity not installed. Run: pip install azure-identity" >&2
        fi
    fi
    
    if [[ -z "$token" ]]; then
        echo "Warning: Failed to get token; AZURE_OPENAI_API_KEY not set."
        return 0
    fi
    
    export AZURE_OPENAI_API_KEY="$token"
    echo "Token refreshed (length: ${#AZURE_OPENAI_API_KEY})"
}
