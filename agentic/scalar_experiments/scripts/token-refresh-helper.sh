#!/bin/bash
#
# Token refresh helper for Azure OpenAI.
# Source this file to get refresh_azure_token_if_needed and refresh_azure_token_until_ok.
#
# Usage:
#   source scripts/token-refresh-helper.sh
#   refresh_azure_token_if_needed   # single attempt
#   refresh_azure_token_until_ok    # retry with backoff (recommended before runs)
#

# Default: max attempts and delays (seconds) between attempts
TOKEN_REFRESH_MAX_ATTEMPTS="${TOKEN_REFRESH_MAX_ATTEMPTS:-5}"
TOKEN_REFRESH_DELAYS="${TOKEN_REFRESH_DELAYS:-2 5 10 20 30}"  # space-separated

# Single attempt: get Azure token and export AZURE_OPENAI_API_KEY.
# Returns 0 on success, 1 on failure. Use "refresh_azure_token_if_needed || true" under set -e.
refresh_azure_token_if_needed() {
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
    sys.exit(1)
PYTHON
) || true
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
        echo "Warning: Failed to get token; AZURE_OPENAI_API_KEY not set." >&2
        return 1
    fi

    export AZURE_OPENAI_API_KEY="$token"
    echo "Token refreshed (length: ${#AZURE_OPENAI_API_KEY})" >&2
    return 0
}

# Retry token refresh with backoff until success or max attempts.
# Exports AZURE_OPENAI_API_KEY on success. Returns 0 on success, 1 on final failure.
refresh_azure_token_until_ok() {
    local attempt=1
    local max_attempts="$TOKEN_REFRESH_MAX_ATTEMPTS"
    local delays=($TOKEN_REFRESH_DELAYS)

    while true; do
        echo "[token-refresh] Attempt $attempt/$max_attempts..." >&2
        if refresh_azure_token_if_needed; then
            return 0
        fi
        if [[ $attempt -ge $max_attempts ]]; then
            echo "[token-refresh] All $max_attempts attempts failed." >&2
            return 1
        fi
        local delay="${delays[$((attempt-1))]:-30}"
        echo "[token-refresh] Retrying in ${delay}s..." >&2
        sleep "$delay"
        ((attempt++)) || true
    done
}
