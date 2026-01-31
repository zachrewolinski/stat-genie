#!/bin/bash
#
# Helper functions for Azure token refresh.
# Source this file to use the functions.
#
# Usage:
#   source scripts/token-refresh-helper.sh
#   refresh_azure_token_if_needed

# Track last refresh time to avoid refreshing too frequently
_LAST_TOKEN_REFRESH=${_LAST_TOKEN_REFRESH:-0}
_TOKEN_REFRESH_INTERVAL=${_TOKEN_REFRESH_INTERVAL:-1800}  # 30 minutes

# Check if we're using Azure by looking at the codex config
is_using_azure() {
    local config_file="$HOME/.codex/config.toml"
    if [[ -f "$config_file" ]]; then
        if grep -q 'model_provider.*=.*"azure"' "$config_file" 2>/dev/null; then
            return 0
        fi
    fi
    return 1
}

# Check if azure-identity is installed
has_azure_identity() {
    python3 -c "import azure.identity" 2>/dev/null
}

# Refresh the Azure token
do_refresh_azure_token() {
    if ! has_azure_identity; then
        echo "[token-refresh] Warning: azure-identity not installed. Run: pip install azure-identity"
        return 1
    fi

    echo "[token-refresh] Refreshing Azure Entra ID token..."
    local token
    token=$(python3 << 'PYTHON'
from azure.identity import DefaultAzureCredential
cred = DefaultAzureCredential()
print(cred.get_token("https://cognitiveservices.azure.com/.default").token)
PYTHON
    )

    if [[ -z "$token" ]]; then
        echo "[token-refresh] Error: Failed to get token"
        return 1
    fi

    export AZURE_OPENAI_API_KEY="$token"
    _LAST_TOKEN_REFRESH=$(date +%s)
    echo "[token-refresh] Token refreshed successfully (length: ${#token})"
}

# Refresh token if using Azure and token is stale or missing
refresh_azure_token_if_needed() {
    # Skip if not using Azure
    if ! is_using_azure; then
        return 0
    fi

    local now
    now=$(date +%s)
    local elapsed=$((now - _LAST_TOKEN_REFRESH))

    # Refresh if: no token, or token is older than interval
    if [[ -z "${AZURE_OPENAI_API_KEY:-}" ]] || [[ $elapsed -ge $_TOKEN_REFRESH_INTERVAL ]]; then
        do_refresh_azure_token
    fi
}

# Force refresh regardless of timing
force_refresh_azure_token() {
    if is_using_azure; then
        do_refresh_azure_token
    else
        echo "[token-refresh] Not using Azure, skipping refresh"
    fi
}
