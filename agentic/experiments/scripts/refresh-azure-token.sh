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

# NOTE: This script is often sourced into an existing shell/tmux session.
# Avoid strict fail-fast behavior so that a transient auth or import error
# doesn't close your shell. Errors will be reported but won't exit.

if ! python3 -c "import azure.identity" 2>/dev/null; then
	echo "Warning: azure-identity not installed. Run: pip install azure-identity"
	return 0 2>/dev/null || exit 0
fi

AZURE_OPENAI_API_KEY=$(python3 << 'PYTHON'
from azure.identity import DefaultAzureCredential

scope = "https://cognitiveservices.azure.com/.default"

try:
    cred = DefaultAzureCredential(
        exclude_managed_identity_credential=True,
    )
    token = cred.get_token(scope).token
    print(token)
except Exception as e:
    import sys
    print(f"Error getting token: {e}", file=sys.stderr)
PYTHON
)

if [ -z "${AZURE_OPENAI_API_KEY:-}" ]; then
	echo "Warning: Failed to obtain Azure token; AZURE_OPENAI_API_KEY not set."
	return 0 2>/dev/null || exit 0
fi

export AZURE_OPENAI_API_KEY
echo "Azure token refreshed (length: ${#AZURE_OPENAI_API_KEY})"
