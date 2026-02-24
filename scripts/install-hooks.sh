#!/usr/bin/env bash
#
# Install git hooks by symlinking from scripts/ into .git/hooks/.
# Run once after cloning:  bash scripts/install-hooks.sh

set -e

REPO_ROOT="$(git rev-parse --show-toplevel)"
HOOKS_DIR="${REPO_ROOT}/.git/hooks"
SCRIPTS_DIR="${REPO_ROOT}/scripts"

ln -sf "${SCRIPTS_DIR}/pre-commit" "${HOOKS_DIR}/pre-commit"
echo "Installed pre-commit hook -> scripts/pre-commit"
