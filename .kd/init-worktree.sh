#!/usr/bin/env bash
# Kingdom worktree init — runs after "kd peasant start" creates a worktree.
# The worktree path is passed as $1.
#
# Examples:
#   cd "$1" && uv sync && pre-commit install
#   cd "$1" && npm install
#
echo "⚔️  Preparing the realm at $1"
