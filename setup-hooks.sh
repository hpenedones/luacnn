#!/bin/bash
# Setup script to install git hooks for the repository

HOOKS_DIR="$(git rev-parse --show-toplevel)/hooks"
GIT_HOOKS_DIR="$(git rev-parse --git-dir)/hooks"

echo "Installing git hooks..."

# Copy pre-commit hook
if [ -f "$HOOKS_DIR/pre-commit" ]; then
    cp "$HOOKS_DIR/pre-commit" "$GIT_HOOKS_DIR/pre-commit"
    chmod +x "$GIT_HOOKS_DIR/pre-commit"
    echo "✓ Installed pre-commit hook"
else
    echo "✗ pre-commit hook not found in $HOOKS_DIR"
    exit 1
fi

echo "Git hooks installed successfully!"
echo ""
echo "Note: Make sure StyLua is installed on your system."
echo "Install from: https://github.com/JohnnyMorganz/StyLua#installation"
