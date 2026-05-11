#!/usr/bin/env bash
# run_install_matrix.sh — build wheel then run all Docker install scenarios
#
# Usage (from repo root):
#   bash tests/e2e/run_install_matrix.sh
#
# Requirements: docker, python (with build package)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

GREEN='\033[0;32m'
RED='\033[0;31m'
BOLD='\033[1m'
RESET='\033[0m'

pass() { echo -e "${GREEN}✅ PASS${RESET}  $1"; }
fail() { echo -e "${RED}❌ FAIL${RESET}  $1"; FAILED+=("$1"); }

FAILED=()

# -------------------------------------------------------------------------
# Step 1: Build the wheel
# -------------------------------------------------------------------------
echo -e "\n${BOLD}Building wheel...${RESET}"
python -m build --wheel --outdir dist/ 2>&1 | tail -3
WHEEL=$(ls dist/fish_bridge_mcp-*.whl 2>/dev/null | head -1)
if [[ -z "$WHEEL" ]]; then
  echo -e "${RED}No wheel found in dist/ — build failed.${RESET}"
  exit 1
fi
echo "Using wheel: $WHEEL"

# -------------------------------------------------------------------------
# Step 2: Run each Dockerfile scenario
# -------------------------------------------------------------------------
run_docker() {
  local name="$1"
  local dockerfile="$2"
  local tag="fb-test-$(echo "$name" | tr ' ' '-' | tr '[:upper:]' '[:lower:]')"

  echo -e "\n${BOLD}[$name]${RESET} Building image..."
  if docker build -f "$dockerfile" -t "$tag" . > /tmp/docker-"$tag"-build.log 2>&1; then
    echo -e "${BOLD}[$name]${RESET} Running container..."
    if docker run --rm "$tag" > /tmp/docker-"$tag"-run.log 2>&1; then
      pass "$name"
    else
      fail "$name (container run failed)"
      echo "  → see /tmp/docker-${tag}-run.log"
    fi
  else
    fail "$name (image build failed)"
    echo "  → see /tmp/docker-${tag}-build.log"
  fi

  # Clean up image
  docker rmi "$tag" --force > /dev/null 2>&1 || true
}

run_docker "uv tool install"  "tests/e2e/Dockerfile.uv-install"
run_docker "pip install"      "tests/e2e/Dockerfile.pip-install"

# -------------------------------------------------------------------------
# Step 3: Summary
# -------------------------------------------------------------------------
echo ""
if [[ ${#FAILED[@]} -eq 0 ]]; then
  echo -e "${GREEN}${BOLD}All install matrix tests passed.${RESET}"
  exit 0
else
  echo -e "${RED}${BOLD}${#FAILED[@]} scenario(s) failed:${RESET}"
  for f in "${FAILED[@]}"; do
    echo "  - $f"
  done
  exit 1
fi
