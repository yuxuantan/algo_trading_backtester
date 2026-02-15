#!/usr/bin/env bash
set -euo pipefail

show_help() {
  cat <<'EOF'
Usage:
  bash scripts/setup_silicon_mt5_local.sh [options]

Options:
  --repo-dir PATH          Directory for silicon-metatrader5 clone.
                           Default: <project>/.third_party/silicon-metatrader5
  --with-python-deps       Install project extra: .[mt5-mac]
  --force-restart-colima   Restart Colima even if already running.
  -h, --help               Show this help.

Environment overrides:
  COLIMA_CPU=4
  COLIMA_MEMORY=6
  COLIMA_DISK=60
  SILICON_MT5_PORT=8001
  SILICON_MT5_VNC_PORT=6081
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  show_help
  exit 0
fi

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_DIR_DEFAULT="${PROJECT_ROOT}/.third_party/silicon-metatrader5"
REPO_DIR="${REPO_DIR_DEFAULT}"
WITH_PYTHON_DEPS=0
FORCE_RESTART_COLIMA=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo-dir)
      REPO_DIR="$2"
      shift 2
      ;;
    --with-python-deps)
      WITH_PYTHON_DEPS=1
      shift
      ;;
    --force-restart-colima)
      FORCE_RESTART_COLIMA=1
      shift
      ;;
    -h|--help)
      show_help
      exit 0
      ;;
    *)
      echo "Unknown argument: $1"
      show_help
      exit 1
      ;;
  esac
done

COLIMA_CPU="${COLIMA_CPU:-4}"
COLIMA_MEMORY="${COLIMA_MEMORY:-6}"
COLIMA_DISK="${COLIMA_DISK:-60}"
SILICON_MT5_PORT="${SILICON_MT5_PORT:-8001}"
SILICON_MT5_VNC_PORT="${SILICON_MT5_VNC_PORT:-6081}"
SILICON_REPO_URL="https://github.com/bahadirumutiscimen/silicon-metatrader5.git"

need_cmd() {
  command -v "$1" >/dev/null 2>&1
}

ensure_brew_formula() {
  local cmd="$1"
  local formula="$2"
  if ! need_cmd "$cmd"; then
    echo "Installing ${formula}..."
    brew install "$formula"
  fi
}

ensure_brew_formula_installed() {
  local formula="$1"
  if ! brew list --formula "$formula" >/dev/null 2>&1; then
    echo "Installing ${formula}..."
    brew install "$formula"
  fi
}

find_compose_file() {
  local repo_dir="$1"
  local candidates=(
    "${repo_dir}/compose.yaml"
    "${repo_dir}/compose.yml"
    "${repo_dir}/docker-compose.yaml"
    "${repo_dir}/docker-compose.yml"
    "${repo_dir}/docker/compose.yaml"
    "${repo_dir}/docker/compose.yml"
    "${repo_dir}/docker/docker-compose.yaml"
    "${repo_dir}/docker/docker-compose.yml"
  )

  for f in "${candidates[@]}"; do
    if [[ -f "${f}" ]]; then
      echo "${f}"
      return 0
    fi
  done

  local discovered
  discovered="$(find "${repo_dir}" -maxdepth 4 -type f \( -name "compose.yaml" -o -name "compose.yml" -o -name "docker-compose.yaml" -o -name "docker-compose.yml" \) | head -n 1)"
  if [[ -n "${discovered}" ]]; then
    echo "${discovered}"
    return 0
  fi
  return 1
}

start_colima_x86() {
  if colima start --arch x86_64 --cpu "${COLIMA_CPU}" --memory "${COLIMA_MEMORY}" --disk "${COLIMA_DISK}"; then
    return 0
  fi

  echo "Colima start failed. Attempting one-time repair by recreating the VM..."
  colima delete -f || true
  colima start --arch x86_64 --cpu "${COLIMA_CPU}" --memory "${COLIMA_MEMORY}" --disk "${COLIMA_DISK}"
}

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "This setup script is for macOS only."
  exit 1
fi

if ! need_cmd brew; then
  echo "Homebrew is required. Install from https://brew.sh and run this script again."
  exit 1
fi

echo "Checking required tools..."
ensure_brew_formula git git
ensure_brew_formula colima colima
ensure_brew_formula docker docker
ensure_brew_formula docker-compose docker-compose
ensure_brew_formula qemu qemu
ensure_brew_formula_installed lima-additional-guestagents
ensure_brew_formula nc netcat

COLIMA_STATUS_FILE="$(mktemp)"
COLIMA_RUNNING=0
if colima status >"${COLIMA_STATUS_FILE}" 2>&1; then
  COLIMA_RUNNING=1
fi

if [[ "${FORCE_RESTART_COLIMA}" == "1" && "${COLIMA_RUNNING}" == "1" ]]; then
  echo "Force restarting Colima..."
  colima stop || true
  COLIMA_RUNNING=0
fi

if [[ "${COLIMA_RUNNING}" == "1" ]]; then
  if grep -Eqi 'arch:\s*x86_64|architecture:\s*x86_64' "${COLIMA_STATUS_FILE}"; then
    echo "Colima already running with x86_64."
  else
    echo "Colima is running but not in x86_64 mode. Restarting..."
    colima stop || true
    start_colima_x86
  fi
else
  echo "Starting Colima (x86_64)..."
  start_colima_x86
fi
rm -f "${COLIMA_STATUS_FILE}"

if ! docker info >/dev/null 2>&1; then
  echo "Docker is not reachable even after Colima start."
  echo "Try: colima status && docker info"
  exit 1
fi

if docker compose version >/dev/null 2>&1; then
  COMPOSE_CMD=(docker compose)
elif need_cmd docker-compose; then
  COMPOSE_CMD=(docker-compose)
else
  echo "Docker Compose is unavailable."
  exit 1
fi

mkdir -p "$(dirname "${REPO_DIR}")"
if [[ -d "${REPO_DIR}/.git" ]]; then
  echo "Updating silicon-metatrader5..."
  git -C "${REPO_DIR}" pull --ff-only
else
  echo "Cloning silicon-metatrader5..."
  git clone "${SILICON_REPO_URL}" "${REPO_DIR}"
fi

COMPOSE_FILE="$(find_compose_file "${REPO_DIR}" || true)"
if [[ -z "${COMPOSE_FILE}" ]]; then
  echo "No Docker Compose file found under ${REPO_DIR}"
  exit 1
fi
COMPOSE_DIR="$(cd "$(dirname "${COMPOSE_FILE}")" && pwd)"
COMPOSE_BASENAME="$(basename "${COMPOSE_FILE}")"

echo "Building and starting silicon-metatrader5 containers..."
(
  cd "${COMPOSE_DIR}"
  "${COMPOSE_CMD[@]}" -f "${COMPOSE_BASENAME}" up -d --build
)

echo "Waiting for MT5 bridge at localhost:${SILICON_MT5_PORT}..."
READY=0
for _ in $(seq 1 90); do
  if nc -z localhost "${SILICON_MT5_PORT}" >/dev/null 2>&1; then
    READY=1
    break
  fi
  sleep 2
done

if [[ "${READY}" != "1" ]]; then
  echo "MT5 bridge port did not open in time. Check container logs:"
  echo "  cd ${COMPOSE_DIR} && ${COMPOSE_CMD[*]} -f ${COMPOSE_BASENAME} logs --tail=200"
  exit 1
fi

if [[ "${WITH_PYTHON_DEPS}" == "1" ]]; then
  if [[ -x "${PROJECT_ROOT}/.venv/bin/pip" ]]; then
    PIP_CMD="${PROJECT_ROOT}/.venv/bin/pip"
  elif need_cmd pip3; then
    PIP_CMD="pip3"
  elif need_cmd pip; then
    PIP_CMD="pip"
  else
    echo "Skipping Python deps install (pip not found)."
    PIP_CMD=""
  fi

  if [[ -n "${PIP_CMD}" ]]; then
    echo "Installing Python dependencies: .[mt5-mac]"
    (
      cd "${PROJECT_ROOT}"
      "${PIP_CMD}" install -e ".[mt5-mac]"
    )
  fi
fi

cat <<EOF

Setup complete.

1) Open VNC and log into MT5 terminal:
   http://localhost:${SILICON_MT5_VNC_PORT}/vnc.html
   Default VNC password: 123456

2) Fetch bars from MT5 bridge:
   python3 scripts/fetch_mt5_live_data.py \\
     --provider silicon \\
     --host localhost \\
     --port ${SILICON_MT5_PORT} \\
     --symbol EURUSD \\
     --timeframe M5 \\
     --bars 500

3) If needed, inspect logs:
   cd ${COMPOSE_DIR}
   ${COMPOSE_CMD[*]} -f ${COMPOSE_BASENAME} logs --tail=200

EOF
