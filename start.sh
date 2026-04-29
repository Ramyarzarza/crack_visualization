#!/usr/bin/env bash
# start.sh — starts both the FastAPI backend and React frontend
set -e

ROOT="$(cd "$(dirname "$0")" && pwd)"
BACKEND="$ROOT/backend"
FRONTEND="$ROOT/frontend"

# Load nvm if available
export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"

# ── Backend ──────────────────────────────────────────────────────────────────
echo "==> Setting up Python backend..."
cd "$BACKEND"

if [ ! -d ".venv" ]; then
  echo "    Creating virtual environment..."
  python3 -m venv .venv
fi

source .venv/bin/activate
pip install -q --upgrade pip
pip install -q -r requirements.txt
echo "    Backend dependencies ready."

# Print inference mode
if [ -n "$SAGEMAKER_ENDPOINT" ]; then
  echo "    Inference mode: SageMaker endpoint '$SAGEMAKER_ENDPOINT' (region: ${AWS_DEFAULT_REGION:-us-east-1})"
else
  echo "    Inference mode: local PyTorch"
  echo "    (Set SAGEMAKER_ENDPOINT=<name> before running to use SageMaker)"
fi

# Start backend in background
uvicorn main:app --host 0.0.0.0 --port 8000 --reload &
BACKEND_PID=$!
echo "    Backend running on http://localhost:8000 (PID $BACKEND_PID)"

# ── Frontend ─────────────────────────────────────────────────────────────────
echo ""
echo "==> Setting up React frontend..."
cd "$FRONTEND"

if [ ! -d "node_modules" ]; then
  echo "    Installing npm packages..."
  npm install
fi

echo "    Starting Vite dev server..."
npm run dev &
FRONTEND_PID=$!
echo "    Frontend running on http://localhost:5173 (PID $FRONTEND_PID)"

# ── Cleanup on exit ───────────────────────────────────────────────────────────
trap "echo ''; echo 'Stopping servers...'; kill $BACKEND_PID $FRONTEND_PID 2>/dev/null" EXIT INT TERM

echo ""
echo "======================================================"
echo "  App running at:  http://localhost:5173"
echo "  API docs at:     http://localhost:8000/docs"
echo "  Press Ctrl+C to stop both servers."
echo "======================================================"

wait
