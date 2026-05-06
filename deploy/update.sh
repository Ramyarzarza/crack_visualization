#!/usr/bin/env bash
# deploy/update.sh
# Run on EC2 to pull latest code, rebuild frontend, and restart the backend.
set -e

REPO_DIR="/opt/crack-visualization"
SERVICE_NAME="crack-backend"

echo "==> Pulling latest code..."
cd "$REPO_DIR"
git pull
git lfs pull

echo "==> Updating Python dependencies..."
source backend/.venv/bin/activate
pip install -q -r backend/requirements.txt
deactivate

echo "==> Rebuilding React frontend..."
cd "$REPO_DIR/frontend"
npm ci --silent
npm run build

echo "==> Restarting backend service..."
sudo systemctl restart ${SERVICE_NAME}

echo "Done! Service status:"
sudo systemctl status ${SERVICE_NAME} --no-pager
