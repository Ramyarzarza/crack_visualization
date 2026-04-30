#!/usr/bin/env bash
# deploy/setup_ec2.sh
# Run this ONCE on a fresh Ubuntu 22.04/24.04 EC2 instance.
# Both the React frontend and FastAPI backend run on the same instance.
# Nginx serves React static files on port 80 and proxies /api/ to uvicorn.
# Usage: bash setup_ec2.sh
set -e

REPO_DIR="/opt/crack-visualization"
SERVICE_NAME="crack-backend"

echo "==> [1/7] Installing system packages..."
sudo apt-get update -q
sudo apt-get install -y -q python3 python3-pip python3-venv nginx git libgl1 curl

echo "==> [2/7] Installing Node.js 20..."
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash - -q
sudo apt-get install -y -q nodejs

echo "==> [3/7] Cloning / updating repo..."
if [ ! -d "$REPO_DIR" ]; then
    sudo git clone https://github.com/YOUR_GITHUB_USER/YOUR_REPO_NAME.git "$REPO_DIR"
else
    cd "$REPO_DIR" && sudo git pull
fi
sudo chown -R ubuntu:ubuntu "$REPO_DIR"

echo "==> [4/7] Setting up Python virtualenv..."
cd "$REPO_DIR/backend"
python3 -m venv .venv
source .venv/bin/activate
pip install -q --upgrade pip
pip install -q -r requirements.txt
deactivate

echo "==> [5/7] Building React frontend..."
cd "$REPO_DIR/frontend"
npm ci --silent
# VITE_API_URL is intentionally empty: Nginx serves both on the same origin,
# so /api resolves correctly without a full URL (no CORS needed).
npm run build

echo "==> [6/7] Creating systemd service for backend..."
sudo tee /etc/systemd/system/${SERVICE_NAME}.service > /dev/null <<EOF
[Unit]
Description=Crack Detection FastAPI Backend
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=${REPO_DIR}/backend
Environment="SAGEMAKER_ENDPOINT=crack-detection-endpoint"
Environment="AWS_DEFAULT_REGION=us-east-1"
ExecStart=${REPO_DIR}/backend/.venv/bin/uvicorn main:app --host 127.0.0.1 --port 8000
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable ${SERVICE_NAME}
sudo systemctl restart ${SERVICE_NAME}
echo "    Backend service started."

echo "==> [7/7] Configuring Nginx..."
sudo tee /etc/nginx/sites-available/crack-app > /dev/null <<EOF
server {
    listen 80;
    server_name _;

    client_max_body_size 20M;

    # Proxy API requests to FastAPI
    location /api/ {
        proxy_pass http://127.0.0.1:8000/;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_read_timeout 120s;
    }

    # Serve React static files; fallback to index.html for SPA routing
    root ${REPO_DIR}/frontend/dist;
    index index.html;
    location / {
        try_files \$uri \$uri/ /index.html;
    }
}
EOF

sudo ln -sf /etc/nginx/sites-available/crack-app /etc/nginx/sites-enabled/crack-app
sudo rm -f /etc/nginx/sites-enabled/default
sudo nginx -t
sudo systemctl restart nginx
echo "    Nginx configured."

echo ""
echo "==> Setup complete!"
echo "    App:          http://YOUR_EC2_PUBLIC_IP/"
echo "    API health:   http://YOUR_EC2_PUBLIC_IP/api/models"
echo "    Backend logs: sudo journalctl -u ${SERVICE_NAME} -f"
