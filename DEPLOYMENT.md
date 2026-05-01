# Deployment Guide — Crack Visualization App

## Architecture

```
User Browser
     │
     ▼
https://crack.ramyarzarza.com  (Cloudflare DNS + SSL proxy)
     │
     ▼
EC2 Instance (54.88.167.121)  —  Ubuntu 24.04, t3.small
     │
     ├── Nginx :80
     │     ├── /api/*  →  uvicorn :8000  (FastAPI backend)
     │     └── /*      →  /opt/crack-visualization/frontend/dist/  (React)
     │
     └── uvicorn :8000
           └── boto3  →  AWS SageMaker endpoint (crack-detection-endpoint)
                              └── All 4 .pt models (us-east-1)
```

**SSL**: Cloudflare Flexible mode — browser↔Cloudflare is HTTPS, Cloudflare↔EC2 is HTTP.  
**AWS Auth**: EC2 IAM Instance Role (`EC2-CrackDetection` with `AmazonSageMakerFullAccess`) — no access keys stored anywhere.

---

## Files

| File | Purpose |
|---|---|
| `backend/main.py` | FastAPI app — serves `/models`, `/samples`, `/predict` |
| `backend/requirements.txt` | Python dependencies |
| `frontend/src/App.jsx` | React UI |
| `sagemaker/inference.py` | SageMaker inference handler (model_fn, predict_fn, etc.) |
| `sagemaker/deploy.py` | Script to package and deploy models to SageMaker |
| `deploy/setup_ec2.sh` | One-time EC2 setup script |
| `deploy/update.sh` | Re-deploy after code changes |

---

## Environment Variables (EC2)

Set in `/etc/systemd/system/crack-backend.service`:

| Variable | Value | Purpose |
|---|---|---|
| `SAGEMAKER_ENDPOINT` | `crack-detection-endpoint` | Routes inference to SageMaker instead of local PyTorch |
| `AWS_DEFAULT_REGION` | `us-east-1` | AWS region for boto3 |
| `ALLOWED_ORIGINS` | *(not set — defaults to localhost)* | CORS origins; not needed when Nginx serves both frontend and backend on same origin |

**To edit env vars:**
```bash
sudo nano /etc/systemd/system/crack-backend.service
sudo systemctl daemon-reload
sudo systemctl restart crack-backend
```

---

## SSH Into EC2

```bash
ssh -i ~/Downloads/crack-detection-key.pem ubuntu@54.88.167.121
```

---

## Daily / Common Commands

### Check backend status
```bash
sudo systemctl status crack-backend
```

### View live backend logs
```bash
sudo journalctl -u crack-backend -f
```

### Restart backend
```bash
sudo systemctl restart crack-backend
```

### Test API locally on EC2
```bash
curl http://localhost/api/models
curl http://localhost/api/health
```

### Check Nginx status
```bash
sudo systemctl status nginx
sudo nginx -t                    # test config syntax
sudo systemctl restart nginx
```

---

## Deploy Code Updates

After pushing changes to GitHub, SSH into EC2 and run:

```bash
cd /opt/crack-visualization && bash deploy/update.sh
```

This will:
1. `git pull` latest code
2. `pip install -r requirements.txt` (if deps changed)
3. `npm ci && npm run build` (rebuilds React)
4. `systemctl restart crack-backend`

---

## Local Development

```bash
# Terminal 1 — backend
cd backend
source .venv/bin/activate
uvicorn main:app --reload

# Terminal 2 — frontend (Vite proxy routes /api → localhost:8000)
cd frontend
npm run dev
```

App available at `http://localhost:5173`.

---

## Re-deploy SageMaker Models

Only needed if you change `sagemaker/inference.py` or add new models:

```bash
cd "/Users/ramyar/Git/Crack Visualization"
source backend/.venv/bin/activate
python sagemaker/deploy.py \
  --role "arn:aws:iam::588978531879:role/SageMakerCrackDetectionRole" \
  --bucket "crack-detection-ramyar" \
  --region "us-east-1"
```

Check status: AWS Console → SageMaker → Inference → Endpoints → `crack-detection-endpoint`

---

## AWS Resources

| Resource | Name / ID |
|---|---|
| EC2 Instance | `Crack_visualization_backend` — `i-0231e85ed9846993f` |
| EC2 Public IP | `54.88.167.121` |
| EC2 IAM Role | `EC2-CrackDetection` |
| SageMaker Endpoint | `crack-detection-endpoint` (us-east-1) |
| S3 Bucket | `crack-detection-ramyar` |
| Domain | `crack.ramyarzarza.com` (Cloudflare) |
| AWS Account | `588978531879` |

---

## Troubleshooting

**App not loading:**
```bash
sudo systemctl status crack-backend
sudo systemctl status nginx
sudo journalctl -u crack-backend -n 50 --no-pager
```

**SageMaker errors (NoCredentialsError):**
- Check IAM role is attached: EC2 Console → Instance → Security → IAM Role
- Should show `EC2-CrackDetection`

**SageMaker errors (endpoint not found):**
```bash
# Check endpoint status from EC2:
source /opt/crack-visualization/backend/.venv/bin/activate
python3 -c "
import boto3
sm = boto3.client('sagemaker', region_name='us-east-1')
r = sm.describe_endpoint(EndpointName='crack-detection-endpoint')
print('Status:', r['EndpointStatus'])
"
```

**After EC2 reboot** — everything auto-starts (both `crack-backend` and `nginx` are enabled via systemd).
