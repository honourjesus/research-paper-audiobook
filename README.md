# 📚 Research Paper Audiobook API

Convert ML/AI research papers into smart audiobooks — handling equations, tables, and citations gracefully using local AI. **No API keys required.**

![Python](https://img.shields.io/badge/Python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

##  The Problem This Solves

Research papers are full of things that sound terrible when read aloud:
- LaTeX equations like `$\alpha = \frac{1}{2}\sigma^2$`
- Raw table data with no context
- Citation brackets like `[23]` or `(Smith et al., 2020)`
- Dense academic language

This project solves that by building an intelligent pipeline that converts research papers into natural, listenable audiobooks.

---

## ⚙️ How It Works

```
PDF / ArXiv URL
      │
      ▼
┌──────────────────┐
│   PDF Parser     │  PyMuPDF detects headings via font size/weight
│                  │  pdfplumber extracts tables with boundary detection
└──────────────────┘
      │
      ▼
┌──────────────────┐
│ Equation Handler │  Converts LaTeX math → spoken English
│                  │  e.g. "$\sigma^2$" → "sigma squared"
└──────────────────┘
      │
      ▼
┌──────────────────┐
│ Table Summarizer │  Generates narrative summaries using pandas
│                  │  statistics, trends, and correlations
└──────────────────┘
      │
      ▼
┌──────────────────┐
│   TTS Engine     │  Converts cleaned text to speech
│                  │  gTTS (online) or SpeechT5 (offline)
└──────────────────┘
      │
      ▼
   📢 Audiobook .wav
```

---

##  Quick Start

### Prerequisites
- Python 3.11+
- [Ollama](https://ollama.com) for local LLM (`ollama serve` + `ollama pull llama3`)

### Run Locally

```bash
# Clone the repo
git clone https://github.com/honourjesus/research-paper-audiobook.git
cd research-paper-audiobook

# Create and activate virtual environment
python -m venv venv
venv\Scripts\Activate.ps1        # Windows
# source venv/bin/activate       # Mac/Linux

# Install dependencies
pip install -r requirements.txt

# Start the server
uvicorn src.api:app --reload
```

Visit `http://127.0.0.1:8000/docs` for the interactive API.

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Service status |
| GET | `/health` | Health check for monitoring |
| POST | `/convert` | Upload PDF or provide ArXiv URL |
| GET | `/status/{job_id}` | Check conversion progress |
| GET | `/download/{job_id}` | Download completed audiobook |

### Example Request

```bash
# Upload a PDF
curl -X POST "http://localhost:8000/convert" \
  -F "file=@paper.pdf"

# Response
{
  "job_id": "abc-123",
  "status": "processing",
  "message": "Paper conversion started"
}

# Check progress
curl "http://localhost:8000/status/abc-123"

# Download when complete
curl "http://localhost:8000/download/abc-123" -o audiobook.wav
```

---

## 🛠️ Tech Stack

| Component | Technology | Reason |
|-----------|-----------|--------|
| API Framework | FastAPI | Async processing, auto-generates docs |
| PDF Parsing | PyMuPDF + pdfplumber | Best of both: font metadata + table detection |
| LLM | Ollama (llama3/phi3) | 100% local, free, private |
| Text-to-Speech | gTTS / SpeechT5 | Flexible: online fast or offline neural |
| Table Analysis | Pandas | Statistics and narrative generation |
| Containers | Docker + Docker Compose | One-command deployment anywhere |

---

## 🐳 Docker Deployment

```bash
# Run everything with one command
docker-compose up --build
```

This starts the API, Redis cache, and Nginx load balancer automatically.

---

## ☁️ AWS Deployment (Step by Step)

### Prerequisites
- AWS account
- AWS CLI installed and configured (`aws configure`)
- Docker installed

### Option A: AWS Elastic Container Service (ECS) — Recommended

**Step 1: Push your Docker image to ECR**
```bash
# Create ECR repository
aws ecr create-repository --repository-name research-paper-audiobook --region us-east-1

# Authenticate Docker to ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS \
  --password-stdin YOUR_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com

# Build, tag and push
docker build -t research-paper-audiobook .
docker tag research-paper-audiobook:latest \
  YOUR_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/research-paper-audiobook:latest
docker push YOUR_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/research-paper-audiobook:latest
```

**Step 2: Create ECS Cluster**
```bash
aws ecs create-cluster --cluster-name paper-audiobook-cluster
```

**Step 3: Create Task Definition**
- Go to AWS Console → ECS → Task Definitions → Create new
- Select Fargate (serverless, no server management)
- Set memory: 2GB, CPU: 1 vCPU
- Add container: use your ECR image URL
- Port mapping: 8000

**Step 4: Create ECS Service**
```bash
aws ecs create-service \
  --cluster paper-audiobook-cluster \
  --service-name paper-audiobook-service \
  --task-definition paper-audiobook:1 \
  --desired-count 1 \
  --launch-type FARGATE \
  --network-configuration "awsvpcConfiguration={subnets=[subnet-xxx],securityGroups=[sg-xxx],assignPublicIp=ENABLED}"
```

**Step 5: Add Load Balancer (optional but recommended)**
- AWS Console → EC2 → Load Balancers → Create Application Load Balancer
- Point it to your ECS service on port 8000
- Your app is now accessible via the ALB DNS name

### Option B: AWS EC2 — Simpler for learning

```bash
# Launch EC2 instance (Ubuntu t2.medium recommended)
aws ec2 run-instances \
  --image-id ami-0c7217cdde317cfec \
  --instance-type t2.medium \
  --key-name your-key-pair \
  --security-group-ids sg-xxx \
  --count 1

# SSH into instance
ssh -i your-key.pem ubuntu@YOUR_EC2_IP

# On the EC2 instance:
sudo apt update
sudo apt install -y docker.io docker-compose git
git clone https://github.com/honourjesus/research-paper-audiobook.git
cd research-paper-audiobook
sudo docker-compose up -d
```

Your app is now running at `http://YOUR_EC2_IP:8000`

### AWS Cost Estimate
- EC2 t2.medium: ~$33/month
- ECS Fargate (1 vCPU, 2GB): ~$30/month
- ECR storage: ~$1/month

---

## ☁️ GCP Deployment (Step by Step)

### Prerequisites
- Google Cloud account with billing enabled
- `gcloud` CLI installed and authenticated (`gcloud auth login`)
- Docker installed

### Option A: Google Cloud Run — Recommended (Serverless)

**Step 1: Set up project**
```bash
# Set your project
gcloud config set project YOUR_PROJECT_ID

# Enable required APIs
gcloud services enable run.googleapis.com
gcloud services enable containerregistry.googleapis.com
```

**Step 2: Build and push to Google Container Registry**
```bash
# Build with Cloud Build (no local Docker needed)
gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/research-paper-audiobook

# OR build locally and push
docker build -t gcr.io/YOUR_PROJECT_ID/research-paper-audiobook .
docker push gcr.io/YOUR_PROJECT_ID/research-paper-audiobook
```

**Step 3: Deploy to Cloud Run**
```bash
gcloud run deploy research-paper-audiobook \
  --image gcr.io/YOUR_PROJECT_ID/research-paper-audiobook \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 2 \
  --timeout 300s \
  --port 8000
```

Cloud Run gives you a URL like:
`https://research-paper-audiobook-xxxx-uc.a.run.app`

Your API is live. Cloud Run scales to zero when not in use, so you pay nothing when idle.

### Option B: Google Compute Engine (VM) — More control

```bash
# Create VM instance
gcloud compute instances create paper-audiobook-vm \
  --zone=us-central1-a \
  --machine-type=e2-standard-2 \
  --image-family=ubuntu-2204-lts \
  --image-project=ubuntu-os-cloud \
  --tags=http-server,https-server

# Allow port 8000
gcloud compute firewall-rules create allow-8000 \
  --allow tcp:8000 \
  --target-tags http-server

# SSH into VM
gcloud compute ssh paper-audiobook-vm --zone=us-central1-a

# On the VM:
sudo apt update
sudo apt install -y docker.io docker-compose git
git clone https://github.com/honourjesus/research-paper-audiobook.git
cd research-paper-audiobook
sudo docker-compose up -d
```

### GCP Cost Estimate
- Cloud Run: Pay per request (~$0 for low traffic, scales automatically)
- Compute Engine e2-standard-2: ~$50/month
- Container Registry: ~$0.10/GB/month

### AWS vs GCP — Which Should You Use?

| Factor | AWS | GCP |
|--------|-----|-----|
| Market share | Largest (33%) | Third (10%) |
| Learning curve | Steeper | Easier |
| Free tier | 12 months | Always free tier |
| Best for | Enterprise jobs | ML/AI projects |
| Cloud Run equivalent | App Runner | Cloud Run |

**Recommendation:** Use **GCP Cloud Run** for this project because it's simpler, has a generous free tier, and GCP is known for ML/AI workloads which matches this project perfectly.

---

##  Evaluation Metrics

The API evaluates conversion quality using:
- **ROUGE Score** — measures text overlap between original and converted
- **BLEU Score** — measures translation quality
- **WER (Word Error Rate)** — measures transcription accuracy

---

##  Future Improvements

- [ ] Add ArXiv ID direct input (no PDF upload needed)
- [ ] Support for multiple languages
- [ ] Chapter markers in audio output
- [ ] Web frontend with drag-and-drop upload
- [ ] Streaming audio (start listening before full conversion)
- [ ] Support for EPUB and HTML papers

---

## 📄 License

MIT License — free to use, modify and distribute.

---

##  Author

**Honour Jesus**
- GitHub: [@honourjesus](https://github.com/honourjesus)

---

*Built with FastAPI, PyMuPDF, pdfplumber, Ollama, and gTTS*
