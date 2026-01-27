# Docker Deployment Guide

## Quick Start

### 1. Configure Environment

```bash
cd docker
cp .env.example .env
vim .env  # Set your API_KEY
```

**Required Configuration:**
- `API_KEY`: 

### 2. Start Service

```bash
docker compose up -d
```

### 3. Access Application

Open browser: http://localhost:8501
