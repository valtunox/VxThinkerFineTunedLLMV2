# VA LLM Specialist Model Deployment Guide

This guide covers deployment options for VA LLM Specialist Model on Virtual Machines and Azure Kubernetes Service (AKS).

## Table of Contents

- [Prerequisites](#prerequisites)
- [Architecture Overview](#architecture-overview)
- [Deployment Options](#deploymentsdfsdf-options)
  - [Option 1: Virtual Machine Deployment](#option-1-virtual-machine-deployment)
  - [Option 2: Azure Kubernetes Service (AKS) Deployment](#option-2-azure-kubernetes-service-aks-deployment)
- [CI/CD Pipelines](#cicd-pipelines)
- [Configuration](#configuration)
- [Monitoring and Troubleshooting](#monitoring-and-troubleshooting)
- [Scaling and Performance](#scaling-and-performance)

---

## Prerequisites

### General Requirements

- Docker and Docker Compose installed
- Access to container registry (Docker Hub or Azure Container Registry)
- SSH access to target VM (for VM deployment)
- kubectl configured (for Kubernetes deployment)
- Required environment variables and secrets configured

### Application Requirements

- **Python**: 3.11
- **Memory**: Minimum 2GB RAM (4GB recommended for production)
- **CPU**: 2+ cores recommended
- **Storage**: 20GB+ for data, models, and vectorstore
- **Network**: Port 8000 accessible

### Data Requirements

Ensure the following data files are available in `app/data/`:
- CSV knowledge base files (`*.csv`)
- PDF documents (optional)
- Vector index artifacts (`vectorstore/faiss_index.bin`)
- Trained LLM artifacts (`model/` directory)

---

## Architecture Overview

VA LLM Specialist Model is a FastAPI application that provides:
- **Vector Search**: FAISS-based semantic search
- **LLM Inference**: Local model inference (optional)
- **RAG Pipeline**: Retrieval-Augmented Generation
- **Multiple API Versions**: v1 (RAG + Reasoning), v2 (NLP + Documents), v3 (Incidents)

### Key Components

```
┌─────────────────────────────────────────────────────────┐
│              VA LLM Specialist Model Application          │
├─────────────────────────────────────────────────────────┤
│  FastAPI Server (Port 8000)                             │
│  ├── Vector Store (FAISS)                                │
│  ├── Embeddings (sentence-transformers)                  │
│  ├── Reasoning Engine                                    │
│  └── LLM Model (optional)                                │
└─────────────────────────────────────────────────────────┘
```

---

## Deployment Options

### Option 1: Virtual Machine Deployment

#### Quick Start

1. **Prepare the VM**
   ```bash
   # SSH into your VM
   ssh user@your-vm-ip
   
   # Install Docker and Docker Compose
   curl -fsSL https://get.docker.com -o get-docker.sh
   sudo sh get-docker.sh
   sudo usermod -aG docker $USER
   sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
   sudo chmod +x /usr/local/bin/docker-compose
   ```

2. **Deploy using Docker Compose**
   ```bash
   # Clone or copy your project files
   mkdir -p ~/va-llm-v1
   cd ~/va-llm-v1
   
   # Copy docker-compose.yml and create .env file
   cat > .env << EOF
   ENVIRONMENT=production
   PORT=8000
   HOST=0.0.0.0
   DOCKER_IMAGE=your-registry/va-llm-v1:latest
   DATABASE_HOST=your-db-host
   DATABASE_PORT=5432
   DATABASE_NAME=recruitment
   DATABASE_USER=postgres
   DATABASE_PASSWORD=your-password
   GOOGLE_API_KEY=your-key
   OPENAI_API_KEY=your-key
   ANTHROPIC_API_KEY=your-key
   APOLLO_API_KEY=your-key
   AWS_ACCESS_KEY_ID=your-key
   AWS_SECRET_ACCESS_KEY=your-secret
   AWS_REGION=us-east-1
   S3_BUCKET=your-bucket
   EOF
   
   # Start the service
   docker-compose up -d
   
   # Check logs
   docker-compose logs -f
   ```

3. **Verify Deployment**
   ```bash
   # Health check
   curl http://localhost:8000/health
   
   # API documentation
   curl http://localhost:8000/docs
   ```

#### Manual Deployment Steps

1. **Pull the Docker Image**
   ```bash
   docker pull your-registry/va-llm-v1:latest
   ```

2. **Run the Container**
   ```bash
   docker run -d \
     --name va-llm-v1 \
     -p 8000:8000 \
     -v $(pwd)/app/data:/app/data \
     -e ENVIRONMENT=production \
     -e DATABASE_HOST=your-db-host \
     -e DATABASE_PASSWORD=your-password \
     your-registry/va-llm-v1:latest
   ```

#### Using Azure Pipelines

The `azure-pipelines.yml` includes a VM deployment stage. Configure:

1. **Service Connections**:
   - `VM_SSH_CONNECTION`: SSH endpoint for your VM
   - `ACR_SERVICE_CONNECTION`: Azure Container Registry connection

2. **Variables**:
   - `DeployTarget`: Set to `VM` for VM deployment
   - `VM_HOST`: VM IP address or hostname
   - `VM_USER`: SSH username

3. **Trigger Deployment**:
   ```bash
   # Push to main branch or manually trigger pipeline
   az pipelines run --name "Your Pipeline Name"
   ```

#### Using GitHub Actions

The `.github/workflows/data-pipeline.yml` includes VM deployment. Configure:

1. **Secrets** (in GitHub repository settings):
   - `VM_HOST`: VM IP address
   - `VM_USERNAME`: SSH username
   - `VM_SSH_KEY`: Private SSH key
   - `VM_SSH_PORT`: SSH port (default: 22)
   - `DOCKER_USERNAME`: Docker Hub username
   - `DOCKER_PASSWORD`: Docker Hub password/token

2. **Trigger Deployment**:
   - Push to `main` or `production` branch
   - Or use workflow_dispatch with `deploy_target: vm`

---

### Option 2: Azure Kubernetes Service (AKS) Deployment

#### Prerequisites

- Azure subscription with AKS cluster created
- kubectl configured and connected to AKS
- Azure CLI installed
- Container registry (ACR or Docker Hub)

#### Quick Start

1. **Create Namespace**
   ```bash
   kubectl create namespace va-llm-v1
   ```

2. **Create Docker Registry Secret**
   ```bash
   # For Docker Hub
   kubectl create secret docker-registry docker-registry-secret \
     --docker-server=docker.io \
     --docker-username=your-username \
     --docker-password=your-password \
     --namespace=va-llm-v1
   
   # For Azure Container Registry
   kubectl create secret docker-registry docker-registry-secret \
     --docker-server=your-registry.azurecr.io \
     --docker-username=your-username \
     --docker-password=your-password \
     --namespace=va-llm-v1
   ```

3. **Create Application Secrets**
   ```bash
   # Edit k8s/service.yaml and update the secret values, then apply
   kubectl apply -f k8s/service.yaml
   ```

4. **Deploy Application**
   ```bash
   kubectl apply -f k8s/deployment.yaml
   ```

5. **Verify Deployment**
   ```bash
   # Check pods
   kubectl get pods -n va-llm-v1
   
   # Check services
   kubectl get services -n va-llm-v1
   
   # Get LoadBalancer IP
   kubectl get service va-llm-v1-service -n va-llm-v1
   
   # Check logs
   kubectl logs -f deployment/va-llm-v1 -n va-llm-v1
   ```

#### Using Azure Pipelines

1. **Configure Service Connections**:
   - `AZURE_SERVICE_CONNECTION`: Azure subscription connection
   - `ACR_SERVICE_CONNECTION`: Azure Container Registry connection

2. **Set Variables**:
   - `DeployTarget`: Set to `K8S` for Kubernetes deployment
   - `AKS_CLUSTER_NAME`: Your AKS cluster name
   - `AKS_RESOURCE_GROUP`: Resource group containing AKS cluster
   - `ACR_NAME`: Azure Container Registry name

3. **Deploy**:
   ```bash
   az pipelines run --name "Your Pipeline Name"
   ```

#### Using GitHub Actions

1. **Configure Secrets**:
   - `AZURE_CREDENTIALS`: Azure service principal credentials (JSON)
   - `AKS_RESOURCE_GROUP`: Resource group name
   - `AKS_CLUSTER_NAME`: AKS cluster name
   - `DOCKER_USERNAME`: Container registry username
   - `DOCKER_PASSWORD`: Container registry password

2. **Deploy**:
   - Push to `main` or `production` branch
   - Or use workflow_dispatch with `deploy_target: kubernetes`

#### Persistent Storage

The deployment uses PersistentVolumeClaim for data storage. Ensure your AKS cluster has a storage class configured:

```bash
# Check available storage classes
kubectl get storageclass

# If using Azure, the default 'managed-premium' should work
# For custom storage class, update k8s/deployment.yaml
```

#### Scaling

```bash
# Scale manually
kubectl scale deployment va-llm-v1 --replicas=3 -n va-llm-v1

# Or configure Horizontal Pod Autoscaler
kubectl autoscale deployment va-llm-v1 \
  --cpu-percent=70 \
  --min=2 \
  --max=10 \
  -n va-llm-v1
```

---

## CI/CD Pipelines

### Azure Pipelines

The `azure-pipelines.yml` includes:

1. **Build Stage**: Builds and pushes Docker image to ACR
2. **Deploy to VM Stage**: Deploys to VM (conditional on `DeployTarget=VM`)
3. **Deploy to AKS Stage**: Deploys to Kubernetes (conditional on `DeployTarget=K8S`)

**Configuration Steps**:

1. Create service connections in Azure DevOps
2. Set pipeline variables
3. Configure `DeployTarget` variable to choose deployment target

### GitHub Actions

The `.github/workflows/data-pipeline.yml` includes:

1. **Test Job**: Runs linting and basic tests
2. **Build Job**: Builds and pushes Docker image
3. **Deploy VM Job**: Deploys to VM (on main/production branches)
4. **Deploy Kubernetes Job**: Deploys to AKS (on main/production branches)

**Configuration Steps**:

1. Add required secrets to GitHub repository
2. Configure environment protection rules (optional)
3. Push to trigger or use workflow_dispatch

---

## Configuration

### Environment Variables

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `ENVIRONMENT` | Deployment environment | `production` | No |
| `PORT` | Application port | `8000` | No |
| `HOST` | Bind address | `0.0.0.0` | No |
| `VALLM_AUTO_PRECOMPUTE` | Auto-build FAISS index | `true` | No |
| `VALLM_AUTO_TRAIN` | Auto-train LLM model | `false` | No |
| `USE_CUDA` | Enable GPU support | `false` | No |
| `DATABASE_HOST` | PostgreSQL host | - | Yes (if using DB) |
| `DATABASE_PORT` | PostgreSQL port | `5432` | No |
| `DATABASE_NAME` | Database name | - | Yes (if using DB) |
| `DATABASE_USER` | Database user | - | Yes (if using DB) |
| `DATABASE_PASSWORD` | Database password | - | Yes (if using DB) |
| `GOOGLE_API_KEY` | Google API key | - | Optional |
| `OPENAI_API_KEY` | OpenAI API key | - | Optional |
| `ANTHROPIC_API_KEY` | Anthropic API key | - | Optional |
| `APOLLO_API_KEY` | Apollo API key | - | Optional |
| `AWS_ACCESS_KEY_ID` | AWS access key | - | Optional |
| `AWS_SECRET_ACCESS_KEY` | AWS secret key | - | Optional |
| `AWS_REGION` | AWS region | `us-east-1` | No |
| `S3_BUCKET` | S3 bucket name | - | Optional |

### Kubernetes Secrets

Update `k8s/service.yaml` with your actual secret values:

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: va-llm-v1-secrets
  namespace: va-llm-v1
type: Opaque
stringData:
  database-host: "your-db-host"
  database-password: "your-secure-password"
  # ... other secrets
```

### Resource Limits

Adjust in `k8s/deployment.yaml`:

```yaml
resources:
  requests:
    memory: "2Gi"
    cpu: "1000m"
  limits:
    memory: "4Gi"
    cpu: "2000m"
```

For GPU support, add:

```yaml
resources:
  limits:
    nvidia.com/gpu: 1
```

---

## Monitoring and Troubleshooting

### Health Checks

The application exposes a health endpoint:

```bash
curl http://your-host:8000/health
```

Expected response:
```json
{"status": "healthy"}
```

### Logs

**Docker/VM**:
```bash
# Docker Compose
docker-compose logs -f

# Docker
docker logs va-llm-v1 -f
```

**Kubernetes**:
```bash
# Pod logs
kubectl logs -f deployment/va-llm-v1 -n va-llm-v1

# All pods
kubectl logs -f -l app=va-llm-v1 -n va-llm-v1
```

### Common Issues

1. **Container fails to start**
   - Check logs: `docker logs va-llm-v1` or `kubectl logs ...`
   - Verify environment variables
   - Check data directory permissions

2. **Health check fails**
   - Ensure port 8000 is accessible
   - Check application logs for errors
   - Verify data files exist in `/app/data`

3. **Out of memory**
   - Increase container memory limits
   - Reduce number of replicas
   - Check vectorstore size

4. **Slow performance**
   - Enable GPU if available (`USE_CUDA=true`)
   - Increase CPU/memory resources
   - Check network latency

### Application Logs

Access application logs via API:

```bash
# Recent logs
curl http://your-host:8000/logs?lines=100

# Log statistics
curl http://your-host:8000/logs/stats
```

---

## Scaling and Performance

### Horizontal Scaling (Kubernetes)

```bash
# Scale deployment
kubectl scale deployment va-llm-v1 --replicas=5 -n va-llm-v1

# Auto-scaling
kubectl autoscale deployment va-llm-v1 \
  --cpu-percent=70 \
  --min=2 \
  --max=10 \
  -n va-llm-v1
```

### Vertical Scaling

Update resource requests/limits in `k8s/deployment.yaml`:

```yaml
resources:
  requests:
    memory: "4Gi"
    cpu: "2000m"
  limits:
    memory: "8Gi"
    cpu: "4000m"
```

### Performance Optimization

1. **Enable GPU** (if available):
   ```yaml
   env:
   - name: USE_CUDA
     value: "true"
   ```

2. **Pre-compute FAISS index**:
   ```bash
   python app/precompute.py
   ```

3. **Use persistent volumes** for data/models to avoid re-computation

4. **Configure session affinity** (already configured in service.yaml)

---

## Security Considerations

1. **Secrets Management**:
   - Never commit secrets to version control
   - Use Kubernetes secrets or Azure Key Vault
   - Rotate credentials regularly

2. **Network Security**:
   - Use internal load balancers for AKS
   - Configure firewall rules
   - Enable TLS/HTTPS (configure ingress)

3. **Container Security**:
   - Use non-root user in container
   - Scan images for vulnerabilities
   - Keep base images updated

4. **API Security**:
   - Implement authentication/authorization
   - Rate limiting
   - Input validation

---

## Rollback Procedures

### VM Deployment

```bash
# SSH into VM
ssh user@your-vm-ip

# Stop current container
docker-compose down

# Pull previous version
docker pull your-registry/va-llm-v1:previous-tag

# Update docker-compose.yml to use previous tag
# Then restart
docker-compose up -d
```

### Kubernetes Deployment

```bash
# Rollback to previous revision
kubectl rollout undo deployment/va-llm-v1 -n va-llm-v1

# Check rollout history
kubectl rollout history deployment/va-llm-v1 -n va-llm-v1

# Rollback to specific revision
kubectl rollout undo deployment/va-llm-v1 --to-revision=2 -n va-llm-v1
```

---

## Support and Resources

- **API Documentation**: `http://your-host:8000/docs`
- **Health Check**: `http://your-host:8000/health`
- **Application Logs**: `http://your-host:8000/logs`

For issues or questions, check the application logs and health endpoints first.
sdf
