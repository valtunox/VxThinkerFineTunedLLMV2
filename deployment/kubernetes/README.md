# Kubernetes Deployment Files

Production-ready Kubernetes deployment manifests for VaLLM.

## 📁 Directory Structure

```
kubernetes/
├── eks/
│   └── deploy_kubernetes_eks.yml    # Complete EKS deployment
├── aks/
│   └── deploy_kubernetes_aks.yml    # Complete AKS deployment
├── DEPLOYMENT_GUIDE.md              # Detailed deployment guide
└── README.md                        # This file
```

## 🚀 Quick Deploy

### EKS
```bash
kubectl apply -f deployment/kubernetes/eks/deploy_kubernetes_eks.yml
```

### AKS
```bash
kubectl apply -f deployment/kubernetes/aks/deploy_kubernetes_aks.yml
```

## 📋 What's Included

### Core Components
- ✅ **Deployment** - Main application with 3+ replicas
- ✅ **Service** - LoadBalancer/Ingress for external access
- ✅ **HPA** - Horizontal Pod Autoscaler (3-50 replicas)
- ✅ **VPA** - Vertical Pod Autoscaler (optional)
- ✅ **PDB** - Pod Disruption Budget
- ✅ **Network Policies** - Security rules
- ✅ **ConfigMaps** - Application configuration
- ✅ **Secrets** - Sensitive data (use external secrets in production)

### Data Pipeline
- ✅ **S3 Fetcher** - CronJob for AWS S3 datasets (EKS)
- ✅ **Azure Blob Fetcher** - CronJob for Azure Blob Storage (AKS)
- ✅ **URL Fetcher** - CronJob for URL-based datasets
- ✅ **Dataset Processor** - CronJob for data processing

### Infrastructure
- ✅ **Persistent Volumes** - Data and logs storage
- ✅ **Service Accounts** - IAM/Managed Identity integration
- ✅ **Redis** - Rate limiting (optional, use managed service in production)
- ✅ **Monitoring** - Prometheus ServiceMonitor

## 🔧 Prerequisites

See [DEPLOYMENT_GUIDE.md](./DEPLOYMENT_GUIDE.md) for detailed setup instructions.

## 📊 Features

### High Availability
- Multi-replica deployment
- Pod anti-affinity
- Health checks (liveness, readiness, startup)
- Pod disruption budgets

### Scalability
- Horizontal Pod Autoscaler
- Vertical Pod Autoscaler
- Resource limits and requests
- Efficient resource utilization

### Security
- Service accounts with IAM/Managed Identity
- Network policies
- Secrets management
- Non-root containers

### Data Pipeline
- Automated dataset fetching
- Support for S3, Azure Blob, and URLs
- Data validation and deduplication
- Scheduled processing

## 🔍 Monitoring

- Prometheus metrics at `/metrics`
- Health endpoints: `/health`, `/health/live`, `/health/ready`
- ServiceMonitor for Prometheus Operator

## 📝 Customization

Before deploying, update:
1. Container image registry URLs
2. IAM/Managed Identity configurations
3. Storage class names
4. Certificate ARNs (for TLS)
5. Domain names (for Ingress)
6. Resource limits (based on workload)

## 🆘 Support

For issues or questions:
1. Check [DEPLOYMENT_GUIDE.md](./DEPLOYMENT_GUIDE.md) troubleshooting section
2. Review Kubernetes events: `kubectl get events -n vallm-production`
3. Check pod logs: `kubectl logs -f deployment/vallm-api -n vallm-production`
