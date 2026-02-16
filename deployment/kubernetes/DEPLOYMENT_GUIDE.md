# Kubernetes Deployment Guide

This guide covers deploying VaLLM to production Kubernetes clusters on AWS EKS and Azure AKS.

## 📋 Prerequisites

### For EKS (AWS)
- AWS CLI configured
- kubectl configured for EKS cluster
- ECR repository for container images
- EFS file system (for persistent storage)
- IAM role with S3 access permissions
- AWS Load Balancer Controller installed

### For AKS (Azure)
- Azure CLI configured
- kubectl configured for AKS cluster
- Azure Container Registry (ACR)
- Azure Storage Account with Blob Storage
- Managed Identity with Storage Blob Data Contributor role
- Application Gateway Ingress Controller (optional)

## 🚀 Quick Start

### EKS Deployment

1. **Configure IAM Role for Service Account (IRSA)**
   ```bash
   # Create IAM role
   eksctl create iamserviceaccount \
     --name vallm-service-account \
     --namespace vallm-production \
     --cluster your-cluster-name \
     --role-name vallm-eks-role \
     --attach-policy-arn arn:aws:iam::aws:policy/AmazonS3ReadOnlyAccess \
     --approve
   ```

2. **Create EFS Storage Class**
   ```bash
   # Install EFS CSI driver
   kubectl apply -k "github.com/kubernetes-sigs/aws-efs-csi-driver/deploy/kubernetes/overlays/stable/?ref=release-1.7"
   
   # Create storage class
   kubectl apply -f - <<EOF
   apiVersion: storage.k8s.io/v1
   kind: StorageClass
   metadata:
     name: efs-sc
   provisioner: efs.csi.aws.com
   parameters:
     provisioningMode: efs-ap
     fileSystemId: fs-xxxxxxxxx
     directoryPerms: "0755"
   EOF
   ```

3. **Update deployment file**
   - Replace `YOUR_ECR_REGISTRY` with your ECR registry URL
   - Replace `ACCOUNT_ID` in ServiceAccount annotation
   - Update certificate ARN in Ingress annotations
   - Update S3 bucket name in secrets

4. **Deploy**
   ```bash
   kubectl apply -f deployment/kubernetes/eks/deploy_kubernetes_eks.yml
   ```

### AKS Deployment

1. **Create Managed Identity**
   ```bash
   # Create managed identity
   az identity create \
     --name vallm-identity \
     --resource-group your-resource-group
   
   # Get client ID
   CLIENT_ID=$(az identity show \
     --name vallm-identity \
     --resource-group your-resource-group \
     --query clientId -o tsv)
   
   # Grant Storage Blob Data Contributor role
   az role assignment create \
     --role "Storage Blob Data Contributor" \
     --assignee $CLIENT_ID \
     --scope /subscriptions/SUBSCRIPTION_ID/resourceGroups/RESOURCE_GROUP/providers/Microsoft.Storage/storageAccounts/STORAGE_ACCOUNT
   ```

2. **Enable Workload Identity**
   ```bash
   # Enable workload identity on AKS
   az aks update \
     --name your-cluster-name \
     --resource-group your-resource-group \
     --enable-oidc-issuer \
     --enable-workload-identity
   
   # Create federated credential
   az identity federated-credential create \
     --name vallm-federated-credential \
     --identity-name vallm-identity \
     --resource-group your-resource-group \
     --issuer $(az aks show --name your-cluster-name --resource-group your-resource-group --query "oidcIssuerProfile.issuerUrl" -o tsv) \
     --subject system:serviceaccount:vallm-production:vallm-service-account \
     --audience api://AzureADTokenExchange
   ```

3. **Create Storage Account**
   ```bash
   az storage account create \
     --name vallmstorage \
     --resource-group your-resource-group \
     --location eastus \
     --sku Standard_LRS
   
   az storage container create \
     --name datasets \
     --account-name vallmstorage \
     --auth-mode login
   ```

4. **Update deployment file**
   - Replace `YOUR_ACR_REGISTRY` with your ACR registry URL
   - Replace `MANAGED_IDENTITY_CLIENT_ID` with actual client ID
   - Update storage account name

5. **Deploy**
   ```bash
   kubectl apply -f deployment/kubernetes/aks/deploy_kubernetes_aks.yml
   ```

## 🔧 Configuration

### Environment Variables

Key environment variables can be set via ConfigMap or Secrets:

**ConfigMap (vallm-config)**:
- `ENVIRONMENT`: production
- `DATABASE_HOST`: PostgreSQL service
- `REDIS_HOST`: Redis service
- `VALLM_TENANT_RATE_LIMIT_ENABLED`: true
- `VALLM_DEFAULT_TPM_LIMIT`: 1000000

**Secrets (vallm-secrets)**:
- Database credentials
- API keys
- Redis password
- S3 bucket name (EKS) or Storage account (AKS)

### Resource Limits

Default resource requests/limits:
- **Requests**: 4Gi memory, 2000m CPU
- **Limits**: 8Gi memory, 4000m CPU

Adjust based on your workload requirements.

### Scaling

**Horizontal Pod Autoscaler (HPA)**:
- Min replicas: 3
- Max replicas: 50
- CPU target: 70%
- Memory target: 80%

**Vertical Pod Autoscaler (VPA)** (optional):
- Auto mode enabled
- Min: 2Gi memory, 1000m CPU
- Max: 16Gi memory, 8000m CPU

## 📊 Data Pipeline

### S3 Dataset Fetcher (EKS)

Runs daily at 2 AM UTC:
```bash
# Manual trigger
kubectl create job --from=cronjob/vallm-s3-dataset-fetcher vallm-s3-fetcher-manual -n vallm-production
```

### Azure Blob Fetcher (AKS)

Runs daily at 2 AM UTC:
```bash
# Manual trigger
kubectl create job --from=cronjob/vallm-azure-blob-fetcher vallm-blob-fetcher-manual -n vallm-production
```

### URL Dataset Fetcher

Runs every 6 hours:
```bash
# Manual trigger
kubectl create job --from=cronjob/vallm-url-dataset-fetcher vallm-url-fetcher-manual -n vallm-production
```

### Dataset Processor

Runs daily at 3 AM UTC (after fetchers):
```bash
# Manual trigger
kubectl create job --from=cronjob/vallm-dataset-processor vallm-processor-manual -n vallm-production
```

## 🔍 Monitoring

### Health Checks

- **Liveness**: `/health/live` - Checks if pod is alive
- **Readiness**: `/health/ready` - Checks if pod can accept traffic
- **Startup**: `/health` - Initial health check

### Metrics

Prometheus metrics available at `/metrics`:
- HTTP request metrics
- Model performance metrics
- Rate limiting metrics

### Logs

View logs:
```bash
# Application logs
kubectl logs -f deployment/vallm-api -n vallm-production

# Data pipeline logs
kubectl logs -f job/vallm-s3-fetcher-manual -n vallm-production
```

## 🔐 Security

### Secrets Management

**Production Recommendations**:
- **EKS**: Use AWS Secrets Manager with External Secrets Operator
- **AKS**: Use Azure Key Vault with Secrets Store CSI Driver

### Network Policies

Network policies restrict:
- Ingress: Only from ingress controller and same namespace
- Egress: Database, Redis, and external APIs (HTTPS)

### Service Accounts

- **EKS**: Uses IRSA (IAM Roles for Service Accounts)
- **AKS**: Uses Workload Identity with Managed Identity

## 🛠️ Troubleshooting

### Pods Not Starting

1. Check events:
   ```bash
   kubectl describe pod <pod-name> -n vallm-production
   ```

2. Check logs:
   ```bash
   kubectl logs <pod-name> -n vallm-production
   ```

3. Verify secrets:
   ```bash
   kubectl get secret vallm-secrets -n vallm-production -o yaml
   ```

### Data Pipeline Issues

1. Check CronJob status:
   ```bash
   kubectl get cronjobs -n vallm-production
   ```

2. Check Job history:
   ```bash
   kubectl get jobs -n vallm-production
   ```

3. View Job logs:
   ```bash
   kubectl logs job/<job-name> -n vallm-production
   ```

### Storage Issues

1. Check PVC status:
   ```bash
   kubectl get pvc -n vallm-production
   ```

2. Check storage class:
   ```bash
   kubectl get storageclass
   ```

### Rate Limiting Issues

1. Check Redis connection:
   ```bash
   kubectl exec -it deployment/vallm-redis -n vallm-production -- redis-cli ping
   ```

2. Check Redis logs:
   ```bash
   kubectl logs deployment/vallm-redis -n vallm-production
   ```

## 📈 Scaling

### Manual Scaling

```bash
# Scale deployment
kubectl scale deployment vallm-api --replicas=10 -n vallm-production
```

### Auto Scaling

HPA automatically scales based on CPU/memory usage. Monitor with:
```bash
kubectl get hpa vallm-api-hpa -n vallm-production
```

## 🔄 Updates

### Rolling Update

```bash
# Update image
kubectl set image deployment/vallm-api vallm-api=YOUR_REGISTRY/vallm:v2.0.0 -n vallm-production

# Monitor rollout
kubectl rollout status deployment/vallm-api -n vallm-production
```

### Rollback

```bash
kubectl rollout undo deployment/vallm-api -n vallm-production
```

## 📝 Maintenance

### Database Migrations

Run migrations via init container or Job:
```bash
kubectl create job --from=cronjob/vallm-migrate vallm-migrate-manual -n vallm-production
```

### Backup

Backup persistent volumes:
- **EKS**: Use AWS Backup or snapshot EFS
- **AKS**: Use Azure Backup or snapshot Azure Files

## 🎯 Best Practices

1. **Use managed services** for PostgreSQL and Redis when possible
2. **Enable monitoring** with Prometheus and Grafana
3. **Set up alerts** for pod failures, high latency, errors
4. **Regular backups** of persistent volumes
5. **Test disaster recovery** procedures
6. **Monitor costs** - especially for storage and compute
7. **Use resource quotas** to prevent resource exhaustion
8. **Enable pod security policies** for additional security

## 📚 Additional Resources

- [EKS Best Practices](https://aws.github.io/aws-eks-best-practices/)
- [AKS Best Practices](https://docs.microsoft.com/azure/aks/best-practices)
- [Kubernetes Documentation](https://kubernetes.io/docs/)
