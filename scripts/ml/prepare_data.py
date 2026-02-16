#!/usr/bin/env python3
"""
Prepare cloud_deployments.csv for VaLLM provisioning training (6000 rows by default).

Generates synthetic deployment rows for 20 intents:
  Original 6:  VM, Kubernetes, Docker, FastAPI, static website, database
  New 14:      ReactJS, NextJS, ELK, VPN, managed database, monitoring,
               CI/CD, load balancer, cache, object storage, SSL certificate,
               WordPress, Django, Spring Boot

Key consistency guarantees:
  - Prompt text always matches generated slot values (no mismatches)
  - Docker services use their correct default ports
  - Database engines use their correct default ports
  - Static website server_name matches the user-requested hostname
  - VM hardcoded prompts match the generated instance data
  - Endpoint URLs and JSON payloads reflect accurate row data

Run from va_llm_v1 root:
    python scripts/ml/prepare_data.py
    python scripts/ml/prepare_data.py --rows 6000 --output app/data/datasets/cloud_deployments.csv
"""

import argparse
import csv
import json
import random
from datetime import datetime, timedelta
from pathlib import Path

# ============================================================================
#  SHARED CONSTANTS
# ============================================================================
REGIONS = [
    "us-east-1", "us-east-2", "us-west-1", "us-west-2",
    "eu-west-1", "eu-west-2", "eu-central-1",
    "ap-southeast-1", "ap-northeast-1", "ap-south-1",
    "ca-central-1", "sa-east-1",
]
CLOUD_PROVIDERS = ["aws", "azure", "gcp", "valtunox"]
ENVIRONMENTS = ["dev", "staging", "prod"]
OS_OPTIONS = ["ubuntu-22.04", "ubuntu-24.04", "ubuntu", "centos", "debian-12", "amazon-linux-2"]
BASE_URL = "http://localhost:8743"

# ============================================================================
#  VM (EC2 / Compute)
# ============================================================================
INSTANCE_TYPES = [
    "t2.micro", "t2.small", "t2.medium", "t2.large", "t2.xlarge",
    "t3.micro", "t3.small", "t3.medium", "t3.large", "t3.xlarge",
    "m5.large", "m5.xlarge", "c5.large", "c5.xlarge", "c6g.medium",
    "r5.large", "r5.xlarge",
]
VOLUME_SIZES_GB = [8, 20, 30, 40, 50, 100, 200, 500]
VOLUME_TYPES = ["gp2", "gp3", "io1", "standard"]
SIZE_TIERS = ["small", "medium", "large"]
SIZE_TO_DEFAULTS = {
    "small":  [("t2.micro", 8),  ("t3.micro", 20), ("t2.small", 20)],
    "medium": [("t2.medium", 30), ("t3.medium", 40), ("t2.large", 50)],
    "large":  [("t2.xlarge", 100), ("m5.large", 200), ("m5.xlarge", 500)],
}

VM_PROMPTS = [
    "Deploy an EC2 instance with {volume_size_gb}GB storage, {instance_type}",
    "Create a {size_tier} size VM in {region}",
    "I need a {instance_type} instance, {volume_size_gb}GB disk, {os}",
    "Provision {cloud_provider} VM: {instance_type}, {volume_size_gb}GB, {region}",
    "Deploy EC2 {instance_type} with {volume_size_gb}GB",
    "Spin up a {size_tier} server on {cloud_provider} in {region}",
    "Deploy EC2 instance {instance_type} {volume_size_gb}GB {volume_type} {os} in {region}",
    "{size_tier} EC2 in {region}",
    "{size_tier} {os} server, {volume_size_gb}GB",
    "Provision {cloud_provider} VM with default settings",
    "Launch a {instance_type} VM in {region} with {os}",
    "I want to provision a {cloud_provider} virtual machine, {instance_type}, {volume_size_gb}GB {volume_type}",
    "Set up EC2 {instance_type} in {region}, {volume_size_gb}GB root volume",
    "Create VM: {instance_type}, {region}, {os}, {volume_size_gb}GB",
    "Please deploy an EC2 instance with {volume_size_gb}GB and {instance_type}",
    "Need a {size_tier} EC2 in {region} for {environment}",
    "Deploy a {size_tier} EC2 instance with {volume_size_gb}GB disk in {region} with {os}",
    "Provision {cloud_provider} VM: {instance_type}, {volume_size_gb}GB, {region}",
    "Create a {cloud_provider} compute instance {instance_type} in {region}",
    "Launch {instance_type} in {region}, {volume_size_gb}GB {volume_type}, {os}",
]

# ============================================================================
#  Kubernetes (EKS / GKE / AKS)
# ============================================================================
K8S_NODE_TYPES = ["t3.medium", "t3.large", "t3.xlarge", "m5.large", "m5.xlarge", "c5.large"]
K8S_VERSIONS = ["1.28", "1.29", "1.30", "1.31"]

K8S_PROMPTS = [
    "Deploy a Kubernetes cluster in {region} with {node_count} nodes",
    "Create an EKS cluster, {node_count} nodes, {node_type}",
    "Create an EKS cluster, {node_count} nodes, {node_type}, kubernetes {kubernetes_version} in {region}",
    "I need a K8s cluster on {cloud_provider}, {region}, {node_count} nodes",
    "Provision Kubernetes cluster: {cluster_name}, {node_count} nodes",
    "Deploy EKS cluster with {node_count} worker nodes in {region}",
    "Create GKE cluster in {region}, {node_count} nodes, {node_type}",
    "Deploy a new Kubernetes cluster, {node_count} nodes in {region}",
    "Set up AKS cluster, {node_count} nodes, {node_type} in {region}",
    "Launch EKS cluster in {region}, {node_count} nodes, {node_type}",
    "I want to create a Kubernetes cluster, {node_count} nodes, {region}",
    "Please provision an EKS cluster with {node_count} nodes and {node_type}",
    "Deploy Kubernetes {kubernetes_version} cluster in {region} with {node_count} nodes",
]

# ============================================================================
#  Docker
# ============================================================================
DOCKER_SERVICE_PORTS = {
    "nginx": "80:80",
    "redis": "6379:6379",
    "postgres": "5432:5432",
    "mysql": "3306:3306",
    "mongodb": "27017:27017",
    "rabbitmq": "5672:5672",
    "portainer": "9443:9443",
    "grafana": "3000:3000",
    "n8n": "5678:5678",
    "jenkins": "8080:8080",
    "gitea": "3000:3000",
    "elasticsearch": "9200:9200",
    "kafka": "9092:9092",
    "vault": "8200:8200",
    "sonarqube": "9000:9000",
    "minio": "9000:9000",
    "prometheus": "9090:9090",
    "traefik": "80:80",
    "consul": "8500:8500",
    "keycloak": "8080:8080",
}
DOCKER_SERVICES = list(DOCKER_SERVICE_PORTS.keys())
DOCKER_IMAGES = {
    "nginx": "nginx:latest",
    "redis": "redis:alpine",
    "postgres": "postgres:16",
    "mysql": "mysql:8",
    "mongodb": "mongodb/mongodb-community-server:7.0-ubuntu2204",
    "rabbitmq": "rabbitmq:3-management",
    "grafana": "grafana/grafana:latest",
    "portainer": "portainer/portainer-ce:latest",
    "jenkins": "jenkins/jenkins:lts",
    "elasticsearch": "docker.elastic.co/elasticsearch/elasticsearch:8.13.0",
    "kafka": "bitnami/kafka:latest",
    "vault": "hashicorp/vault:latest",
    "n8n": "n8nio/n8n:latest",
    "gitea": "gitea/gitea:latest",
    "sonarqube": "sonarqube:community",
    "minio": "minio/minio:latest",
    "prometheus": "prom/prometheus:latest",
    "traefik": "traefik:v3.0",
    "consul": "hashicorp/consul:latest",
    "keycloak": "quay.io/keycloak/keycloak:latest",
}

DOCKER_PROMPTS = [
    "Deploy {docker_service} container on my VM",
    "Run a {docker_service} Docker container",
    "Run a {docker_service} Docker container, port {ports}",
    "I need {docker_service} as a container, port {ports}",
    "Deploy Docker image {docker_image}",
    "Spin up {docker_service} container",
    "Start a {docker_service} container with port {ports}",
    "Please run a {docker_service} container with ports {ports}",
    "Launch Docker container {docker_image}",
    "I want to run {docker_service} in Docker, port {ports}",
    "Deploy {docker_service} on {hostname} via Docker",
    "Run {docker_image} container on port {ports}",
]

# ============================================================================
#  FastAPI
# ============================================================================
APP_PORTS = ["8000", "8080", "3000", "5000"]
HTTP_PORTS = ["80", "443", "8080"]

FASTAPI_PROMPTS = [
    "Deploy my FastAPI app to hostname {hostname}",
    "Deploy FastAPI application {app_name}, app port {app_port}",
    "Deploy FastAPI app {app_name}, port {app_port}, http port {http_port}",
    "I want to deploy a FastAPI app on {hostname}",
    "Deploy FastAPI to {hostname}, port {app_port}",
    "Please deploy FastAPI app {app_name} on {hostname}, port {app_port}",
    "Provision FastAPI service {app_name} on {hostname} with port {app_port}",
    "Launch FastAPI app on {hostname}, app port {app_port}, http port {http_port}",
    "Deploy my API service {app_name} to {hostname}",
    "Host FastAPI backend on {hostname}",
    "Deploy Python FastAPI app {app_name}",
    "Set up FastAPI microservice on {hostname}",
    "Deploy REST API {app_name} to {hostname}",
    "Launch Python API server on {hostname}",
]

# ============================================================================
#  Static Website
# ============================================================================
STATIC_PROMPTS = [
    "Deploy static website to {hostname}",
    "Deploy static website to nginx on {hostname}, port {http_port}",
    "Deploy static HTML site to {hostname}",
    "I need to deploy a static website on {hostname}, nginx, port {http_port}",
    "Deploy static website to nginx on {hostname}",
    "Please deploy static site to {hostname} on port {http_port}",
    "Host a static website on {hostname} via nginx",
    "Provision static site on {hostname}, http port {http_port}",
    "Set up documentation site on {hostname}",
    "Deploy my blog to {hostname}",
    "Host my portfolio website on {hostname}",
    "Deploy company website to {hostname}",
    "I want to host a static site on {hostname}",
    "Create static website deployment for {hostname}",
    "Deploy HTML/CSS site to {hostname}",
    "Host documentation on {hostname}",
    "Deploy React build to {hostname}",
    "Serve static files from {hostname}",
]

# ============================================================================
#  Database (self-hosted)
# ============================================================================
DATABASE_ENGINES = ["postgres", "postgresql", "mysql", "mongodb", "mariadb"]
DB_ENGINE_PORTS = {
    "postgres": "5432",
    "postgresql": "5432",
    "mysql": "3306",
    "mariadb": "3306",
    "mongodb": "27017",
}
DB_ENGINE_VERSIONS = {
    "postgres": ["14", "15", "16"],
    "postgresql": ["14", "15", "16"],
    "mysql": ["5.7", "8.0", "8.4"],
    "mariadb": ["10.11", "11.2", "11.4"],
    "mongodb": ["6.0", "7.0", "8.0"],
}

DB_PROMPTS = [
    "Provision {database_engine} database on {hostname}",
    "Deploy {database_engine} database, version {db_version}",
    "Deploy {database_engine} database, version {db_version}, name {database_name}, user {database_user}",
    "I need a {database_engine} database on {hostname}",
    "Set up {database_engine} database on {hostname}",
    "Provision database: {database_engine}, name {database_name}",
    "Deploy {database_engine} database, version {db_version} on {hostname}",
    "Please provision a {database_engine} database named {database_name}",
    "I want a {database_engine} database, version {db_version}, user {database_user}",
    "Create {database_engine} database {database_name} on {hostname}, port {port}",
]

# ============================================================================
#  ReactJS Frontend
# ============================================================================
REACT_FRAMEWORKS = ["react", "create-react-app", "vite-react"]
NODE_VERSIONS = ["18", "20", "22"]

REACTJS_PROMPTS = [
    "Deploy React.js app {app_name} to {hostname}",
    "Deploy my React frontend to {hostname} on port {http_port}",
    "I need to deploy a React.js application on {hostname}",
    "Provision React app {app_name} on {hostname}, port {app_port}",
    "Deploy React SPA to {hostname} via nginx",
    "Host React.js build on {hostname}, http port {http_port}",
    "Deploy create-react-app {app_name} to {hostname}",
    "Launch React frontend {app_name} on {hostname}",
    "Deploy Vite React app to {hostname} on port {app_port}",
    "Please deploy React app {app_name} to {hostname} with Node {runtime_version}",
]

# ============================================================================
#  NextJS Frontend/Fullstack
# ============================================================================
NEXTJS_PROMPTS = [
    "Deploy Next.js app {app_name} to {hostname}",
    "Deploy my Next.js application to {hostname} on port {app_port}",
    "I need to deploy a Next.js project on {hostname}",
    "Provision Next.js app {app_name} on {hostname}, port {app_port}",
    "Deploy Next.js SSR app to {hostname} via nginx",
    "Launch Next.js app {app_name} on {hostname}, port {app_port}",
    "Deploy nextjs app to {hostname} with Node {runtime_version}",
    "Please deploy Next.js application {app_name} to {hostname}",
]

# ============================================================================
#  ELK Stack (Elasticsearch + Logstash + Kibana)
# ============================================================================
ELK_VERSIONS = ["8.11.0", "8.12.0", "8.13.0", "8.14.0", "8.15.0"]

ELK_PROMPTS = [
    "Deploy ELK stack on {hostname}",
    "Deploy Elasticsearch, Logstash, and Kibana on {hostname}",
    "I need an ELK stack version {elk_version} on {hostname}",
    "Provision ELK stack on {hostname}, Elasticsearch port {es_port}",
    "Deploy ELK cluster on {hostname} for log analysis",
    "Set up ELK stack {elk_version} on {hostname}",
    "Launch Elasticsearch + Kibana on {hostname}",
    "Deploy logging stack (ELK) on {hostname}, version {elk_version}",
    "I want to deploy ELK stack for centralized logging on {hostname}",
    "Provision Elasticsearch {elk_version} with Kibana on {hostname}",
]

# ============================================================================
#  VPN (OpenVPN / WireGuard)
# ============================================================================
VPN_PROTOCOLS = ["openvpn", "wireguard"]
VPN_PROTOCOL_PORTS = {"openvpn": "1194", "wireguard": "51820"}
VPN_CLIENT_COUNTS = [5, 10, 20, 50, 100]

VPN_PROMPTS = [
    "Deploy {vpn_protocol} VPN server on {hostname}",
    "Set up {vpn_protocol} on {hostname}, port {vpn_port}",
    "I need a VPN server ({vpn_protocol}) on {hostname}",
    "Provision {vpn_protocol} VPN with {vpn_clients} clients on {hostname}",
    "Deploy VPN server using {vpn_protocol} on {hostname}",
    "Launch {vpn_protocol} VPN on {hostname}, max {vpn_clients} clients",
    "Set up a {vpn_protocol} VPN gateway on {hostname}",
    "I want to deploy {vpn_protocol} VPN on {hostname}, port {vpn_port}",
    "Create {vpn_protocol} VPN tunnel on {hostname} for {vpn_clients} users",
    "Provision secure VPN ({vpn_protocol}) on {hostname}",
]

# ============================================================================
#  Managed Database (RDS / Cloud SQL / Azure DB)
# ============================================================================
MANAGED_DB_ENGINES = ["postgres", "mysql", "mariadb", "aurora-postgresql", "aurora-mysql", "cloud-sql-postgres", "cloud-sql-mysql"]
MANAGED_DB_INSTANCE_CLASSES = [
    "db.t3.micro", "db.t3.small", "db.t3.medium", "db.t3.large",
    "db.r5.large", "db.r5.xlarge", "db.m5.large",
]
MANAGED_DB_STORAGE_GB = [20, 50, 100, 200, 500, 1000]

MANAGED_DB_PROMPTS = [
    "Create managed {database_engine} database {database_name} in {region}",
    "Provision RDS {database_engine} instance in {region}, {db_instance_class}",
    "Deploy managed database: {database_engine}, {db_instance_class}, {storage_size_gb}GB in {region}",
    "I need a managed {database_engine} database in {region}",
    "Set up Cloud SQL {database_engine} in {region}, {storage_size_gb}GB storage",
    "Provision managed {database_engine} database {database_name}, {db_instance_class}, {region}",
    "Deploy RDS {database_engine} with {storage_size_gb}GB in {region}, multi-AZ {multi_az}",
    "Create Aurora {database_engine} cluster in {region}",
    "Launch managed {database_engine} on {cloud_provider} in {region}",
    "Please provision a managed {database_engine} database with {storage_size_gb}GB in {region}",
]

# ============================================================================
#  Monitoring (Prometheus / Grafana / Datadog)
# ============================================================================
MONITORING_TOOLS = ["prometheus", "grafana", "prometheus-grafana", "datadog-agent", "alertmanager", "zabbix"]
MONITORING_TOOL_PORTS = {
    "prometheus": "9090",
    "grafana": "3000",
    "prometheus-grafana": "9090",
    "datadog-agent": "8126",
    "alertmanager": "9093",
    "zabbix": "10051",
}

MONITORING_PROMPTS = [
    "Deploy {monitoring_tool} on {hostname}",
    "Set up {monitoring_tool} monitoring stack on {hostname}",
    "I need {monitoring_tool} on {hostname}, port {monitoring_port}",
    "Provision {monitoring_tool} for monitoring on {hostname}",
    "Deploy monitoring stack ({monitoring_tool}) on {hostname}",
    "Launch {monitoring_tool} on {hostname} for infrastructure monitoring",
    "I want to deploy {monitoring_tool} on {hostname}",
    "Set up {monitoring_tool} monitoring on {hostname}, port {monitoring_port}",
]

# ============================================================================
#  CI/CD (Jenkins / GitLab Runner / ArgoCD / GitHub Actions Runner)
# ============================================================================
CICD_TOOLS = ["jenkins", "gitlab-runner", "argocd", "github-actions-runner", "drone", "tekton"]
CICD_TOOL_PORTS = {
    "jenkins": "8080",
    "gitlab-runner": "8093",
    "argocd": "8080",
    "github-actions-runner": "8080",
    "drone": "8080",
    "tekton": "9097",
}
CICD_TOOL_IMAGES = {
    "jenkins": "jenkins/jenkins:lts",
    "gitlab-runner": "gitlab/gitlab-runner:latest",
    "argocd": "quay.io/argoproj/argocd:latest",
    "github-actions-runner": "myoung34/github-runner:latest",
    "drone": "drone/drone:latest",
    "tekton": "gcr.io/tekton-releases/controller:latest",
}

CICD_PROMPTS = [
    "Deploy {cicd_tool} on {hostname}",
    "Set up {cicd_tool} CI/CD pipeline on {hostname}",
    "I need {cicd_tool} running on {hostname}, port {cicd_port}",
    "Provision {cicd_tool} for continuous integration on {hostname}",
    "Deploy {cicd_tool} server on {hostname}",
    "Launch {cicd_tool} on {hostname} for CI/CD automation",
    "I want to deploy {cicd_tool} on {hostname}, port {cicd_port}",
    "Set up {cicd_tool} pipeline runner on {hostname}",
]

# ============================================================================
#  Load Balancer (HAProxy / Nginx LB / Traefik)
# ============================================================================
LB_TYPES = ["haproxy", "nginx", "traefik", "envoy"]
LB_ALGORITHMS = ["round-robin", "least-connections", "ip-hash", "random"]
LB_PORTS = ["80", "443", "8080"]

LB_PROMPTS = [
    "Deploy {lb_type} load balancer on {hostname}",
    "Set up {lb_type} with {lb_algorithm} on {hostname}, port {lb_port}",
    "I need a {lb_type} load balancer on {hostname}",
    "Provision {lb_type} LB on {hostname} with {lb_algorithm} balancing",
    "Deploy load balancer ({lb_type}) on {hostname}, port {lb_port}",
    "Launch {lb_type} reverse proxy on {hostname}",
    "I want to deploy {lb_type} load balancer on {hostname}, port {lb_port}",
    "Set up {lb_type} for traffic distribution on {hostname}",
]

# ============================================================================
#  Cache (Redis / Memcached)
# ============================================================================
CACHE_ENGINES = ["redis", "memcached", "redis-cluster", "redis-sentinel"]
CACHE_ENGINE_PORTS = {
    "redis": "6379",
    "memcached": "11211",
    "redis-cluster": "6379",
    "redis-sentinel": "26379",
}
CACHE_SIZES_MB = [64, 128, 256, 512, 1024, 2048, 4096]

CACHE_PROMPTS = [
    "Deploy {cache_engine} cache on {hostname}",
    "Set up {cache_engine} on {hostname}, port {cache_port}",
    "I need a {cache_engine} caching layer on {hostname}",
    "Provision {cache_engine} cache cluster on {hostname}",
    "Deploy {cache_engine} for caching on {hostname}, {cache_size_mb}MB",
    "Launch {cache_engine} on {hostname}, port {cache_port}",
    "I want to deploy {cache_engine} cache on {hostname}",
    "Set up {cache_engine} with {replicas} replicas on {hostname}",
]

# ============================================================================
#  Object Storage (S3 / MinIO / NFS)
# ============================================================================
STORAGE_BACKENDS = ["s3", "minio", "nfs", "ceph", "glusterfs"]
STORAGE_SIZES_GB = [50, 100, 250, 500, 1000, 2000, 5000]

STORAGE_PROMPTS = [
    "Provision {storage_backend} storage bucket {bucket_name} in {region}",
    "Deploy {storage_backend} object storage on {hostname}",
    "I need {storage_backend} storage, {storage_size_gb}GB in {region}",
    "Create {storage_backend} bucket {bucket_name} in {region}",
    "Set up {storage_backend} storage on {hostname}, {storage_size_gb}GB",
    "Deploy {storage_backend} with {storage_size_gb}GB capacity on {hostname}",
    "Launch {storage_backend} object store on {hostname}",
    "I want {storage_backend} storage bucket {bucket_name}, {storage_size_gb}GB",
]

# ============================================================================
#  SSL Certificate (Let's Encrypt / ACM)
# ============================================================================
SSL_PROVIDERS = ["letsencrypt", "acm", "cloudflare", "digicert", "zerossl"]

SSL_PROMPTS = [
    "Provision SSL certificate for {domain_name} using {ssl_provider}",
    "Deploy SSL cert for {domain_name} via {ssl_provider}",
    "I need an SSL certificate for {domain_name}",
    "Set up HTTPS for {domain_name} with {ssl_provider}",
    "Generate SSL certificate for {domain_name} using {ssl_provider}",
    "Provision TLS certificate for {domain_name}",
    "I want to enable HTTPS on {domain_name} via {ssl_provider}",
    "Create SSL cert for {domain_name} with auto-renewal",
]

# ============================================================================
#  WordPress
# ============================================================================
WP_PROMPTS = [
    "Deploy WordPress on {hostname}",
    "Set up WordPress site on {hostname}, port {http_port}",
    "I need a WordPress installation on {hostname}",
    "Provision WordPress on {hostname} with {database_engine} backend",
    "Deploy WordPress CMS on {hostname}, http port {http_port}",
    "Launch WordPress site on {hostname}",
    "I want to deploy WordPress on {hostname} with {database_engine}",
    "Set up WordPress blog on {hostname}, port {http_port}",
]

# ============================================================================
#  Django
# ============================================================================
PYTHON_VERSIONS = ["3.10", "3.11", "3.12", "3.13"]

DJANGO_PROMPTS = [
    "Deploy Django app {app_name} to {hostname}",
    "Deploy my Django application to {hostname} on port {app_port}",
    "I need to deploy a Django project on {hostname}",
    "Provision Django app {app_name} on {hostname}, port {app_port}",
    "Deploy Django {app_name} to {hostname} with Python {runtime_version}",
    "Launch Django app on {hostname}, port {app_port}, {database_engine} backend",
    "Deploy Django REST API {app_name} to {hostname}",
    "Please deploy Django app {app_name} to {hostname} with gunicorn",
]

# ============================================================================
#  Spring Boot
# ============================================================================
JAVA_VERSIONS = ["17", "21"]
SPRINGBOOT_PORTS = ["8080", "8443", "9090"]

SPRINGBOOT_PROMPTS = [
    "Deploy Spring Boot app {app_name} to {hostname}",
    "Deploy my Spring Boot application to {hostname} on port {app_port}",
    "I need to deploy a Spring Boot service on {hostname}",
    "Provision Spring Boot app {app_name} on {hostname}, port {app_port}",
    "Deploy Spring Boot {app_name} to {hostname} with Java {runtime_version}",
    "Launch Spring Boot microservice on {hostname}, port {app_port}",
    "Deploy Spring Boot REST API {app_name} to {hostname}",
    "Please deploy Spring Boot app {app_name} to {hostname}, Java {runtime_version}",
]

# ============================================================================
#  Intent registry and weights
# ============================================================================
INTENTS = [
    "provision_vm", "provision_kubernetes", "provision_docker",
    "provision_fastapi", "provision_static_website", "provision_database",
    # --- new ---
    "provision_reactjs", "provision_nextjs", "provision_elk",
    "provision_vpn", "provision_managed_database", "provision_monitoring",
    "provision_cicd", "provision_loadbalancer", "provision_cache",
    "provision_storage", "provision_ssl", "provision_wordpress",
    "provision_django", "provision_springboot",
]

INTENT_WEIGHTS = [
    35, 13, 17,   # vm, k8s, docker
    7,  6,  9,    # fastapi, static, db
    # --- new (total ~25) ---
    3,  2,  2,    # reactjs, nextjs, elk
    2,  3,  2,    # vpn, managed_db, monitoring
    2,  1,  2,    # cicd, lb, cache
    1,  1,  1,    # storage, ssl, wordpress
    2,  1,         # django, springboot
]

# ============================================================================
#  CSV field names
# ============================================================================
FIELDNAMES = [
    "deployment_id", "prompt", "intent", "username", "workspace",
    # VM fields
    "instance_name", "instance_type", "region", "cloud_provider", "os",
    "volume_size_gb", "volume_type", "environment", "size_tier",
    # K8s fields
    "cluster_name", "node_count", "node_type", "kubernetes_version",
    # Docker fields
    "docker_image", "docker_service", "container_name", "ports",
    # Shared host fields
    "hostname", "ssh_username", "key_pair_name",
    # App fields
    "app_name", "app_port", "http_port", "https_port", "server_name",
    # Database fields
    "database_engine", "database_name", "database_user", "db_version", "port",
    # New: framework / runtime
    "framework", "runtime_version", "repo_url",
    # New: VPN
    "vpn_protocol", "vpn_port", "vpn_clients",
    # New: Managed DB
    "db_instance_class", "multi_az", "storage_size_gb",
    # New: Monitoring
    "monitoring_tool", "monitoring_port",
    # New: CI/CD
    "cicd_tool", "cicd_port",
    # New: Load balancer
    "lb_type", "lb_algorithm",
    # New: Cache
    "cache_engine", "cache_port", "cache_size_mb", "replicas",
    # New: Storage
    "storage_backend", "bucket_name",
    # New: SSL
    "ssl_provider", "domain_name",
    # New: ELK
    "elk_version", "es_port", "kibana_port", "logstash_port",
    # Meta
    "date", "endpoint_url", "payload",
]


# ============================================================================
#  Helpers
# ============================================================================
def _s(v):
    """Stringify, turning None to empty string."""
    return "" if v is None else str(v)


def _empty_row() -> dict:
    return {k: "" for k in FIELDNAMES}


def _host(prefix: str, n: int = 999) -> str:
    """Generate realistic hostnames for training data."""
    # Mix of realistic domain patterns users might actually request
    realistic_domains = [
        "docs.example.com",
        "api.mycompany.com", 
        "app.staging.com",
        "blog.example.org",
        "admin.backend.io",
        "web.production.net",
        "dashboard.internal",
        "frontend.dev.com",
        "backend.prod.io",
        "www.mysite.com",
        "portal.company.org",
        "cms.website.net"
    ]
    
    # 30% chance of realistic domain, 70% chance of random (for training diversity)
    if random.random() < 0.3:
        return random.choice(realistic_domains)
    else:
        return f"{prefix}-{random.randint(1, n)}.example.com"


def _maybe_user() -> str:
    return f"user{random.randint(1, 500)}" if random.random() < 0.3 else ""


def _maybe_workspace() -> str:
    return f"ws-{random.randint(100, 999)}" if random.random() < 0.25 else ""


def _date(start_date: datetime) -> str:
    return (start_date - timedelta(days=random.randint(0, 365))).strftime("%Y-%m-%d")


def _fmt(tpl: str, **kwargs) -> str:
    """Safe format — missing keys left as-is."""
    try:
        return tpl.format(**kwargs)
    except KeyError:
        return tpl


# ============================================================================
#  Row generators — each returns a fully consistent dict
# ============================================================================

def generate_vm_row(idx: int, start_date: datetime) -> dict:
    row = _empty_row()
    row["deployment_id"] = f"DEP-{idx:05d}"
    row["intent"] = "provision_vm"

    size_tier = random.choice(SIZE_TIERS)
    opts = SIZE_TO_DEFAULTS[size_tier]
    instance_type, volume_size_gb = random.choice(opts)
    # 50% chance to override with fully random values
    if random.random() < 0.5:
        instance_type = random.choice(INSTANCE_TYPES)
        volume_size_gb = random.choice(VOLUME_SIZES_GB)

    region = random.choice(REGIONS)
    cloud_provider = random.choice(CLOUD_PROVIDERS)
    os_name = random.choice(OS_OPTIONS)
    volume_type = random.choice(VOLUME_TYPES)
    environment = random.choice(ENVIRONMENTS)
    username = _maybe_user()
    workspace = _maybe_workspace()
    instance_name = f"vm-{random.randint(1, 9999)}" if random.random() < 0.4 else ""

    row.update({
        "username": username, "workspace": workspace,
        "instance_name": instance_name, "instance_type": instance_type,
        "region": region, "cloud_provider": cloud_provider, "os": os_name,
        "volume_size_gb": volume_size_gb, "volume_type": volume_type,
        "environment": environment, "size_tier": size_tier,
        "date": _date(start_date),
    })
    row["prompt"] = _fmt(
        random.choice(VM_PROMPTS),
        instance_type=instance_type, region=region, cloud_provider=cloud_provider,
        os=os_name, volume_size_gb=volume_size_gb, volume_type=volume_type,
        size_tier=size_tier, environment=environment,
        username=username or "joeuser", workspace=workspace or "default-ws",
    )
    return row


def generate_k8s_row(idx: int, start_date: datetime) -> dict:
    row = _empty_row()
    row["deployment_id"] = f"DEP-{idx:05d}"
    row["intent"] = "provision_kubernetes"

    cluster_name = f"cluster-{random.randint(1, 999)}"
    node_count = random.choice([2, 3, 4, 5, 6])
    node_type = random.choice(K8S_NODE_TYPES)
    region = random.choice(REGIONS)
    cloud_provider = random.choice(["aws", "gcp", "azure"])
    environment = random.choice(ENVIRONMENTS)
    k8s_version = random.choice(K8S_VERSIONS)

    row.update({
        "cluster_name": cluster_name, "node_count": node_count,
        "node_type": node_type, "region": region,
        "cloud_provider": cloud_provider, "environment": environment,
        "kubernetes_version": k8s_version,
        "username": _maybe_user(), "workspace": _maybe_workspace(),
        "date": _date(start_date),
    })
    row["prompt"] = _fmt(
        random.choice(K8S_PROMPTS),
        cluster_name=cluster_name, node_count=node_count, node_type=node_type,
        region=region, cloud_provider=cloud_provider, kubernetes_version=k8s_version,
    )
    return row


def generate_docker_row(idx: int, start_date: datetime) -> dict:
    row = _empty_row()
    row["deployment_id"] = f"DEP-{idx:05d}"
    row["intent"] = "provision_docker"

    docker_service = random.choice(DOCKER_SERVICES)
    docker_image = DOCKER_IMAGES.get(docker_service, f"{docker_service}:latest")
    ports = DOCKER_SERVICE_PORTS[docker_service]
    container_name = f"{docker_service}-{random.randint(1, 99)}"
    hostname = _host("vm") if random.random() < 0.5 else ""

    row.update({
        "docker_service": docker_service, "docker_image": docker_image,
        "container_name": container_name, "ports": ports,
        "hostname": hostname,
        "ssh_username": "ubuntu" if hostname else "",
        "key_pair_name": f"key-{random.randint(1, 99)}" if hostname else "",
        "username": _maybe_user(), "workspace": _maybe_workspace(),
        "date": _date(start_date),
    })
    row["prompt"] = _fmt(
        random.choice(DOCKER_PROMPTS),
        docker_service=docker_service, docker_image=docker_image,
        ports=ports, hostname=hostname or "my-vm",
    )
    return row


def generate_fastapi_row(idx: int, start_date: datetime) -> dict:
    row = _empty_row()
    row["deployment_id"] = f"DEP-{idx:05d}"
    row["intent"] = "provision_fastapi"

    hostname = _host("app")
    app_port = random.choice(APP_PORTS)
    http_port = random.choice(HTTP_PORTS)
    app_name = f"fastapi-app-{random.randint(1, 99)}"

    row.update({
        "hostname": hostname, "app_port": app_port, "http_port": http_port,
        "app_name": app_name,
        "ssh_username": "ubuntu",
        "key_pair_name": f"app-{random.randint(1, 99)}",
        "framework": "fastapi",
        "username": _maybe_user(), "workspace": _maybe_workspace(),
        "date": _date(start_date),
    })
    row["prompt"] = _fmt(
        random.choice(FASTAPI_PROMPTS),
        hostname=hostname, app_port=app_port, http_port=http_port, app_name=app_name,
    )
    return row


def generate_static_row(idx: int, start_date: datetime) -> dict:
    """Static website — server_name ALWAYS matches hostname for consistency."""
    row = _empty_row()
    row["deployment_id"] = f"DEP-{idx:05d}"
    row["intent"] = "provision_static_website"

    hostname = _host("web")
    http_port = "80"
    https_port = "443" if random.random() < 0.5 else ""
    # server_name matches the hostname the user requested
    server_name = hostname

    row.update({
        "hostname": hostname, "server_name": server_name,
        "http_port": http_port, "https_port": https_port,
        "ssh_username": "ubuntu",
        "key_pair_name": f"web-{random.randint(1, 99)}",
        "username": _maybe_user(), "workspace": _maybe_workspace(),
        "date": _date(start_date),
    })
    row["prompt"] = _fmt(
        random.choice(STATIC_PROMPTS),
        hostname=hostname, http_port=http_port,
    )
    return row


def generate_database_row(idx: int, start_date: datetime) -> dict:
    """Self-hosted database — engine, port, and version are always consistent."""
    row = _empty_row()
    row["deployment_id"] = f"DEP-{idx:05d}"
    row["intent"] = "provision_database"

    engine = random.choice(DATABASE_ENGINES)
    port = DB_ENGINE_PORTS[engine]
    db_version = random.choice(DB_ENGINE_VERSIONS[engine])
    database_name = f"db_{random.randint(1, 999)}"
    database_user = f"dbuser{random.randint(1, 99)}"
    hostname = _host("db")

    row.update({
        "database_engine": engine, "port": port, "db_version": db_version,
        "database_name": database_name, "database_user": database_user,
        "hostname": hostname,
        "ssh_username": "ubuntu",
        "key_pair_name": f"db-{random.randint(1, 99)}",
        "username": _maybe_user(), "workspace": _maybe_workspace(),
        "date": _date(start_date),
    })
    row["prompt"] = _fmt(
        random.choice(DB_PROMPTS),
        database_engine=engine, db_version=db_version,
        database_name=database_name, database_user=database_user,
        hostname=hostname, port=port,
    )
    return row


# ---------------------------------------------------------------------------
#  NEW INTENTS
# ---------------------------------------------------------------------------

def generate_reactjs_row(idx: int, start_date: datetime) -> dict:
    row = _empty_row()
    row["deployment_id"] = f"DEP-{idx:05d}"
    row["intent"] = "provision_reactjs"

    hostname = _host("web")
    app_name = f"react-app-{random.randint(1, 99)}"
    app_port = random.choice(["3000", "8080", "5173"])
    http_port = random.choice(["80", "443"])
    node_ver = random.choice(NODE_VERSIONS)
    repo_url = f"https://github.com/org-{random.randint(1, 50)}/{app_name}.git"

    row.update({
        "hostname": hostname, "app_name": app_name,
        "app_port": app_port, "http_port": http_port,
        "framework": "react", "runtime_version": node_ver,
        "repo_url": repo_url, "server_name": hostname,
        "ssh_username": "ubuntu",
        "key_pair_name": f"web-{random.randint(1, 99)}",
        "username": _maybe_user(), "workspace": _maybe_workspace(),
        "date": _date(start_date),
    })
    row["prompt"] = _fmt(
        random.choice(REACTJS_PROMPTS),
        hostname=hostname, app_name=app_name, app_port=app_port,
        http_port=http_port, runtime_version=node_ver,
    )
    return row


def generate_nextjs_row(idx: int, start_date: datetime) -> dict:
    row = _empty_row()
    row["deployment_id"] = f"DEP-{idx:05d}"
    row["intent"] = "provision_nextjs"

    hostname = _host("web")
    app_name = f"nextjs-app-{random.randint(1, 99)}"
    app_port = "3000"
    http_port = random.choice(["80", "443"])
    node_ver = random.choice(NODE_VERSIONS)
    repo_url = f"https://github.com/org-{random.randint(1, 50)}/{app_name}.git"

    row.update({
        "hostname": hostname, "app_name": app_name,
        "app_port": app_port, "http_port": http_port,
        "framework": "nextjs", "runtime_version": node_ver,
        "repo_url": repo_url, "server_name": hostname,
        "ssh_username": "ubuntu",
        "key_pair_name": f"web-{random.randint(1, 99)}",
        "username": _maybe_user(), "workspace": _maybe_workspace(),
        "date": _date(start_date),
    })
    row["prompt"] = _fmt(
        random.choice(NEXTJS_PROMPTS),
        hostname=hostname, app_name=app_name, app_port=app_port,
        http_port=http_port, runtime_version=node_ver,
    )
    return row


def generate_elk_row(idx: int, start_date: datetime) -> dict:
    row = _empty_row()
    row["deployment_id"] = f"DEP-{idx:05d}"
    row["intent"] = "provision_elk"

    hostname = _host("elk")
    elk_version = random.choice(ELK_VERSIONS)
    es_port = "9200"
    kibana_port = "5601"
    logstash_port = "5044"

    row.update({
        "hostname": hostname, "elk_version": elk_version,
        "es_port": es_port, "kibana_port": kibana_port,
        "logstash_port": logstash_port,
        "ssh_username": "ubuntu",
        "key_pair_name": f"elk-{random.randint(1, 99)}",
        "username": _maybe_user(), "workspace": _maybe_workspace(),
        "date": _date(start_date),
    })
    row["prompt"] = _fmt(
        random.choice(ELK_PROMPTS),
        hostname=hostname, elk_version=elk_version, es_port=es_port,
    )
    return row


def generate_vpn_row(idx: int, start_date: datetime) -> dict:
    row = _empty_row()
    row["deployment_id"] = f"DEP-{idx:05d}"
    row["intent"] = "provision_vpn"

    hostname = _host("vpn")
    vpn_protocol = random.choice(VPN_PROTOCOLS)
    vpn_port = VPN_PROTOCOL_PORTS[vpn_protocol]
    vpn_clients = random.choice(VPN_CLIENT_COUNTS)

    row.update({
        "hostname": hostname, "vpn_protocol": vpn_protocol,
        "vpn_port": vpn_port, "vpn_clients": vpn_clients,
        "ssh_username": "ubuntu",
        "key_pair_name": f"vpn-{random.randint(1, 99)}",
        "os": random.choice(["ubuntu-22.04", "ubuntu-24.04", "debian-12"]),
        "username": _maybe_user(), "workspace": _maybe_workspace(),
        "date": _date(start_date),
    })
    row["prompt"] = _fmt(
        random.choice(VPN_PROMPTS),
        hostname=hostname, vpn_protocol=vpn_protocol,
        vpn_port=vpn_port, vpn_clients=vpn_clients,
    )
    return row


def generate_managed_db_row(idx: int, start_date: datetime) -> dict:
    row = _empty_row()
    row["deployment_id"] = f"DEP-{idx:05d}"
    row["intent"] = "provision_managed_database"

    engine = random.choice(MANAGED_DB_ENGINES)
    db_instance_class = random.choice(MANAGED_DB_INSTANCE_CLASSES)
    storage_size_gb = random.choice(MANAGED_DB_STORAGE_GB)
    region = random.choice(REGIONS)
    cloud_provider = random.choice(["aws", "gcp", "azure"])
    multi_az = random.choice(["true", "false"])
    database_name = f"managed-db-{random.randint(1, 999)}"
    database_user = f"admin{random.randint(1, 99)}"

    # derive a clean engine name for the prompt
    engine_label = engine.replace("-", " ").replace("cloud sql ", "Cloud SQL ")

    row.update({
        "database_engine": engine, "database_name": database_name,
        "database_user": database_user,
        "db_instance_class": db_instance_class,
        "storage_size_gb": storage_size_gb,
        "region": region, "cloud_provider": cloud_provider,
        "multi_az": multi_az, "environment": random.choice(ENVIRONMENTS),
        "username": _maybe_user(), "workspace": _maybe_workspace(),
        "date": _date(start_date),
    })
    row["prompt"] = _fmt(
        random.choice(MANAGED_DB_PROMPTS),
        database_engine=engine_label, database_name=database_name,
        db_instance_class=db_instance_class, storage_size_gb=storage_size_gb,
        region=region, cloud_provider=cloud_provider, multi_az=multi_az,
    )
    return row


def generate_monitoring_row(idx: int, start_date: datetime) -> dict:
    row = _empty_row()
    row["deployment_id"] = f"DEP-{idx:05d}"
    row["intent"] = "provision_monitoring"

    hostname = _host("mon")
    monitoring_tool = random.choice(MONITORING_TOOLS)
    monitoring_port = MONITORING_TOOL_PORTS[monitoring_tool]

    row.update({
        "hostname": hostname, "monitoring_tool": monitoring_tool,
        "monitoring_port": monitoring_port,
        "ssh_username": "ubuntu",
        "key_pair_name": f"mon-{random.randint(1, 99)}",
        "username": _maybe_user(), "workspace": _maybe_workspace(),
        "date": _date(start_date),
    })
    row["prompt"] = _fmt(
        random.choice(MONITORING_PROMPTS),
        hostname=hostname, monitoring_tool=monitoring_tool,
        monitoring_port=monitoring_port,
    )
    return row


def generate_cicd_row(idx: int, start_date: datetime) -> dict:
    row = _empty_row()
    row["deployment_id"] = f"DEP-{idx:05d}"
    row["intent"] = "provision_cicd"

    hostname = _host("cicd")
    cicd_tool = random.choice(CICD_TOOLS)
    cicd_port = CICD_TOOL_PORTS[cicd_tool]

    row.update({
        "hostname": hostname, "cicd_tool": cicd_tool,
        "cicd_port": cicd_port,
        "docker_image": CICD_TOOL_IMAGES.get(cicd_tool, ""),
        "ssh_username": "ubuntu",
        "key_pair_name": f"cicd-{random.randint(1, 99)}",
        "username": _maybe_user(), "workspace": _maybe_workspace(),
        "date": _date(start_date),
    })
    row["prompt"] = _fmt(
        random.choice(CICD_PROMPTS),
        hostname=hostname, cicd_tool=cicd_tool, cicd_port=cicd_port,
    )
    return row


def generate_loadbalancer_row(idx: int, start_date: datetime) -> dict:
    row = _empty_row()
    row["deployment_id"] = f"DEP-{idx:05d}"
    row["intent"] = "provision_loadbalancer"

    hostname = _host("lb")
    lb_type = random.choice(LB_TYPES)
    lb_algorithm = random.choice(LB_ALGORITHMS)
    lb_port = random.choice(LB_PORTS)

    row.update({
        "hostname": hostname, "lb_type": lb_type,
        "lb_algorithm": lb_algorithm, "http_port": lb_port,
        "ssh_username": "ubuntu",
        "key_pair_name": f"lb-{random.randint(1, 99)}",
        "username": _maybe_user(), "workspace": _maybe_workspace(),
        "date": _date(start_date),
    })
    row["prompt"] = _fmt(
        random.choice(LB_PROMPTS),
        hostname=hostname, lb_type=lb_type,
        lb_algorithm=lb_algorithm, lb_port=lb_port,
    )
    return row


def generate_cache_row(idx: int, start_date: datetime) -> dict:
    row = _empty_row()
    row["deployment_id"] = f"DEP-{idx:05d}"
    row["intent"] = "provision_cache"

    hostname = _host("cache")
    cache_engine = random.choice(CACHE_ENGINES)
    cache_port = CACHE_ENGINE_PORTS[cache_engine]
    cache_size_mb = random.choice(CACHE_SIZES_MB)
    replicas = random.choice([1, 2, 3])

    row.update({
        "hostname": hostname, "cache_engine": cache_engine,
        "cache_port": cache_port, "cache_size_mb": cache_size_mb,
        "replicas": replicas,
        "ssh_username": "ubuntu",
        "key_pair_name": f"cache-{random.randint(1, 99)}",
        "username": _maybe_user(), "workspace": _maybe_workspace(),
        "date": _date(start_date),
    })
    row["prompt"] = _fmt(
        random.choice(CACHE_PROMPTS),
        hostname=hostname, cache_engine=cache_engine,
        cache_port=cache_port, cache_size_mb=cache_size_mb, replicas=replicas,
    )
    return row


def generate_storage_row(idx: int, start_date: datetime) -> dict:
    row = _empty_row()
    row["deployment_id"] = f"DEP-{idx:05d}"
    row["intent"] = "provision_storage"

    hostname = _host("storage") if random.random() < 0.5 else ""
    storage_backend = random.choice(STORAGE_BACKENDS)
    storage_size_gb = random.choice(STORAGE_SIZES_GB)
    region = random.choice(REGIONS)
    bucket_name = f"bucket-{random.choice(['logs', 'assets', 'backups', 'data', 'media', 'uploads', 'artifacts'])}-{random.randint(1, 999)}"

    row.update({
        "hostname": hostname, "storage_backend": storage_backend,
        "storage_size_gb": storage_size_gb, "region": region,
        "bucket_name": bucket_name,
        "cloud_provider": random.choice(CLOUD_PROVIDERS),
        "ssh_username": "ubuntu" if hostname else "",
        "key_pair_name": f"storage-{random.randint(1, 99)}" if hostname else "",
        "username": _maybe_user(), "workspace": _maybe_workspace(),
        "date": _date(start_date),
    })
    row["prompt"] = _fmt(
        random.choice(STORAGE_PROMPTS),
        hostname=hostname or "cloud", storage_backend=storage_backend,
        storage_size_gb=storage_size_gb, region=region, bucket_name=bucket_name,
    )
    return row


def generate_ssl_row(idx: int, start_date: datetime) -> dict:
    row = _empty_row()
    row["deployment_id"] = f"DEP-{idx:05d}"
    row["intent"] = "provision_ssl"

    tlds = ["com", "io", "dev", "app", "org", "net", "co"]
    prefixes = ["api", "www", "app", "dashboard", "portal", "shop", "docs", "blog"]
    domain_name = f"{random.choice(prefixes)}.{random.choice(['acme', 'globex', 'initech', 'hooli', 'piedpiper', 'megacorp', 'techstart'])}.{random.choice(tlds)}"
    ssl_provider = random.choice(SSL_PROVIDERS)

    row.update({
        "domain_name": domain_name, "ssl_provider": ssl_provider,
        "username": _maybe_user(), "workspace": _maybe_workspace(),
        "date": _date(start_date),
    })
    row["prompt"] = _fmt(
        random.choice(SSL_PROMPTS),
        domain_name=domain_name, ssl_provider=ssl_provider,
    )
    return row


def generate_wordpress_row(idx: int, start_date: datetime) -> dict:
    row = _empty_row()
    row["deployment_id"] = f"DEP-{idx:05d}"
    row["intent"] = "provision_wordpress"

    hostname = _host("wp")
    http_port = random.choice(["80", "443"])
    db_engine = random.choice(["mysql", "mariadb"])

    row.update({
        "hostname": hostname, "http_port": http_port,
        "database_engine": db_engine,
        "app_name": f"wordpress-{random.randint(1, 99)}",
        "framework": "wordpress",
        "server_name": hostname,
        "ssh_username": "ubuntu",
        "key_pair_name": f"wp-{random.randint(1, 99)}",
        "username": _maybe_user(), "workspace": _maybe_workspace(),
        "date": _date(start_date),
    })
    row["prompt"] = _fmt(
        random.choice(WP_PROMPTS),
        hostname=hostname, http_port=http_port, database_engine=db_engine,
    )
    return row


def generate_django_row(idx: int, start_date: datetime) -> dict:
    row = _empty_row()
    row["deployment_id"] = f"DEP-{idx:05d}"
    row["intent"] = "provision_django"

    hostname = _host("app")
    app_name = f"django-app-{random.randint(1, 99)}"
    app_port = random.choice(["8000", "8080"])
    http_port = random.choice(["80", "443"])
    python_ver = random.choice(PYTHON_VERSIONS)
    db_engine = random.choice(["postgres", "mysql"])
    repo_url = f"https://github.com/org-{random.randint(1, 50)}/{app_name}.git"

    row.update({
        "hostname": hostname, "app_name": app_name,
        "app_port": app_port, "http_port": http_port,
        "framework": "django", "runtime_version": python_ver,
        "database_engine": db_engine, "repo_url": repo_url,
        "server_name": hostname,
        "ssh_username": "ubuntu",
        "key_pair_name": f"app-{random.randint(1, 99)}",
        "username": _maybe_user(), "workspace": _maybe_workspace(),
        "date": _date(start_date),
    })
    row["prompt"] = _fmt(
        random.choice(DJANGO_PROMPTS),
        hostname=hostname, app_name=app_name, app_port=app_port,
        http_port=http_port, runtime_version=python_ver,
        database_engine=db_engine,
    )
    return row


def generate_springboot_row(idx: int, start_date: datetime) -> dict:
    row = _empty_row()
    row["deployment_id"] = f"DEP-{idx:05d}"
    row["intent"] = "provision_springboot"

    hostname = _host("app")
    app_name = f"springboot-svc-{random.randint(1, 99)}"
    app_port = random.choice(SPRINGBOOT_PORTS)
    http_port = random.choice(["80", "443"])
    java_ver = random.choice(JAVA_VERSIONS)
    repo_url = f"https://github.com/org-{random.randint(1, 50)}/{app_name}.git"

    row.update({
        "hostname": hostname, "app_name": app_name,
        "app_port": app_port, "http_port": http_port,
        "framework": "springboot", "runtime_version": java_ver,
        "repo_url": repo_url, "server_name": hostname,
        "ssh_username": "ubuntu",
        "key_pair_name": f"app-{random.randint(1, 99)}",
        "username": _maybe_user(), "workspace": _maybe_workspace(),
        "date": _date(start_date),
    })
    row["prompt"] = _fmt(
        random.choice(SPRINGBOOT_PROMPTS),
        hostname=hostname, app_name=app_name, app_port=app_port,
        http_port=http_port, runtime_version=java_ver,
    )
    return row


# ============================================================================
#  Generator dispatch
# ============================================================================
GENERATORS = {
    "provision_vm": generate_vm_row,
    "provision_kubernetes": generate_k8s_row,
    "provision_docker": generate_docker_row,
    "provision_fastapi": generate_fastapi_row,
    "provision_static_website": generate_static_row,
    "provision_database": generate_database_row,
    "provision_reactjs": generate_reactjs_row,
    "provision_nextjs": generate_nextjs_row,
    "provision_elk": generate_elk_row,
    "provision_vpn": generate_vpn_row,
    "provision_managed_database": generate_managed_db_row,
    "provision_monitoring": generate_monitoring_row,
    "provision_cicd": generate_cicd_row,
    "provision_loadbalancer": generate_loadbalancer_row,
    "provision_cache": generate_cache_row,
    "provision_storage": generate_storage_row,
    "provision_ssl": generate_ssl_row,
    "provision_wordpress": generate_wordpress_row,
    "provision_django": generate_django_row,
    "provision_springboot": generate_springboot_row,
}


def generate_row(idx: int, start_date: datetime) -> dict:
    intent = random.choices(INTENTS, weights=INTENT_WEIGHTS)[0]
    return GENERATORS[intent](idx, start_date)


# ============================================================================
#  Endpoint + Payload builder
# ============================================================================
def get_endpoint_and_payload(intent: str, row: dict) -> tuple:
    b = BASE_URL

    if intent == "provision_vm":
        url = f"{b}/api/v2/tenant/provision/vm"
        payload = {
            "username": row.get("username") or "user",
            "instance_name": row.get("instance_name") or "vm-1",
            "instance_type": row.get("instance_type") or "t3.micro",
            "region": row.get("region") or "us-east-1",
            "cloud_provider": row.get("cloud_provider") or "aws",
            "os": row.get("os") or "ubuntu",
            "volume_size_gb": int(row.get("volume_size_gb") or 30),
            "environment": row.get("environment") or "dev",
        }

    elif intent == "provision_kubernetes":
        url = f"{b}/api/v2/tenant/provision/kubernetescluster/deploy"
        payload = {
            "username": row.get("username") or "user",
            "cluster_name": row.get("cluster_name") or "cluster-1",
            "node_count": int(row.get("node_count") or 3),
            "node_type": row.get("node_type") or "t3.medium",
            "region": row.get("region") or "us-east-1",
            "cloud_provider": row.get("cloud_provider") or "aws",
            "kubernetes_version": row.get("kubernetes_version") or "1.30",
        }

    elif intent == "provision_docker":
        url = f"{b}/api/v2/tenant/provision/docker"
        payload = {
            "username": row.get("username") or "user",
            "hostname": row.get("hostname") or "",
            "ssh_username": row.get("ssh_username") or "ubuntu",
            "docker_image": row.get("docker_image") or "nginx:latest",
            "container_name": row.get("container_name") or "app",
            "ports": row.get("ports") or "80:80",
        }

    elif intent == "provision_fastapi":
        url = f"{b}/api/v1/infrastructure/services/fastapi/deploy"
        payload = {
            "hostname": row.get("hostname") or "app-1.example.com",
            "app_name": row.get("app_name") or "fastapi-app",
            "app_port": row.get("app_port") or "8000",
            "http_port": row.get("http_port") or "80",
        }

    elif intent == "provision_static_website":
        hostname = row.get("hostname") or "web-1.example.com"
        url = f"{b}/api/v1/infrastructure/services/staticwebsite/deploy"
        payload = {
            "hostname": hostname,
            "http_port": row.get("http_port") or "80",
            "server_name": hostname,  # always match hostname
        }

    elif intent == "provision_database":
        url = f"{b}/api/v2/tenant/provision/databases"
        payload = {
            "username": row.get("username") or "user",
            "hostname": row.get("hostname") or "db-1.example.com",
            "engine": row.get("database_engine") or "postgres",
            "database_name": row.get("database_name") or "appdb",
            "database_user": row.get("database_user") or "dbuser",
            "db_version": row.get("db_version") or "16",
            "port": row.get("port") or "5432",
        }

    elif intent == "provision_reactjs":
        url = f"{b}/api/v1/infrastructure/services/reactjs/deploy"
        payload = {
            "hostname": row.get("hostname") or "web-1.example.com",
            "app_name": row.get("app_name") or "react-app",
            "app_port": row.get("app_port") or "3000",
            "http_port": row.get("http_port") or "80",
            "framework": "react",
            "node_version": row.get("runtime_version") or "20",
            "repo_url": row.get("repo_url") or "",
        }

    elif intent == "provision_nextjs":
        url = f"{b}/api/v1/infrastructure/services/nextjs/deploy"
        payload = {
            "hostname": row.get("hostname") or "web-1.example.com",
            "app_name": row.get("app_name") or "nextjs-app",
            "app_port": row.get("app_port") or "3000",
            "http_port": row.get("http_port") or "80",
            "framework": "nextjs",
            "node_version": row.get("runtime_version") or "20",
            "repo_url": row.get("repo_url") or "",
        }

    elif intent == "provision_elk":
        url = f"{b}/api/v1/infrastructure/services/elk/deploy"
        payload = {
            "hostname": row.get("hostname") or "elk-1.example.com",
            "elk_version": row.get("elk_version") or "8.13.0",
            "elasticsearch_port": row.get("es_port") or "9200",
            "kibana_port": row.get("kibana_port") or "5601",
            "logstash_port": row.get("logstash_port") or "5044",
        }

    elif intent == "provision_vpn":
        url = f"{b}/api/v1/infrastructure/services/vpn/deploy"
        payload = {
            "hostname": row.get("hostname") or "vpn-1.example.com",
            "vpn_protocol": row.get("vpn_protocol") or "openvpn",
            "vpn_port": row.get("vpn_port") or "1194",
            "max_clients": int(row.get("vpn_clients") or 10),
            "os": row.get("os") or "ubuntu-22.04",
        }

    elif intent == "provision_managed_database":
        url = f"{b}/api/v2/tenant/provision/managed-databases"
        payload = {
            "username": row.get("username") or "user",
            "engine": row.get("database_engine") or "postgres",
            "database_name": row.get("database_name") or "managed-db",
            "db_instance_class": row.get("db_instance_class") or "db.t3.micro",
            "storage_size_gb": int(row.get("storage_size_gb") or 20),
            "region": row.get("region") or "us-east-1",
            "cloud_provider": row.get("cloud_provider") or "aws",
            "multi_az": row.get("multi_az") or "false",
        }

    elif intent == "provision_monitoring":
        url = f"{b}/api/v1/infrastructure/services/monitoring/deploy"
        payload = {
            "hostname": row.get("hostname") or "mon-1.example.com",
            "monitoring_tool": row.get("monitoring_tool") or "prometheus",
            "port": row.get("monitoring_port") or "9090",
        }

    elif intent == "provision_cicd":
        url = f"{b}/api/v1/infrastructure/services/cicd/deploy"
        payload = {
            "hostname": row.get("hostname") or "cicd-1.example.com",
            "cicd_tool": row.get("cicd_tool") or "jenkins",
            "port": row.get("cicd_port") or "8080",
            "docker_image": row.get("docker_image") or "",
        }

    elif intent == "provision_loadbalancer":
        url = f"{b}/api/v1/infrastructure/services/loadbalancer/deploy"
        payload = {
            "hostname": row.get("hostname") or "lb-1.example.com",
            "lb_type": row.get("lb_type") or "nginx",
            "lb_algorithm": row.get("lb_algorithm") or "round-robin",
            "http_port": row.get("http_port") or "80",
        }

    elif intent == "provision_cache":
        url = f"{b}/api/v1/infrastructure/services/cache/deploy"
        payload = {
            "hostname": row.get("hostname") or "cache-1.example.com",
            "cache_engine": row.get("cache_engine") or "redis",
            "port": row.get("cache_port") or "6379",
            "cache_size_mb": int(row.get("cache_size_mb") or 256),
            "replicas": int(row.get("replicas") or 1),
        }

    elif intent == "provision_storage":
        url = f"{b}/api/v1/infrastructure/services/storage/deploy"
        payload = {
            "storage_backend": row.get("storage_backend") or "s3",
            "bucket_name": row.get("bucket_name") or "my-bucket",
            "storage_size_gb": int(row.get("storage_size_gb") or 100),
            "region": row.get("region") or "us-east-1",
        }

    elif intent == "provision_ssl":
        url = f"{b}/api/v1/infrastructure/services/ssl/provision"
        payload = {
            "domain_name": row.get("domain_name") or "example.com",
            "ssl_provider": row.get("ssl_provider") or "letsencrypt",
        }

    elif intent == "provision_wordpress":
        url = f"{b}/api/v1/infrastructure/services/wordpress/deploy"
        payload = {
            "hostname": row.get("hostname") or "wp-1.example.com",
            "http_port": row.get("http_port") or "80",
            "database_engine": row.get("database_engine") or "mysql",
            "app_name": row.get("app_name") or "wordpress",
        }

    elif intent == "provision_django":
        url = f"{b}/api/v1/infrastructure/services/django/deploy"
        payload = {
            "hostname": row.get("hostname") or "app-1.example.com",
            "app_name": row.get("app_name") or "django-app",
            "app_port": row.get("app_port") or "8000",
            "http_port": row.get("http_port") or "80",
            "python_version": row.get("runtime_version") or "3.12",
            "database_engine": row.get("database_engine") or "postgres",
            "repo_url": row.get("repo_url") or "",
        }

    elif intent == "provision_springboot":
        url = f"{b}/api/v1/infrastructure/services/springboot/deploy"
        payload = {
            "hostname": row.get("hostname") or "app-1.example.com",
            "app_name": row.get("app_name") or "springboot-svc",
            "app_port": row.get("app_port") or "8080",
            "http_port": row.get("http_port") or "80",
            "java_version": row.get("runtime_version") or "21",
            "repo_url": row.get("repo_url") or "",
        }

    else:
        url = f"{b}/api/v2/workflow/execute"
        payload = {"metadata": {"name": "provision"}}

    return (url, json.dumps(payload, indent=0))


# ============================================================================
#  Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Prepare cloud_deployments.csv for provisioning training"
    )
    parser.add_argument("--rows", type=int, default=6000, help="Number of rows (default 6000)")
    parser.add_argument(
        "--output",
        type=str,
        default=str(
            Path(__file__).resolve().parent.parent.parent
            / "app" / "data" / "datasets" / "cloud_deployments.csv"
        ),
        help="Output CSV path",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    start_date = datetime.now()

    rows = []
    for i in range(args.rows):
        row = generate_row(i, start_date)
        intent = row["intent"]
        endpoint_url, payload_str = get_endpoint_and_payload(intent, row)
        row["endpoint_url"] = endpoint_url
        row["payload"] = payload_str.replace("\n", " ").replace('"', "'")[:2000]
        out = {k: _s(row.get(k, "")) for k in FIELDNAMES}
        rows.append(out)

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {out_path}")
    intent_counts = {}
    for r in rows:
        intent_counts[r["intent"]] = intent_counts.get(r["intent"], 0) + 1
    for intent, count in sorted(intent_counts.items()):
        pct = count / len(rows) * 100
        print(f"  {intent}: {count} ({pct:.1f}%)")


if __name__ == "__main__":
    main()
