#!/bin/bash

# Deploy LLM System to Kubernetes
# Usage: ./deploy.sh [namespace]

set -e

NAMESPACE=${1:-llm-system}
KUBECTL_CMD="kubectl"

echo "🚀 Deploying LLM System to Kubernetes"
echo "Namespace: $NAMESPACE"
echo "============================================="

# Function to check if kubectl is available
check_kubectl() {
    if ! command -v kubectl &> /dev/null; then
        echo "❌ kubectl is not installed or not in PATH"
        exit 1
    fi
    
    # Check cluster connection
    if ! kubectl cluster-info &> /dev/null; then
        echo "❌ Unable to connect to Kubernetes cluster"
        exit 1
    fi
    
    echo "✅ kubectl is available and connected to cluster"
}

# Function to check if required storage classes exist
check_storage_classes() {
    echo "🔍 Checking storage classes..."
    
    if ! kubectl get storageclass fast-ssd &> /dev/null; then
        echo "⚠️  StorageClass 'fast-ssd' not found. Creating default..."
        cat <<EOF | kubectl apply -f -
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: fast-ssd
provisioner: kubernetes.io/gce-pd
parameters:
  type: pd-ssd
  replication-type: none
allowVolumeExpansion: true
volumeBindingMode: WaitForFirstConsumer
EOF
    fi
    
    echo "✅ Storage classes verified"
}

# Function to create or update namespace
create_namespace() {
    echo "📦 Creating namespace..."
    kubectl apply -f namespace.yaml
    echo "✅ Namespace created/updated"
}

# Function to deploy infrastructure components
deploy_infrastructure() {
    echo "🏗️  Deploying infrastructure components..."
    
    echo "  📊 Deploying Redis..."
    kubectl apply -f redis.yaml
    
    echo "  🗄️  Deploying PostgreSQL..."
    kubectl apply -f postgres.yaml
    
    echo "✅ Infrastructure components deployed"
}

# Function to wait for infrastructure to be ready
wait_for_infrastructure() {
    echo "⏳ Waiting for infrastructure to be ready..."
    
    echo "  Waiting for Redis..."
    kubectl wait --for=condition=ready pod -l app=redis -n $NAMESPACE --timeout=300s
    
    echo "  Waiting for PostgreSQL..."
    kubectl wait --for=condition=ready pod -l app=postgres -n $NAMESPACE --timeout=300s
    
    echo "✅ Infrastructure is ready"
}

# Function to deploy application components
deploy_applications() {
    echo "🚀 Deploying application components..."
    
    echo "  🤖 Deploying Ollama..."
    kubectl apply -f ollama.yaml
    
    echo "  ⚙️  Deploying LLM API..."
    kubectl apply -f llm-api.yaml
    
    echo "  👷 Deploying Celery workers..."
    kubectl apply -f celery-workers.yaml
    
    echo "✅ Application components deployed"
}

# Function to deploy monitoring
deploy_monitoring() {
    echo "📊 Deploying monitoring stack..."
    kubectl apply -f monitoring.yaml
    echo "✅ Monitoring stack deployed"
}

# Function to deploy ingress
deploy_ingress() {
    echo "🌐 Deploying ingress..."
    kubectl apply -f ingress.yaml
    echo "✅ Ingress deployed"
}

# Function to wait for applications to be ready
wait_for_applications() {
    echo "⏳ Waiting for applications to be ready..."
    
    echo "  Waiting for Ollama..."
    kubectl wait --for=condition=ready pod -l app=ollama -n $NAMESPACE --timeout=600s
    
    echo "  Waiting for LLM API..."
    kubectl wait --for=condition=ready pod -l app=llm-api -n $NAMESPACE --timeout=300s
    
    echo "  Waiting for Celery workers..."
    kubectl wait --for=condition=ready pod -l app=celery-worker -n $NAMESPACE --timeout=300s
    
    echo "✅ Applications are ready"
}

# Function to run post-deployment tasks
post_deployment() {
    echo "🔧 Running post-deployment tasks..."
    
    # Run database migrations
    echo "  📋 Running database migrations..."
    kubectl exec -n $NAMESPACE deployment/llm-api -- python -m alembic upgrade head
    
    # Trigger model preloading
    echo "  🤖 Triggering model preloading..."
    kubectl apply -f ollama.yaml  # This includes the preload job
    
    echo "✅ Post-deployment tasks completed"
}

# Function to display deployment status
show_status() {
    echo ""
    echo "📋 Deployment Status"
    echo "===================="
    
    echo ""
    echo "🏠 Namespace:"
    kubectl get namespace $NAMESPACE
    
    echo ""
    echo "📦 Pods:"
    kubectl get pods -n $NAMESPACE -o wide
    
    echo ""
    echo "🔌 Services:"
    kubectl get services -n $NAMESPACE
    
    echo ""
    echo "💾 Storage:"
    kubectl get pvc -n $NAMESPACE
    
    echo ""
    echo "🌐 Ingress:"
    kubectl get ingress -n $NAMESPACE
    
    echo ""
    echo "📊 HPA Status:"
    kubectl get hpa -n $NAMESPACE
}

# Function to show access information
show_access_info() {
    echo ""
    echo "🔑 Access Information"
    echo "===================="
    
    # Get LoadBalancer IP
    LB_IP=$(kubectl get service llm-system-lb -n $NAMESPACE -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null || echo "Pending...")
    
    echo ""
    echo "🌐 External Access:"
    echo "  Load Balancer IP: $LB_IP"
    echo "  API Endpoint: https://llm-api.yourdomain.com"
    echo "  Monitoring: https://llm-monitoring.yourdomain.com"
    
    echo ""
    echo "🔍 Internal Access (kubectl port-forward):"
    echo "  API: kubectl port-forward -n $NAMESPACE svc/llm-api-service 8000:8000"
    echo "  Grafana: kubectl port-forward -n $NAMESPACE svc/grafana-service 3000:3000"
    echo "  Prometheus: kubectl port-forward -n $NAMESPACE svc/prometheus-service 9090:9090"
    echo "  Flower: kubectl port-forward -n $NAMESPACE svc/celery-flower-service 5555:5555"
    
    echo ""
    echo "📋 Default Credentials:"
    echo "  Grafana: admin / grafana-password-change-in-production"
    echo "  Flower: admin / flower-password-change-in-production"
}

# Function to create monitoring dashboards
setup_monitoring() {
    echo "📊 Setting up monitoring dashboards..."
    
    # Wait for Grafana to be ready
    kubectl wait --for=condition=ready pod -l app=grafana -n $NAMESPACE --timeout=300s
    
    echo "✅ Monitoring setup completed"
}

# Main deployment flow
main() {
    echo "Starting deployment process..."
    
    check_kubectl
    check_storage_classes
    create_namespace
    deploy_infrastructure
    wait_for_infrastructure
    deploy_applications
    deploy_monitoring
    deploy_ingress
    wait_for_applications
    setup_monitoring
    post_deployment
    
    echo ""
    echo "🎉 Deployment completed successfully!"
    echo ""
    
    show_status
    show_access_info
    
    echo ""
    echo "💡 Next Steps:"
    echo "  1. Update DNS records to point to the Load Balancer IP"
    echo "  2. Configure SSL certificates (cert-manager should handle this automatically)"
    echo "  3. Update secrets with production API keys"
    echo "  4. Configure monitoring alerts"
    echo "  5. Test the deployment with sample requests"
    
    echo ""
    echo "🔧 Useful Commands:"
    echo "  View logs: kubectl logs -f deployment/llm-api -n $NAMESPACE"
    echo "  Scale workers: kubectl scale deployment celery-worker --replicas=5 -n $NAMESPACE"
    echo "  Check metrics: kubectl top pods -n $NAMESPACE"
    echo "  Debug pod: kubectl exec -it POD_NAME -n $NAMESPACE -- /bin/bash"
}

# Handle script interruption
trap 'echo "❌ Deployment interrupted"; exit 1' INT TERM

# Run main function
main "$@"