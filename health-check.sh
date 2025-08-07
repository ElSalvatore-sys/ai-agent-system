#!/bin/bash

# =======================================================
# AI AGENT SYSTEM - HEALTH CHECK SCRIPT
# =======================================================
# This script verifies all Docker services are healthy
# and running correctly with comprehensive diagnostics.
# =======================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="$SCRIPT_DIR/health-check.log"
MAX_WAIT_TIME=300
CHECK_INTERVAL=5

# Service definitions
declare -A SERVICES=(
    ["postgres"]="5432"
    ["redis"]="6379"
    ["backend"]="8000"
    ["frontend"]="3000"
    ["ollama"]="11434"
    ["prometheus"]="9090"
    ["grafana"]="3001"
)

declare -A HEALTH_ENDPOINTS=(
    ["backend"]="http://localhost:8000/health"
    ["frontend"]="http://localhost:3000/"
    ["ollama"]="http://localhost:11434/api/tags"
    ["prometheus"]="http://localhost:9090/-/healthy"
    ["grafana"]="http://localhost:3001/api/health"
)

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1" | tee -a "$LOG_FILE"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1" | tee -a "$LOG_FILE"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1" | tee -a "$LOG_FILE"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1" | tee -a "$LOG_FILE"
}

# Header
print_header() {
    echo "=======================================================" | tee -a "$LOG_FILE"
    echo "AI AGENT SYSTEM - HEALTH CHECK" | tee -a "$LOG_FILE"
    echo "Started at: $(date)" | tee -a "$LOG_FILE"
    echo "=======================================================" | tee -a "$LOG_FILE"
}

# Check if Docker is running
check_docker() {
    log_info "Checking Docker daemon..."
    if ! docker info >/dev/null 2>&1; then
        log_error "Docker daemon is not running. Please start Docker first."
        exit 1
    fi
    log_success "Docker daemon is running"
}

# Check if docker-compose is available
check_docker_compose() {
    log_info "Checking Docker Compose..."
    if ! command -v docker-compose >/dev/null 2>&1 && ! docker compose version >/dev/null 2>&1; then
        log_error "Docker Compose is not available"
        exit 1
    fi
    log_success "Docker Compose is available"
}

# Check port availability
check_port() {
    local port=$1
    local service=$2
    
    if command -v nc >/dev/null 2>&1; then
        if nc -z localhost "$port" 2>/dev/null; then
            log_success "Port $port ($service) is open"
            return 0
        else
            log_warning "Port $port ($service) is not accessible"
            return 1
        fi
    elif command -v telnet >/dev/null 2>&1; then
        if timeout 3 telnet localhost "$port" >/dev/null 2>&1; then
            log_success "Port $port ($service) is open"
            return 0
        else
            log_warning "Port $port ($service) is not accessible"
            return 1
        fi
    else
        log_warning "No network checking tool available (nc or telnet)"
        return 1
    fi
}

# Check service container status
check_container_status() {
    local service=$1
    local container_name="ai-agent-system-${service}-1"
    
    # Try different possible container names
    local possible_names=(
        "ai-agent-system-${service}-1"
        "ai-agent-system_${service}_1"
        "${service}"
        "$(basename "$SCRIPT_DIR")_${service}_1"
    )
    
    local container_id=""
    for name in "${possible_names[@]}"; do
        if docker ps --format "table {{.Names}}" | grep -q "^${name}$"; then
            container_id=$(docker ps -q --filter "name=${name}")
            break
        fi
    done
    
    if [[ -z "$container_id" ]]; then
        log_error "Container for service '$service' not found"
        return 1
    fi
    
    local status=$(docker inspect --format='{{.State.Status}}' "$container_id" 2>/dev/null)
    local health=$(docker inspect --format='{{.State.Health.Status}}' "$container_id" 2>/dev/null || echo "none")
    
    if [[ "$status" == "running" ]]; then
        if [[ "$health" == "healthy" ]]; then
            log_success "Service '$service' is running and healthy"
            return 0
        elif [[ "$health" == "unhealthy" ]]; then
            log_error "Service '$service' is running but unhealthy"
            return 1
        else
            log_warning "Service '$service' is running (health status: $health)"
            return 0
        fi
    else
        log_error "Service '$service' is not running (status: $status)"
        return 1
    fi
}

# Check HTTP health endpoints
check_http_endpoint() {
    local service=$1
    local endpoint=${HEALTH_ENDPOINTS[$service]}
    
    if [[ -z "$endpoint" ]]; then
        log_info "No health endpoint defined for $service"
        return 0
    fi
    
    log_info "Checking HTTP endpoint for $service: $endpoint"
    
    if command -v curl >/dev/null 2>&1; then
        if curl -f -s --max-time 10 "$endpoint" >/dev/null 2>&1; then
            log_success "HTTP endpoint for $service is responding"
            return 0
        else
            log_error "HTTP endpoint for $service is not responding"
            return 1
        fi
    elif command -v wget >/dev/null 2>&1; then
        if wget --quiet --timeout=10 --tries=1 --spider "$endpoint" >/dev/null 2>&1; then
            log_success "HTTP endpoint for $service is responding"
            return 0
        else
            log_error "HTTP endpoint for $service is not responding"
            return 1
        fi
    else
        log_warning "No HTTP client available (curl or wget) to check endpoint"
        return 0
    fi
}

# Test database connectivity
test_database_connection() {
    log_info "Testing PostgreSQL database connection..."
    
    local db_container=$(docker ps --format "table {{.Names}}" | grep postgres | head -1)
    if [[ -z "$db_container" ]]; then
        log_error "PostgreSQL container not found"
        return 1
    fi
    
    if docker exec "$db_container" pg_isready -U postgres >/dev/null 2>&1; then
        log_success "PostgreSQL database is accepting connections"
    else
        log_error "PostgreSQL database is not accepting connections"
        return 1
    fi
    
    # Test actual database connection
    if docker exec "$db_container" psql -U postgres -d ai_agent_system -c "SELECT 1;" >/dev/null 2>&1; then
        log_success "Database query test successful"
    else
        log_warning "Database exists but query test failed"
    fi
}

# Test Redis connectivity
test_redis_connection() {
    log_info "Testing Redis connection..."
    
    local redis_container=$(docker ps --format "table {{.Names}}" | grep redis | head -1)
    if [[ -z "$redis_container" ]]; then
        log_error "Redis container not found"
        return 1
    fi
    
    if docker exec "$redis_container" redis-cli ping | grep -q "PONG"; then
        log_success "Redis is responding to ping"
    else
        log_error "Redis is not responding to ping"
        return 1
    fi
    
    # Test basic Redis operations
    if docker exec "$redis_container" redis-cli set healthcheck "test" >/dev/null 2>&1 && \
       docker exec "$redis_container" redis-cli get healthcheck | grep -q "test"; then
        log_success "Redis read/write test successful"
        docker exec "$redis_container" redis-cli del healthcheck >/dev/null 2>&1
    else
        log_warning "Redis basic operations test failed"
    fi
}

# Test Ollama model availability
test_ollama_models() {
    log_info "Testing Ollama model availability..."
    
    local ollama_endpoint="http://localhost:11434/api/tags"
    if command -v curl >/dev/null 2>&1; then
        local models=$(curl -s "$ollama_endpoint" 2>/dev/null)
        if [[ $? -eq 0 ]] && [[ -n "$models" ]]; then
            local model_count=$(echo "$models" | grep -o '"name"' | wc -l)
            if [[ $model_count -gt 0 ]]; then
                log_success "Ollama has $model_count model(s) available"
            else
                log_warning "Ollama is running but no models are installed"
            fi
        else
            log_error "Could not retrieve Ollama model information"
            return 1
        fi
    else
        log_warning "Cannot test Ollama models (curl not available)"
    fi
}

# Test API integrations
test_api_integrations() {
    log_info "Testing API integrations..."
    
    # Check if API keys are configured (without revealing them)
    if [[ -f "$SCRIPT_DIR/.env" ]]; then
        local env_file="$SCRIPT_DIR/.env"
        
        if grep -q "OPENAI_API_KEY=.*[^=]$" "$env_file"; then
            log_success "OpenAI API key is configured"
        else
            log_warning "OpenAI API key not configured"
        fi
        
        if grep -q "ANTHROPIC_API_KEY=.*[^=]$" "$env_file"; then
            log_success "Anthropic API key is configured"
        else
            log_warning "Anthropic API key not configured"
        fi
        
        if grep -q "GOOGLE_API_KEY=.*[^=]$" "$env_file"; then
            log_success "Google API key is configured"
        else
            log_warning "Google API key not configured"
        fi
        
        if grep -q "HUGGINGFACE_API_KEY=.*[^=]$" "$env_file"; then
            log_success "Hugging Face API key is configured"
        else
            log_warning "Hugging Face API key not configured"
        fi
    else
        log_warning "No .env file found - API keys may not be configured"
    fi
}

# Test Docker network connectivity
test_docker_network() {
    log_info "Testing Docker network connectivity..."
    
    local network_name="ai-agent-network"
    if docker network inspect "$network_name" >/dev/null 2>&1; then
        log_success "Docker network '$network_name' exists"
        
        # Get network details
        local network_info=$(docker network inspect "$network_name" --format='{{.IPAM.Config}}')
        log_info "Network configuration: $network_info"
    else
        log_error "Docker network '$network_name' not found"
        return 1
    fi
}

# Test service dependencies
test_service_dependencies() {
    log_info "Testing service dependencies..."
    
    # Check if backend can reach postgres
    local backend_container=$(docker ps --format "table {{.Names}}" | grep backend | head -1)
    if [[ -n "$backend_container" ]]; then
        if docker exec "$backend_container" nc -z postgres 5432 2>/dev/null; then
            log_success "Backend can reach PostgreSQL"
        else
            log_error "Backend cannot reach PostgreSQL"
            return 1
        fi
        
        if docker exec "$backend_container" nc -z redis 6379 2>/dev/null; then
            log_success "Backend can reach Redis"
        else
            log_error "Backend cannot reach Redis"
            return 1
        fi
        
        if docker exec "$backend_container" nc -z ollama 11434 2>/dev/null; then
            log_success "Backend can reach Ollama"
        else
            log_warning "Backend cannot reach Ollama (may not be running)"
        fi
    else
        log_warning "Backend container not found, skipping dependency tests"
    fi
}

# Generate health report
generate_health_report() {
    local total_checks=$1
    local passed_checks=$2
    local failed_checks=$3
    
    echo "=======================================================" | tee -a "$LOG_FILE"
    echo "HEALTH CHECK SUMMARY" | tee -a "$LOG_FILE"
    echo "=======================================================" | tee -a "$LOG_FILE"
    echo "Total Checks: $total_checks" | tee -a "$LOG_FILE"
    echo "Passed: $passed_checks" | tee -a "$LOG_FILE"
    echo "Failed: $failed_checks" | tee -a "$LOG_FILE"
    echo "Success Rate: $(( passed_checks * 100 / total_checks ))%" | tee -a "$LOG_FILE"
    echo "Completed at: $(date)" | tee -a "$LOG_FILE"
    echo "=======================================================" | tee -a "$LOG_FILE"
    
    if [[ $failed_checks -eq 0 ]]; then
        log_success "All health checks passed! System is ready."
        return 0
    else
        log_error "$failed_checks health checks failed. Please review the issues above."
        return 1
    fi
}

# Main health check function
main() {
    local total_checks=0
    local passed_checks=0
    local failed_checks=0
    
    # Initialize log file
    echo "" > "$LOG_FILE"
    print_header
    
    # Basic system checks
    ((total_checks++))
    if check_docker; then ((passed_checks++)); else ((failed_checks++)); fi
    
    ((total_checks++))
    if check_docker_compose; then ((passed_checks++)); else ((failed_checks++)); fi
    
    # Service checks
    for service in "${!SERVICES[@]}"; do
        local port=${SERVICES[$service]}
        
        log_info "Checking service: $service"
        
        # Container status check
        ((total_checks++))
        if check_container_status "$service"; then ((passed_checks++)); else ((failed_checks++)); fi
        
        # Port check
        ((total_checks++))
        if check_port "$port" "$service"; then ((passed_checks++)); else ((failed_checks++)); fi
        
        # HTTP endpoint check
        if [[ -n "${HEALTH_ENDPOINTS[$service]}" ]]; then
            ((total_checks++))
            if check_http_endpoint "$service"; then ((passed_checks++)); else ((failed_checks++)); fi
        fi
    done
    
    # Database connectivity tests
    ((total_checks++))
    if test_database_connection; then ((passed_checks++)); else ((failed_checks++)); fi
    
    ((total_checks++))
    if test_redis_connection; then ((passed_checks++)); else ((failed_checks++)); fi
    
    # Ollama model test
    ((total_checks++))
    if test_ollama_models; then ((passed_checks++)); else ((failed_checks++)); fi
    
    # API integration test
    ((total_checks++))
    if test_api_integrations; then ((passed_checks++)); else ((failed_checks++)); fi
    
    # Docker network test
    ((total_checks++))
    if test_docker_network; then ((passed_checks++)); else ((failed_checks++)); fi
    
    # Service dependencies test
    ((total_checks++))
    if test_service_dependencies; then ((passed_checks++)); else ((failed_checks++)); fi
    
    # Generate final report
    generate_health_report "$total_checks" "$passed_checks" "$failed_checks"
}

# Command line options
case "${1:-}" in
    --help|-h)
        echo "Usage: $0 [OPTIONS]"
        echo "Options:"
        echo "  --help, -h     Show this help message"
        echo "  --quiet, -q    Quiet mode (errors only)"
        echo "  --verbose, -v  Verbose mode (debug info)"
        echo "  --log-file     Custom log file location"
        exit 0
        ;;
    --quiet|-q)
        exec 1>/dev/null
        ;;
    --verbose|-v)
        set -x
        ;;
    --log-file)
        LOG_FILE="$2"
        ;;
esac

# Run main function
main
exit_code=$?

# Display log file location
if [[ $exit_code -eq 0 ]]; then
    log_info "Health check completed successfully. Log saved to: $LOG_FILE"
else
    log_error "Health check completed with errors. Log saved to: $LOG_FILE"
fi

exit $exit_code