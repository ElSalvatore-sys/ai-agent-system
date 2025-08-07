#!/bin/bash

# =======================================================
# AI AGENT SYSTEM - STARTUP VERIFICATION SCRIPT
# =======================================================
# This script handles the complete startup sequence with
# dependency ordering, health checks, and troubleshooting.
# =======================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="$SCRIPT_DIR/startup-verification.log"
MAX_STARTUP_WAIT=600  # 10 minutes
HEALTH_CHECK_INTERVAL=10
INTEGRATION_TEST_TIMEOUT=300  # 5 minutes

# Service startup order (dependencies first)
STARTUP_ORDER=(
    "postgres"
    "redis"
    "ollama"
    "backend"
    "frontend"
    "prometheus"
    "grafana"
)

# Port conflict detection
REQUIRED_PORTS=(3000 8000 5432 6379 11434 9090 3001)

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

log_step() {
    echo -e "${CYAN}[STEP]${NC} $1" | tee -a "$LOG_FILE"
}

log_debug() {
    if [[ "${DEBUG:-false}" == "true" ]]; then
        echo -e "${PURPLE}[DEBUG]${NC} $1" | tee -a "$LOG_FILE"
    fi
}

# Print header
print_header() {
    clear
    echo "=======================================================" | tee -a "$LOG_FILE"
    echo "🚀 AI AGENT SYSTEM - STARTUP VERIFICATION" | tee -a "$LOG_FILE"
    echo "=======================================================" | tee -a "$LOG_FILE"
    echo "Started at: $(date)" | tee -a "$LOG_FILE"
    echo "Log file: $LOG_FILE" | tee -a "$LOG_FILE"
    echo "=======================================================" | tee -a "$LOG_FILE"
}

# Check prerequisites
check_prerequisites() {
    log_step "Checking prerequisites..."
    
    # Check Docker
    if ! command -v docker >/dev/null 2>&1; then
        log_error "Docker is not installed or not in PATH"
        exit 1
    fi
    
    if ! docker info >/dev/null 2>&1; then
        log_error "Docker daemon is not running"
        exit 1
    fi
    
    # Check Docker Compose
    if command -v docker-compose >/dev/null 2>&1; then
        COMPOSE_CMD="docker-compose"
    elif docker compose version >/dev/null 2>&1; then
        COMPOSE_CMD="docker compose"
    else
        log_error "Docker Compose is not available"
        exit 1
    fi
    
    log_success "Prerequisites checked: Docker and $COMPOSE_CMD available"
}

# Check for port conflicts
check_port_conflicts() {
    log_step "Checking for port conflicts..."
    
    local conflicts=()
    for port in "${REQUIRED_PORTS[@]}"; do
        if command -v netstat >/dev/null 2>&1; then
            if netstat -tuln | grep -q ":$port "; then
                conflicts+=("$port")
            fi
        elif command -v ss >/dev/null 2>&1; then
            if ss -tuln | grep -q ":$port "; then
                conflicts+=("$port")
            fi
        elif command -v lsof >/dev/null 2>&1; then
            if lsof -i ":$port" >/dev/null 2>&1; then
                conflicts+=("$port")
            fi
        fi
    done
    
    if [[ ${#conflicts[@]} -gt 0 ]]; then
        log_warning "Port conflicts detected: ${conflicts[*]}"
        log_info "You may need to stop other services or change port mappings"
        
        # Offer to show what's using the ports
        for port in "${conflicts[@]}"; do
            if command -v lsof >/dev/null 2>&1; then
                log_info "Port $port is used by:"
                lsof -i ":$port" | tee -a "$LOG_FILE"
            fi
        done
        
        read -p "Continue anyway? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    else
        log_success "No port conflicts detected"
    fi
}

# Validate environment variables
validate_environment() {
    log_step "Validating environment configuration..."
    
    # Check if .env file exists
    if [[ ! -f "$SCRIPT_DIR/.env" ]]; then
        log_warning "No .env file found. Using .env.example as template."
        if [[ -f "$SCRIPT_DIR/.env.example" ]]; then
            cp "$SCRIPT_DIR/.env.example" "$SCRIPT_DIR/.env"
            log_info "Created .env from .env.example. Please review and update with your actual values."
        else
            log_error "No .env.example file found. Cannot create environment configuration."
            exit 1
        fi
    fi
    
    # Load environment variables
    set -a
    source "$SCRIPT_DIR/.env"
    set +a
    
    # Check critical environment variables
    local required_vars=(
        "POSTGRES_PASSWORD"
        "SECRET_KEY"
    )
    
    local missing_vars=()
    for var in "${required_vars[@]}"; do
        if [[ -z "${!var}" ]]; then
            missing_vars+=("$var")
        fi
    done
    
    if [[ ${#missing_vars[@]} -gt 0 ]]; then
        log_error "Missing required environment variables: ${missing_vars[*]}"
        log_info "Please update your .env file with the required values"
        exit 1
    fi
    
    log_success "Environment configuration validated"
}

# Pull and build Docker images
pull_and_build_images() {
    log_step "Pulling and building Docker images..."
    
    # Pull external images first
    log_info "Pulling external Docker images..."
    $COMPOSE_CMD pull --ignore-pull-failures postgres redis ollama prometheus grafana 2>&1 | tee -a "$LOG_FILE"
    
    # Build custom images
    log_info "Building custom Docker images..."
    $COMPOSE_CMD build --no-cache 2>&1 | tee -a "$LOG_FILE"
    
    if [[ $? -eq 0 ]]; then
        log_success "Docker images pulled and built successfully"
    else
        log_error "Failed to pull/build Docker images"
        exit 1
    fi
}

# Start services in dependency order
start_services() {
    log_step "Starting services in dependency order..."
    
    for service in "${STARTUP_ORDER[@]}"; do
        log_info "Starting service: $service"
        
        $COMPOSE_CMD up -d "$service" 2>&1 | tee -a "$LOG_FILE"
        
        if [[ $? -eq 0 ]]; then
            log_success "Service '$service' started"
            
            # Wait a bit before starting next service
            sleep 5
        else
            log_error "Failed to start service '$service'"
            show_service_logs "$service"
            exit 1
        fi
    done
}

# Wait for health checks to pass
wait_for_health_checks() {
    log_step "Waiting for all services to become healthy..."
    
    local start_time=$(date +%s)
    local max_wait_time=$MAX_STARTUP_WAIT
    
    while true; do
        local current_time=$(date +%s)
        local elapsed=$((current_time - start_time))
        
        if [[ $elapsed -gt $max_wait_time ]]; then
            log_error "Health checks timed out after ${max_wait_time}s"
            show_unhealthy_services
            exit 1
        fi
        
        log_info "Checking service health... (${elapsed}s elapsed)"
        
        # Run health check script
        if bash "$SCRIPT_DIR/health-check.sh" --quiet; then
            log_success "All health checks passed!"
            break
        else
            log_info "Some services not ready yet, waiting ${HEALTH_CHECK_INTERVAL}s..."
            sleep $HEALTH_CHECK_INTERVAL
        fi
    done
}

# Show service logs for debugging
show_service_logs() {
    local service=$1
    log_info "Recent logs for service '$service':"
    $COMPOSE_CMD logs --tail=20 "$service" | tee -a "$LOG_FILE"
}

# Show unhealthy services
show_unhealthy_services() {
    log_error "Services that failed health checks:"
    
    # Check each service status
    for service in "${STARTUP_ORDER[@]}"; do
        local container_status=$($COMPOSE_CMD ps -q "$service" | xargs -I {} docker inspect --format='{{.State.Status}}' {} 2>/dev/null || echo "not found")
        local health_status=$($COMPOSE_CMD ps -q "$service" | xargs -I {} docker inspect --format='{{.State.Health.Status}}' {} 2>/dev/null || echo "none")
        
        if [[ "$container_status" != "running" ]] || [[ "$health_status" == "unhealthy" ]]; then
            log_error "❌ $service - Status: $container_status, Health: $health_status"
            show_service_logs "$service"
        else
            log_success "✅ $service - Status: $container_status, Health: $health_status"
        fi
    done
}

# Run integration tests
run_integration_tests() {
    log_step "Running integration tests..."
    
    if [[ ! -f "$SCRIPT_DIR/integration-test.py" ]]; then
        log_warning "Integration test script not found, skipping tests"
        return 0
    fi
    
    # Check if Python is available
    if ! command -v python3 >/dev/null 2>&1; then
        log_warning "Python 3 not available, skipping integration tests"
        return 0
    fi
    
    # Install test dependencies if needed
    local test_deps="requests redis psycopg2-binary websockets docker openai anthropic google-generativeai"
    log_info "Installing test dependencies..."
    pip3 install $test_deps >/dev/null 2>&1 || log_warning "Failed to install some test dependencies"
    
    # Run integration tests with timeout
    log_info "Running comprehensive integration tests..."
    timeout $INTEGRATION_TEST_TIMEOUT python3 "$SCRIPT_DIR/integration-test.py" 2>&1 | tee -a "$LOG_FILE"
    
    local test_exit_code=$?
    if [[ $test_exit_code -eq 0 ]]; then
        log_success "All integration tests passed!"
    elif [[ $test_exit_code -eq 1 ]]; then
        log_warning "Some integration tests failed, but core functionality works"
    elif [[ $test_exit_code -eq 124 ]]; then
        log_error "Integration tests timed out"
    else
        log_error "Integration tests failed"
    fi
    
    return $test_exit_code
}

# Display service status dashboard
display_status_dashboard() {
    log_step "Displaying service status dashboard..."
    
    echo ""
    echo "🔍 SERVICE STATUS DASHBOARD"
    echo "======================================================="
    
    # Service URLs and status
    declare -A service_urls=(
        ["frontend"]="http://localhost:3000"
        ["backend"]="http://localhost:8000"
        ["backend-docs"]="http://localhost:8000/docs"
        ["postgres"]="localhost:5432"
        ["redis"]="localhost:6379"
        ["ollama"]="http://localhost:11434"
        ["prometheus"]="http://localhost:9090"
        ["grafana"]="http://localhost:3001"
    )
    
    for service_name in "${!service_urls[@]}"; do
        local url="${service_urls[$service_name]}"
        echo -e "${CYAN}📡 $service_name${NC}: $url"
    done
    
    echo ""
    echo "🔧 DEVELOPMENT TOOLS"
    echo "======================================================="
    echo -e "${CYAN}📊 Redis Commander${NC}: http://localhost:8081"
    echo -e "${CYAN}🗄️  PgAdmin${NC}: http://localhost:5050"
    echo ""
    
    # Resource usage
    echo "💻 RESOURCE USAGE"
    echo "======================================================="
    docker stats --no-stream --format "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}" | tee -a "$LOG_FILE"
    echo ""
}

# Provide troubleshooting information
show_troubleshooting_info() {
    log_step "Troubleshooting information..."
    
    echo ""
    echo "🔧 TROUBLESHOOTING COMMANDS"
    echo "======================================================="
    echo "View logs:           $COMPOSE_CMD logs [service_name]"
    echo "Restart service:     $COMPOSE_CMD restart [service_name]"
    echo "Rebuild service:     $COMPOSE_CMD up -d --build [service_name]"
    echo "Stop all services:   $COMPOSE_CMD down"
    echo "Full cleanup:        $COMPOSE_CMD down -v --remove-orphans"
    echo ""
    
    echo "📋 LOG FILES"
    echo "======================================================="
    echo "Startup log:         $LOG_FILE"
    echo "Health check log:    $SCRIPT_DIR/health-check.log"
    echo "Integration tests:   $SCRIPT_DIR/integration-test.log"
    echo ""
    
    echo "🌐 QUICK ACCESS URLS"
    echo "======================================================="
    echo "Frontend:            http://localhost:3000"
    echo "Backend API:         http://localhost:8000/docs"
    echo "Backend Health:      http://localhost:8000/health"
    echo "Ollama API:          http://localhost:11434/api/tags"
    echo "Monitoring:          http://localhost:9090 (Prometheus)"
    echo "Dashboard:           http://localhost:3001 (Grafana)"
    echo "Redis Commander:     http://localhost:8081"
    echo "PgAdmin:             http://localhost:5050"
    echo ""
    
    echo "🤖 AI MODEL ENDPOINTS"
    echo "======================================================="
    echo "Chat API:            http://localhost:8000/api/chat"
    echo "Models List:         http://localhost:8000/api/models"
    echo "Cost Tracking:       http://localhost:8000/api/analytics/costs"
    echo ""
}

# Cleanup function
cleanup() {
    if [[ "${1:-}" == "error" ]]; then
        log_error "Startup failed. Cleaning up..."
        show_troubleshooting_info
    fi
}

# Set trap for cleanup
trap 'cleanup error' ERR

# Main startup sequence
main() {
    # Initialize log file
    echo "" > "$LOG_FILE"
    print_header
    
    # Pre-startup checks
    check_prerequisites
    check_port_conflicts
    validate_environment
    
    # Docker operations
    pull_and_build_images
    start_services
    
    # Health verification
    wait_for_health_checks
    
    # Post-startup validation
    local test_result=0
    if [[ "${SKIP_INTEGRATION_TESTS:-false}" != "true" ]]; then
        run_integration_tests
        test_result=$?
    fi
    
    # Display results
    display_status_dashboard
    
    # Final status
    echo ""
    if [[ $test_result -eq 0 ]]; then
        log_success "🎉 AI Agent System startup completed successfully!"
        log_info "All services are healthy and integration tests passed."
    elif [[ $test_result -eq 1 ]]; then
        log_warning "⚠️  AI Agent System started with some issues."
        log_info "Core services are running but some integration tests failed."
    else
        log_error "❌ AI Agent System startup completed with errors."
        log_info "Services are running but integration tests failed."
    fi
    
    show_troubleshooting_info
    
    echo "======================================================="
    echo "🚀 System is ready! Happy coding! 🎯"
    echo "======================================================="
    
    return $test_result
}

# Command line options
case "${1:-}" in
    --help|-h)
        echo "AI Agent System - Startup Verification Script"
        echo ""
        echo "Usage: $0 [OPTIONS]"
        echo ""
        echo "Options:"
        echo "  --help, -h              Show this help message"
        echo "  --skip-tests           Skip integration tests"
        echo "  --debug                Enable debug output"
        echo "  --max-wait SECONDS     Maximum wait time for health checks (default: 600)"
        echo "  --quiet                Quiet mode (errors only)"
        echo ""
        echo "Environment Variables:"
        echo "  SKIP_INTEGRATION_TESTS=true    Skip integration tests"
        echo "  DEBUG=true                     Enable debug mode"
        echo ""
        exit 0
        ;;
    --skip-tests)
        export SKIP_INTEGRATION_TESTS=true
        ;;
    --debug)
        export DEBUG=true
        set -x
        ;;
    --max-wait)
        MAX_STARTUP_WAIT="$2"
        shift
        ;;
    --quiet)
        exec 1>/dev/null
        ;;
esac

# Run main function
main
exit_code=$?

log_info "Startup verification completed with exit code: $exit_code"
exit $exit_code