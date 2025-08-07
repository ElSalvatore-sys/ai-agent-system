#!/bin/bash

# =======================================================
# FRONTEND CONTAINER DIAGNOSTIC SCRIPT
# =======================================================
# Comprehensive frontend troubleshooting and diagnostics
# for React/Vite containerized applications
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
LOG_FILE="$SCRIPT_DIR/frontend-diagnosis.log"
CONTAINER_NAME="ai-agent-system-frontend-1"
SERVICE_NAME="frontend"
EXPECTED_PORT_INTERNAL=3000
EXPECTED_PORT_EXTERNAL=3000
BACKEND_URL="http://localhost:8000"

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
    echo -e "${PURPLE}[DEBUG]${NC} $1" | tee -a "$LOG_FILE"
}

# Print header
print_header() {
    clear
    echo "=======================================================" | tee -a "$LOG_FILE"
    echo "🔍 FRONTEND CONTAINER DIAGNOSTIC SCRIPT" | tee -a "$LOG_FILE"
    echo "=======================================================" | tee -a "$LOG_FILE"
    echo "Started at: $(date)" | tee -a "$LOG_FILE"
    echo "Container: $CONTAINER_NAME" | tee -a "$LOG_FILE"
    echo "Service: $SERVICE_NAME" | tee -a "$LOG_FILE"
    echo "Log file: $LOG_FILE" | tee -a "$LOG_FILE"
    echo "=======================================================" | tee -a "$LOG_FILE"
}

# Check if required tools are available
check_prerequisites() {
    log_step "Checking prerequisites..."
    
    local missing_tools=()
    
    if ! command -v docker >/dev/null 2>&1; then
        missing_tools+=("docker")
    fi
    
    if ! command -v docker-compose >/dev/null 2>&1 && ! docker compose version >/dev/null 2>&1; then
        missing_tools+=("docker-compose")
    fi
    
    if ! command -v curl >/dev/null 2>&1; then
        missing_tools+=("curl")
    fi
    
    if [[ ${#missing_tools[@]} -gt 0 ]]; then
        log_error "Missing required tools: ${missing_tools[*]}"
        exit 1
    fi
    
    log_success "All prerequisites available"
}

# 1. CONTAINER STATUS CHECK
check_container_status() {
    log_step "1. CONTAINER STATUS CHECK"
    echo "=======================================================" | tee -a "$LOG_FILE"
    
    # Check if container exists
    if ! docker ps -a --format "table {{.Names}}" | grep -q "^${CONTAINER_NAME}$"; then
        log_error "Container '$CONTAINER_NAME' not found"
        log_info "Available containers:"
        docker ps -a --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" | tee -a "$LOG_FILE"
        return 1
    fi
    
    # Get container status
    local container_status=$(docker inspect --format='{{.State.Status}}' "$CONTAINER_NAME" 2>/dev/null)
    local exit_code=$(docker inspect --format='{{.State.ExitCode}}' "$CONTAINER_NAME" 2>/dev/null)
    local restart_count=$(docker inspect --format='{{.RestartCount}}' "$CONTAINER_NAME" 2>/dev/null)
    local started_at=$(docker inspect --format='{{.State.StartedAt}}' "$CONTAINER_NAME" 2>/dev/null)
    local finished_at=$(docker inspect --format='{{.State.FinishedAt}}' "$CONTAINER_NAME" 2>/dev/null)
    
    log_info "Container Status: $container_status"
    log_info "Exit Code: $exit_code"
    log_info "Restart Count: $restart_count"
    log_info "Started At: $started_at"
    
    if [[ "$container_status" != "running" ]]; then
        log_error "Container is not running!"
        log_info "Finished At: $finished_at"
        
        if [[ $exit_code -ne 0 ]]; then
            log_error "Container exited with non-zero code: $exit_code"
        fi
        
        if [[ $restart_count -gt 0 ]]; then
            log_warning "Container has restarted $restart_count times (possible crash loop)"
        fi
        
        return 1
    else
        log_success "Container is running"
        
        if [[ $restart_count -gt 0 ]]; then
            log_warning "Container has restarted $restart_count times"
        fi
    fi
    
    # Check health status if available
    local health_status=$(docker inspect --format='{{.State.Health.Status}}' "$CONTAINER_NAME" 2>/dev/null || echo "none")
    if [[ "$health_status" != "none" ]]; then
        log_info "Health Status: $health_status"
        if [[ "$health_status" == "unhealthy" ]]; then
            log_error "Container is unhealthy!"
            return 1
        fi
    fi
    
    return 0
}

# 2. BUILD VERIFICATION
check_build_verification() {
    log_step "2. BUILD VERIFICATION"
    echo "=======================================================" | tee -a "$LOG_FILE"
    
    # Check recent container logs for build errors
    log_info "Checking container logs for build issues..."
    local recent_logs=$(docker logs --tail=50 "$CONTAINER_NAME" 2>&1)
    
    # Check for common build errors
    if echo "$recent_logs" | grep -i "error" >/dev/null; then
        log_error "Build errors found in logs:"
        echo "$recent_logs" | grep -i "error" | head -10 | tee -a "$LOG_FILE"
    fi
    
    if echo "$recent_logs" | grep -i "failed" >/dev/null; then
        log_error "Build failures found in logs:"
        echo "$recent_logs" | grep -i "failed" | head -10 | tee -a "$LOG_FILE"
    fi
    
    # Check for TypeScript errors
    if echo "$recent_logs" | grep -E "(TS\d+|typescript)" >/dev/null; then
        log_error "TypeScript errors found:"
        echo "$recent_logs" | grep -E "(TS\d+|typescript)" | head -10 | tee -a "$LOG_FILE"
    fi
    
    # Check for missing dependencies
    if echo "$recent_logs" | grep -i "cannot resolve\|module not found\|no matching export" >/dev/null; then
        log_error "Missing dependencies or import errors found:"
        echo "$recent_logs" | grep -i "cannot resolve\|module not found\|no matching export" | head -10 | tee -a "$LOG_FILE"
    fi
    
    # Check if Vite server is running
    if echo "$recent_logs" | grep "VITE.*ready" >/dev/null; then
        log_success "Vite development server is running"
        local vite_url=$(echo "$recent_logs" | grep "Local:" | tail -1 | sed 's/.*Local: *//' | sed 's/ .*//')
        log_info "Vite server URL: $vite_url"
    else
        log_warning "Vite server may not be running properly"
    fi
    
    # Check npm/yarn installation
    log_info "Checking package installation..."
    if docker exec "$CONTAINER_NAME" ls -la /app/node_modules >/dev/null 2>&1; then
        local node_modules_size=$(docker exec "$CONTAINER_NAME" du -sh /app/node_modules 2>/dev/null | cut -f1)
        log_success "node_modules directory exists (size: ${node_modules_size:-unknown})"
    else
        log_error "node_modules directory not found or inaccessible"
    fi
    
    # Check package.json
    if docker exec "$CONTAINER_NAME" test -f /app/package.json >/dev/null 2>&1; then
        log_success "package.json found"
        local package_info=$(docker exec "$CONTAINER_NAME" head -10 /app/package.json 2>/dev/null)
        log_debug "Package info: $package_info"
    else
        log_error "package.json not found"
    fi
}

# 3. PORT MAPPING VERIFICATION
check_port_mapping() {
    log_step "3. PORT MAPPING VERIFICATION"
    echo "=======================================================" | tee -a "$LOG_FILE"
    
    # Check Docker port mapping
    local port_mapping=$(docker port "$CONTAINER_NAME" 2>/dev/null || echo "none")
    if [[ "$port_mapping" == "none" ]]; then
        log_error "No port mappings found for container"
        return 1
    fi
    
    log_info "Container port mappings:"
    echo "$port_mapping" | tee -a "$LOG_FILE"
    
    # Check specific port mapping
    local mapped_port=$(docker port "$CONTAINER_NAME" "$EXPECTED_PORT_INTERNAL" 2>/dev/null | cut -d: -f2)
    if [[ -n "$mapped_port" ]]; then
        log_success "Port $EXPECTED_PORT_INTERNAL is mapped to external port $mapped_port"
    else
        log_error "Port $EXPECTED_PORT_INTERNAL is not mapped"
    fi
    
    # Check for port conflicts
    log_info "Checking for port conflicts..."
    if command -v netstat >/dev/null 2>&1; then
        local port_usage=$(netstat -tuln | grep ":$EXPECTED_PORT_EXTERNAL ")
        if [[ -n "$port_usage" ]]; then
            log_info "Port $EXPECTED_PORT_EXTERNAL usage:"
            echo "$port_usage" | tee -a "$LOG_FILE"
        fi
    elif command -v ss >/dev/null 2>&1; then
        local port_usage=$(ss -tuln | grep ":$EXPECTED_PORT_EXTERNAL ")
        if [[ -n "$port_usage" ]]; then
            log_info "Port $EXPECTED_PORT_EXTERNAL usage:"
            echo "$port_usage" | tee -a "$LOG_FILE"
        fi
    fi
    
    # Test port connectivity
    log_info "Testing port connectivity..."
    if curl -s -o /dev/null -w "%{http_code}" --connect-timeout 5 "http://localhost:$EXPECTED_PORT_EXTERNAL" >/dev/null 2>&1; then
        log_success "Port $EXPECTED_PORT_EXTERNAL is accessible"
    else
        log_error "Port $EXPECTED_PORT_EXTERNAL is not accessible"
    fi
}

# 4. CONFIGURATION ISSUES
check_configuration() {
    log_step "4. CONFIGURATION ISSUES"
    echo "=======================================================" | tee -a "$LOG_FILE"
    
    # Check package.json scripts
    log_info "Checking package.json scripts..."
    if docker exec "$CONTAINER_NAME" cat /app/package.json 2>/dev/null | grep -A 10 '"scripts"' | tee -a "$LOG_FILE"; then
        log_success "Package.json scripts found"
    else
        log_error "Could not read package.json scripts"
    fi
    
    # Check Vite config
    log_info "Checking Vite configuration..."
    if docker exec "$CONTAINER_NAME" test -f /app/vite.config.ts >/dev/null 2>&1; then
        log_success "vite.config.ts found"
        docker exec "$CONTAINER_NAME" cat /app/vite.config.ts 2>/dev/null | tee -a "$LOG_FILE"
    elif docker exec "$CONTAINER_NAME" test -f /app/vite.config.js >/dev/null 2>&1; then
        log_success "vite.config.js found"
        docker exec "$CONTAINER_NAME" cat /app/vite.config.js 2>/dev/null | tee -a "$LOG_FILE"
    else
        log_warning "No Vite config file found"
    fi
    
    # Check environment variables
    log_info "Checking environment variables..."
    local env_vars=$(docker exec "$CONTAINER_NAME" printenv | grep -E "(VITE_|NODE_|API_URL)" || echo "none")
    if [[ "$env_vars" != "none" ]]; then
        log_success "Frontend environment variables:"
        echo "$env_vars" | tee -a "$LOG_FILE"
    else
        log_warning "No frontend-specific environment variables found"
    fi
    
    # Check for common config files
    local config_files=("tsconfig.json" "tailwind.config.js" ".env" ".env.local")
    for config_file in "${config_files[@]}"; do
        if docker exec "$CONTAINER_NAME" test -f "/app/$config_file" >/dev/null 2>&1; then
            log_success "$config_file found"
        else
            log_info "$config_file not found (may be optional)"
        fi
    done
}

# 5. NETWORK CONNECTIVITY
check_network_connectivity() {
    log_step "5. NETWORK CONNECTIVITY"
    echo "=======================================================" | tee -a "$LOG_FILE"
    
    # Check if container can reach backend
    log_info "Testing backend connectivity from container..."
    if docker exec "$CONTAINER_NAME" curl -s -o /dev/null -w "%{http_code}" --connect-timeout 5 "$BACKEND_URL/health" >/dev/null 2>&1; then
        log_success "Container can reach backend at $BACKEND_URL"
    else
        log_error "Container cannot reach backend at $BACKEND_URL"
    fi
    
    # Check internal backend connectivity
    log_info "Testing internal backend connectivity..."
    if docker exec "$CONTAINER_NAME" curl -s -o /dev/null -w "%{http_code}" --connect-timeout 5 "http://backend:8000/health" >/dev/null 2>&1; then
        log_success "Container can reach internal backend service"
    else
        log_error "Container cannot reach internal backend service"
    fi
    
    # Check Docker network
    log_info "Checking Docker network..."
    local network_name=$(docker inspect "$CONTAINER_NAME" --format='{{range $k, $v := .NetworkSettings.Networks}}{{$k}}{{end}}' 2>/dev/null)
    if [[ -n "$network_name" ]]; then
        log_success "Container is on network: $network_name"
        
        # List other containers on same network
        local network_containers=$(docker network inspect "$network_name" --format='{{range .Containers}}{{.Name}} {{end}}' 2>/dev/null)
        log_info "Other containers on network: $network_containers"
    else
        log_error "Could not determine container network"
    fi
    
    # Check CORS configuration
    log_info "Checking CORS configuration..."
    local cors_test=$(curl -s -H "Origin: http://localhost:$EXPECTED_PORT_EXTERNAL" \
                          -H "Access-Control-Request-Method: GET" \
                          -H "Access-Control-Request-Headers: X-Requested-With" \
                          -X OPTIONS \
                          "$BACKEND_URL/health" 2>/dev/null || echo "failed")
    
    if [[ "$cors_test" != "failed" ]]; then
        log_success "CORS preflight request successful"
    else
        log_warning "CORS preflight request failed (may be normal)"
    fi
}

# Generate diagnostic report
generate_diagnostic_report() {
    log_step "GENERATING DIAGNOSTIC REPORT"
    echo "=======================================================" | tee -a "$LOG_FILE"
    
    local report_file="$SCRIPT_DIR/frontend-diagnostic-report.txt"
    
    {
        echo "FRONTEND DIAGNOSTIC REPORT"
        echo "Generated at: $(date)"
        echo "======================================================="
        echo ""
        
        echo "CONTAINER STATUS:"
        docker ps --filter "name=$CONTAINER_NAME" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
        echo ""
        
        echo "RECENT LOGS (last 20 lines):"
        docker logs --tail=20 "$CONTAINER_NAME"
        echo ""
        
        echo "PORT MAPPINGS:"
        docker port "$CONTAINER_NAME"
        echo ""
        
        echo "ENVIRONMENT VARIABLES:"
        docker exec "$CONTAINER_NAME" printenv | grep -E "(VITE_|NODE_|API_)" || echo "None found"
        echo ""
        
        echo "NETWORK INFORMATION:"
        docker inspect "$CONTAINER_NAME" --format='{{range $k, $v := .NetworkSettings.Networks}}Network: {{$k}}, IP: {{$v.IPAddress}}{{"\n"}}{{end}}'
        echo ""
        
    } > "$report_file"
    
    log_success "Diagnostic report saved to: $report_file"
}

# Suggest specific fixes
suggest_fixes() {
    log_step "SUGGESTED FIXES"
    echo "=======================================================" | tee -a "$LOG_FILE"
    
    # Check recent logs for specific errors
    local logs=$(docker logs --tail=100 "$CONTAINER_NAME" 2>&1)
    
    # Import/export errors
    if echo "$logs" | grep -i "no matching export\|cannot resolve\|module not found" >/dev/null; then
        log_warning "🔧 IMPORT/EXPORT ERROR DETECTED:"
        echo "   • Check src/components/features/index.ts exports" | tee -a "$LOG_FILE"
        echo "   • Verify component names in imports match exports" | tee -a "$LOG_FILE"
        echo "   • Run: docker exec $CONTAINER_NAME ls -la /app/src/components/features/" | tee -a "$LOG_FILE"
    fi
    
    # Port issues
    if ! curl -s -o /dev/null --connect-timeout 5 "http://localhost:$EXPECTED_PORT_EXTERNAL" 2>/dev/null; then
        log_warning "🔧 PORT ACCESS ISSUE:"
        echo "   • Check docker-compose.yml port mapping" | tee -a "$LOG_FILE"
        echo "   • Verify no other service is using port $EXPECTED_PORT_EXTERNAL" | tee -a "$LOG_FILE"
        echo "   • Try: docker-compose restart frontend" | tee -a "$LOG_FILE"
    fi
    
    # Build errors
    if echo "$logs" | grep -i "error\|failed" >/dev/null; then
        log_warning "🔧 BUILD ERROR DETECTED:"
        echo "   • Check TypeScript compilation errors" | tee -a "$LOG_FILE"
        echo "   • Verify all dependencies are installed" | tee -a "$LOG_FILE"
        echo "   • Run: docker-compose build --no-cache frontend" | tee -a "$LOG_FILE"
    fi
    
    # Network issues
    if ! docker exec "$CONTAINER_NAME" curl -s -o /dev/null --connect-timeout 5 "http://backend:8000/health" 2>/dev/null; then
        log_warning "🔧 NETWORK CONNECTIVITY ISSUE:"
        echo "   • Check backend service is running" | tee -a "$LOG_FILE"
        echo "   • Verify Docker network configuration" | tee -a "$LOG_FILE"
        echo "   • Check CORS settings in backend" | tee -a "$LOG_FILE"
    fi
    
    echo ""
    log_info "💡 QUICK FIX COMMANDS:"
    echo "   # Restart frontend service:" | tee -a "$LOG_FILE"
    echo "   docker-compose restart frontend" | tee -a "$LOG_FILE"
    echo ""
    echo "   # Rebuild frontend (if code changes):" | tee -a "$LOG_FILE"
    echo "   docker-compose build --no-cache frontend" | tee -a "$LOG_FILE"
    echo "   docker-compose up -d frontend" | tee -a "$LOG_FILE"
    echo ""
    echo "   # View real-time logs:" | tee -a "$LOG_FILE"
    echo "   docker-compose logs -f frontend" | tee -a "$LOG_FILE"
    echo ""
    echo "   # Access container shell for debugging:" | tee -a "$LOG_FILE"
    echo "   docker exec -it $CONTAINER_NAME /bin/sh" | tee -a "$LOG_FILE"
}

# Main diagnostic function
main() {
    # Initialize log file
    echo "" > "$LOG_FILE"
    print_header
    
    local exit_code=0
    
    # Run diagnostic checks
    check_prerequisites || exit_code=1
    check_container_status || exit_code=1
    check_build_verification || exit_code=1
    check_port_mapping || exit_code=1
    check_configuration || exit_code=1
    check_network_connectivity || exit_code=1
    
    # Generate reports and suggestions
    generate_diagnostic_report
    suggest_fixes
    
    # Final summary
    echo ""
    echo "=======================================================" | tee -a "$LOG_FILE"
    if [[ $exit_code -eq 0 ]]; then
        log_success "✅ Frontend diagnostic completed - No critical issues found"
    else
        log_error "❌ Frontend diagnostic completed - Issues detected"
        log_info "📋 Check the diagnostic report and suggested fixes above"
    fi
    echo "📄 Full log: $LOG_FILE" | tee -a "$LOG_FILE"
    echo "📊 Report: $SCRIPT_DIR/frontend-diagnostic-report.txt" | tee -a "$LOG_FILE"
    echo "=======================================================" | tee -a "$LOG_FILE"
    
    return $exit_code
}

# Command line options
case "${1:-}" in
    --help|-h)
        echo "Frontend Container Diagnostic Script"
        echo ""
        echo "Usage: $0 [OPTIONS]"
        echo ""
        echo "Options:"
        echo "  --help, -h          Show this help message"
        echo "  --container NAME    Use specific container name"
        echo "  --port PORT         Expected external port (default: 3000)"
        echo "  --verbose           Show verbose output"
        echo "  --logs-only         Show only container logs"
        echo ""
        exit 0
        ;;
    --container)
        CONTAINER_NAME="$2"
        shift
        ;;
    --port)
        EXPECTED_PORT_EXTERNAL="$2"
        shift
        ;;
    --verbose)
        set -x
        ;;
    --logs-only)
        echo "Recent frontend container logs:"
        docker logs --tail=50 "$CONTAINER_NAME"
        exit 0
        ;;
esac

# Run main function
main
exit $?