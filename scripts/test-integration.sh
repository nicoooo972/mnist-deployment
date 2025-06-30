#!/bin/bash

# Tests d'intégration MNIST
# Vérifie le bon fonctionnement de l'ensemble de l'application

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
BACKEND_URL="${BACKEND_URL:-http://localhost:8000}"
FRONTEND_URL="${FRONTEND_URL:-http://localhost:8501}"
TIMEOUT="${TIMEOUT:-30}"

# Couleurs
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

test_count=0
pass_count=0
fail_count=0

# Functions
log() {
    echo -e "${GREEN}[TEST]${NC} $1"
}

error() {
    echo -e "${RED}[FAIL]${NC} $1"
    ((fail_count++))
}

success() {
    echo -e "${GREEN}[PASS]${NC} $1"
    ((pass_count++))
}

info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

run_test() {
    local test_name="$1"
    local test_command="$2"
    
    ((test_count++))
    log "Running: $test_name"
    
    if eval "$test_command"; then
        success "$test_name"
        return 0
    else
        error "$test_name"
        return 1
    fi
}

# Wait for service to be ready
wait_for_service() {
    local url="$1"
    local service_name="$2"
    local max_attempts="$TIMEOUT"
    local attempt=1
    
    info "⏳ Waiting for $service_name to be ready..."
    
    while [ $attempt -le $max_attempts ]; do
        if curl -sf "$url" >/dev/null 2>&1; then
            success "$service_name is ready"
            return 0
        fi
        
        if [ $attempt -eq $max_attempts ]; then
            error "$service_name not ready after $max_attempts attempts"
            return 1
        fi
        
        sleep 1
        ((attempt++))
    done
}

# Test backend health
test_backend_health() {
    curl -sf "$BACKEND_URL/health" | grep -q "ok"
}

# Test backend API info
test_backend_info() {
    curl -sf "$BACKEND_URL/api/info" | grep -q "MNIST"
}

# Test backend docs
test_backend_docs() {
    curl -sf "$BACKEND_URL/docs" | grep -q "FastAPI"
}

# Test model prediction endpoint
test_model_prediction() {
    # Créer une image de test simplifiée (28x28 pixels)
    local test_image="/tmp/test_mnist.json"
    
    # Générer des données d'image aléatoires
    python3 -c "
import json
import random
# Simuler une image 28x28 (784 pixels)
pixels = [random.random() for _ in range(784)]
data = {'image': pixels}
with open('$test_image', 'w') as f:
    json.dump(data, f)
"
    
    # Test de l'endpoint de prédiction
    local response=$(curl -sf -X POST \
        -H "Content-Type: application/json" \
        -d @"$test_image" \
        "$BACKEND_URL/api/v1/predict")
    
    echo "$response" | grep -q "prediction"
}

# Test frontend availability
test_frontend_health() {
    curl -sf "$FRONTEND_URL" | grep -q "MNIST"
}

# Test metrics endpoint (if available)
test_metrics() {
    if curl -sf "$BACKEND_URL/metrics" >/dev/null 2>&1; then
        curl -sf "$BACKEND_URL/metrics" | grep -q "http_requests_total"
    else
        warn "Metrics endpoint not available"
        return 0  # Not critical
    fi
}

# Test database connection (if applicable)
test_database() {
    # Si tu as une base de données, teste la connexion ici
    info "Database tests skipped (no database configured)"
    return 0
}

# Load testing simulation
test_load() {
    local concurrent_requests=5
    local total_requests=20
    
    info "Running basic load test ($total_requests requests, $concurrent_requests concurrent)..."
    
    # Utilise GNU parallel si disponible, sinon séquentiel
    if command -v parallel >/dev/null 2>&1; then
        seq 1 "$total_requests" | parallel -j"$concurrent_requests" \
            "curl -sf $BACKEND_URL/health >/dev/null"
    else
        for i in $(seq 1 "$total_requests"); do
            curl -sf "$BACKEND_URL/health" >/dev/null &
            if [ $((i % concurrent_requests)) -eq 0 ]; then
                wait
            fi
        done
        wait
    fi
}

# Performance test
test_response_time() {
    local max_response_time=2000  # 2 seconds in milliseconds
    
    local response_time=$(curl -sf -w "%{time_total}" -o /dev/null "$BACKEND_URL/health")
    local response_time_ms=$(echo "$response_time * 1000" | bc -l | cut -d. -f1)
    
    if [ "$response_time_ms" -lt "$max_response_time" ]; then
        success "Response time: ${response_time_ms}ms (< ${max_response_time}ms)"
        return 0
    else
        error "Response time too slow: ${response_time_ms}ms (> ${max_response_time}ms)"
        return 1
    fi
}

# Security tests
test_security() {
    info "Running basic security tests..."
    
    # Test CORS headers
    local cors_header=$(curl -sf -I "$BACKEND_URL/health" | grep -i "access-control-allow-origin" || echo "")
    if [ -n "$cors_header" ]; then
        info "CORS configured: $cors_header"
    fi
    
    # Test for SQL injection protection (basic)
    local sql_injection_response=$(curl -sf "$BACKEND_URL/api/info?id=1';DROP TABLE users;--" | head -c 100)
    if echo "$sql_injection_response" | grep -qi "error\|drop\|table"; then
        warn "Potential SQL injection vulnerability detected"
    fi
    
    return 0
}

# Test logs and monitoring
test_monitoring() {
    info "Testing monitoring capabilities..."
    
    # Vérifier si Prometheus est accessible
    if curl -sf "http://localhost:9090" >/dev/null 2>&1; then
        success "Prometheus monitoring available"
    else
        info "Prometheus not running (optional)"
    fi
    
    # Vérifier si Grafana est accessible
    if curl -sf "http://localhost:3000" >/dev/null 2>&1; then
        success "Grafana dashboard available"
    else
        info "Grafana not running (optional)"
    fi
    
    return 0
}

# Main test suite
run_test_suite() {
    log "🧪 Starting MNIST Integration Tests"
    log "Backend URL: $BACKEND_URL"
    log "Frontend URL: $FRONTEND_URL"
    echo ""
    
    # Prerequisites
    wait_for_service "$BACKEND_URL/health" "Backend" || exit 1
    wait_for_service "$FRONTEND_URL" "Frontend" || exit 1
    
    echo ""
    log "🔥 Running test suite..."
    
    # Core functionality tests
    run_test "Backend health check" "test_backend_health"
    run_test "Backend API info" "test_backend_info"
    run_test "Backend documentation" "test_backend_docs"
    run_test "Model prediction" "test_model_prediction"
    run_test "Frontend availability" "test_frontend_health"
    
    # Performance tests
    run_test "Response time" "test_response_time"
    run_test "Load testing" "test_load"
    
    # Optional tests
    run_test "Metrics endpoint" "test_metrics"
    run_test "Database connectivity" "test_database"
    run_test "Security checks" "test_security"
    run_test "Monitoring services" "test_monitoring"
    
    echo ""
    log "📊 Test Results:"
    echo "  Total tests: $test_count"
    echo "  Passed: $pass_count"
    echo "  Failed: $fail_count"
    
    if [ $fail_count -eq 0 ]; then
        success "🎉 All tests passed!"
        exit 0
    else
        error "❌ $fail_count test(s) failed"
        exit 1
    fi
}

# Help
show_help() {
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  --backend-url URL    Backend URL (default: http://localhost:8000)"
    echo "  --frontend-url URL   Frontend URL (default: http://localhost:8501)"
    echo "  --timeout SECONDS    Timeout for service readiness (default: 30)"
    echo "  --help               Show this help"
    echo ""
    echo "Environment variables:"
    echo "  BACKEND_URL          Override default backend URL"
    echo "  FRONTEND_URL         Override default frontend URL"
    echo "  TIMEOUT              Override default timeout"
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --backend-url)
            BACKEND_URL="$2"
            shift 2
            ;;
        --frontend-url)
            FRONTEND_URL="$2"
            shift 2
            ;;
        --timeout)
            TIMEOUT="$2"
            shift 2
            ;;
        --help|-h)
            show_help
            exit 0
            ;;
        *)
            error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Check dependencies
command -v curl >/dev/null 2>&1 || { error "curl is required but not installed"; exit 1; }
command -v python3 >/dev/null 2>&1 || { error "python3 is required but not installed"; exit 1; }

# Run the test suite
run_test_suite 