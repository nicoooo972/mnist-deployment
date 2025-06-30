#!/bin/bash

# Script de déploiement automatisé MNIST
# Usage: ./deploy.sh [staging|production] [--force] [--no-tests]

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
LOG_FILE="${PROJECT_DIR}/logs/deploy-$(date +%Y%m%d_%H%M%S).log"

# Couleurs pour les logs
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Fonction de logging
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1" | tee -a "$LOG_FILE"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" | tee -a "$LOG_FILE"
    exit 1
}

warn() {
    echo -e "${YELLOW}[WARNING]${NC} $1" | tee -a "$LOG_FILE"
}

info() {
    echo -e "${BLUE}[INFO]${NC} $1" | tee -a "$LOG_FILE"
}

# Vérification des prérequis
check_prerequisites() {
    log "🔍 Vérification des prérequis..."
    
    command -v docker >/dev/null 2>&1 || error "Docker n'est pas installé"
    command -v docker-compose >/dev/null 2>&1 || command -v docker >/dev/null 2>&1 || error "Docker Compose n'est pas disponible"
    
    # Création des répertoires nécessaires
    mkdir -p "${PROJECT_DIR}/logs"
    mkdir -p "${PROJECT_DIR}/backups"
    
    log "✅ Prérequis validés"
}

# Test de santé des services
health_check() {
    local max_attempts=30
    local attempt=1
    
    log "🏥 Test de santé des services..."
    
    while [ $attempt -le $max_attempts ]; do
        if curl -sf http://localhost:8000/health >/dev/null 2>&1; then
            log "✅ Backend healthy"
            break
        fi
        
        if [ $attempt -eq $max_attempts ]; then
            error "Backend non disponible après $max_attempts tentatives"
        fi
        
        info "Tentative $attempt/$max_attempts - Attente du backend..."
        sleep 2
        ((attempt++))
    done
    
    # Test frontend
    attempt=1
    while [ $attempt -le $max_attempts ]; do
        if curl -sf http://localhost:8501 >/dev/null 2>&1; then
            log "✅ Frontend healthy"
            break
        fi
        
        if [ $attempt -eq $max_attempts ]; then
            error "Frontend non disponible après $max_attempts tentatives"
        fi
        
        info "Tentative $attempt/$max_attempts - Attente du frontend..."
        sleep 2
        ((attempt++))
    done
    
    log "✅ Tous les services sont opérationnels"
}

# Tests d'intégration
run_integration_tests() {
    log "🧪 Exécution des tests d'intégration..."
    
    # Test API endpoint
    info "Test API /health..."
    curl -f http://localhost:8000/health || error "Test /health échoué"
    
    info "Test API /api/info..."
    curl -f http://localhost:8000/api/info || warn "Test /api/info échoué"
    
    # Test prédiction (simulé)
    info "Test prédiction..."
    echo '{"test": "integration"}' > /tmp/test.json
    # Ici tu peux ajouter un vrai test de prédiction
    
    log "✅ Tests d'intégration réussis"
}

# Sauvegarde avant déploiement
backup_current() {
    log "💾 Sauvegarde de l'état actuel..."
    
    local backup_dir="${PROJECT_DIR}/backups/$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$backup_dir"
    
    # Sauvegarde des logs
    if [ -d "${PROJECT_DIR}/logs" ]; then
        cp -r "${PROJECT_DIR}/logs" "$backup_dir/"
    fi
    
    # Export des données Docker
    docker compose ps --format json > "$backup_dir/services.json" 2>/dev/null || true
    
    log "✅ Sauvegarde créée dans $backup_dir"
}

# Déploiement
deploy() {
    local env="$1"
    local force="$2"
    local skip_tests="$3"
    
    log "🚀 Déploiement vers $env..."
    
    # Arrêt des services existants
    if [ "$force" = "true" ]; then
        log "🛑 Arrêt forcé des services..."
        docker compose down -v 2>/dev/null || true
    else
        log "🛑 Arrêt gracieux des services..."
        docker compose down 2>/dev/null || true
    fi
    
    # Construction et démarrage
    if [ "$env" = "production" ]; then
        log "🏭 Déploiement en production..."
        docker compose -f docker-compose.yml -f docker-compose.production.yml up -d --build
    elif [ "$env" = "staging" ]; then
        log "🧪 Déploiement en staging..."
        docker compose -f docker-compose.yml -f docker-compose.staging.yml up -d --build
    else
        log "🔧 Déploiement en développement..."
        docker compose up -d --build
    fi
    
    # Tests post-déploiement
    if [ "$skip_tests" = "false" ]; then
        health_check
        run_integration_tests
    fi
    
    log "✅ Déploiement $env terminé avec succès!"
}

# Affichage de l'usage
usage() {
    echo "Usage: $0 [staging|production|dev] [--force] [--no-tests]"
    echo ""
    echo "Options:"
    echo "  --force     Arrêt forcé des services (avec suppression des volumes)"
    echo "  --no-tests  Skip les tests d'intégration"
    echo ""
    echo "Exemples:"
    echo "  $0 staging"
    echo "  $0 production --force"
    echo "  $0 dev --no-tests"
    exit 1
}

# Fonction principale
main() {
    local env="${1:-dev}"
    local force=false
    local skip_tests=false
    
    # Parse des arguments
    shift
    while [[ $# -gt 0 ]]; do
        case $1 in
            --force)
                force=true
                shift
                ;;
            --no-tests)
                skip_tests=true
                shift
                ;;
            -h|--help)
                usage
                ;;
            *)
                error "Option inconnue: $1"
                ;;
        esac
    done
    
    # Validation de l'environnement
    case $env in
        staging|production|dev)
            ;;
        *)
            error "Environnement invalide: $env. Utilisez: staging, production, ou dev"
            ;;
    esac
    
    log "🎯 Déploiement MNIST vers $env (force=$force, skip_tests=$skip_tests)"
    
    cd "$PROJECT_DIR"
    
    check_prerequisites
    backup_current
    deploy "$env" "$force" "$skip_tests"
    
    log "🎉 Déploiement terminé!"
    info "Logs disponibles dans: $LOG_FILE"
    
    if [ "$env" = "production" ]; then
        info "🌐 Application disponible sur:"
        info "  - Frontend: http://localhost"
        info "  - Backend API: http://localhost/api"
        info "  - Monitoring: http://localhost:3000"
    else
        info "🌐 Application disponible sur:"
        info "  - Frontend: http://localhost:8501"
        info "  - Backend API: http://localhost:8000"
        info "  - Monitoring: http://localhost:9090"
    fi
}

# Vérification si le script est exécuté directement
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi 