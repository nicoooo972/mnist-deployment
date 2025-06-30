.PHONY: help build up down logs test status clean rebuild dev staging prod deploy-dev deploy-staging deploy-prod

# Couleurs pour les messages
GREEN := \033[0;32m
RED := \033[0;31m
YELLOW := \033[1;33m
BLUE := \033[0;34m
RESET := \033[0m

help: ## Affiche cette aide
	@echo "$(GREEN)🚀 MNIST Deployment Pipeline$(RESET)"
	@echo "Commandes disponibles:"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  $(YELLOW)%-20s$(RESET) %s\n", $$1, $$2}' $(MAKEFILE_LIST)

build: ## Build des images Docker
	@echo "$(GREEN)🔨 Build des images...$(RESET)"
	docker compose build

up: ## Démarre les services (dev)
	@echo "$(GREEN)🚀 Démarrage des services...$(RESET)"
	docker compose up -d
	@echo "$(GREEN)✅ Services démarrés!$(RESET)"
	@echo "Frontend: http://localhost:8501"
	@echo "Backend API: http://localhost:8000"
	@echo "Documentation: http://localhost:8000/docs"

down: ## Arrête les services
	@echo "$(YELLOW)🛑 Arrêt des services...$(RESET)"
	docker compose down

logs: ## Affiche les logs
	docker compose logs -f

test: ## Tests d'intégration complets
	@echo "$(GREEN)🧪 Lancement des tests d'intégration...$(RESET)"
	./scripts/test-integration.sh

test-quick: ## Tests rapides de connectivité
	@echo "$(GREEN)🧪 Tests de connectivité...$(RESET)"
	@echo "Test backend..."
	@curl -sf http://localhost:8000/health > /dev/null && echo "✅ Backend OK" || echo "❌ Backend KO"
	@echo "Test frontend..."
	@curl -sf http://localhost:8501 > /dev/null && echo "✅ Frontend OK" || echo "❌ Frontend KO"

status: ## Statut des services
	@echo "$(GREEN)📊 Statut des services:$(RESET)"
	docker compose ps

clean: ## Nettoyage complet
	@echo "$(RED)🧹 Nettoyage complet...$(RESET)"
	docker compose down -v --remove-orphans
	docker system prune -f

rebuild: clean build up ## Reconstruction complète

# Déploiements avec scripts automatisés
deploy-dev: ## Déploiement développement
	@echo "$(BLUE)🔧 Déploiement développement...$(RESET)"
	./scripts/deploy.sh dev

deploy-staging: ## Déploiement staging
	@echo "$(BLUE)🧪 Déploiement staging...$(RESET)"
	./scripts/deploy.sh staging

deploy-prod: ## Déploiement production
	@echo "$(BLUE)🏭 Déploiement production...$(RESET)"
	./scripts/deploy.sh production

# Alias pour compatibilité
dev: ## Mode développement avec builds locaux

dev-down: ## Arrête les services de développement
	@echo "$(YELLOW)🛑 Arrêt des services dev...$(RESET)"
	docker compose -f docker-compose.yml -f docker-compose.dev.yml down
	@echo "$(BLUE)🔧 Mode développement...$(RESET)"
	docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d
	@echo "$(GREEN)✅ Services dev démarrés!$(RESET)"
	@echo "Frontend: http://localhost:8501"
	@echo "Backend API: http://localhost:8000"
staging: deploy-staging ## Alias pour deploy-staging  
prod: deploy-prod ## Alias pour deploy-prod

# Monitoring
monitoring: ## Démarre uniquement les services de monitoring
	@echo "$(GREEN)📊 Démarrage monitoring...$(RESET)"
	docker compose up -d prometheus grafana
	@echo "Prometheus: http://localhost:9090"
	@echo "Grafana: http://localhost:3000"

# Gestion des logs
logs-backend: ## Logs du backend uniquement
	docker compose logs -f mnist-backend

logs-frontend: ## Logs du frontend uniquement
	docker compose logs -f mnist-frontend

logs-monitoring: ## Logs du monitoring
	docker compose logs -f prometheus grafana

# Utilitaires
shell-backend: ## Shell dans le container backend
	docker compose exec mnist-backend bash

shell-frontend: ## Shell dans le container frontend
	docker compose exec mnist-frontend bash

backup: ## Sauvegarde des données
	@echo "$(GREEN)💾 Sauvegarde des données...$(RESET)"
	mkdir -p backups/$(shell date +%Y%m%d_%H%M%S)
	docker compose exec mnist-backend tar czf - /app/logs 2>/dev/null | tar xzf - -C backups/$(shell date +%Y%m%d_%H%M%S)/ || echo "Pas de logs backend"

# Tests de performance
load-test: ## Test de charge basique
	@echo "$(GREEN)⚡ Test de charge...$(RESET)"
	for i in {1..50}; do curl -s http://localhost:8000/health > /dev/null & done; wait
	@echo "✅ Test de charge terminé"
