.PHONY: help build up down dev logs clean test

help: ## Afficher l'aide
	@echo "🐳 MNIST Deployment - Commandes disponibles :"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

build: ## Construire les images Docker
	docker-compose build

up: ## Démarrer les services
	docker-compose up -d

down: ## Arrêter les services
	docker-compose down

logs: ## Voir les logs en temps réel
	docker-compose logs -f

test: ## Tester si les services répondent
	@echo "🧪 Test du backend..."
	@curl -s http://localhost:8000/health || echo "❌ Backend non disponible"
	@echo "🧪 Test du frontend..."
	@curl -s http://localhost:8501 > /dev/null && echo "✅ Frontend accessible" || echo "❌ Frontend non disponible"

clean: ## Nettoyer
	docker-compose down -v --rmi all

status: ## Statut des services
	docker-compose ps
