# 🚀 MNIST Deployment Pipeline

Pipeline d'orchestration pour le déploiement complet de l'application MNIST avec CI/CD automatisé.

## 📋 Vue d'ensemble

Ce repo coordonne le déploiement de :

- **Backend** : API FastAPI avec modèle PyTorch
- **Frontend** : Interface Streamlit
- **Monitoring** : Prometheus + Grafana
- **CI/CD** : Pipelines GitHub Actions automatisés

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  mnist-backend  │    │ mnist-frontend  │    │mnist-deployment │
│                 │    │                 │    │                 │
│ • FastAPI       │    │ • Streamlit     │    │ • Docker Compose│
│ • PyTorch       │◄──►│ • Interface UI  │◄──►│ • CI/CD         │
│ • Model MNIST   │    │ • Drawing Canvas│    │ • Monitoring    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🚀 Déploiement Rapide

### Option 1: Script automatisé (recommandé)

```bash
# Déploiement en développement
./scripts/deploy.sh dev

# Déploiement en staging
./scripts/deploy.sh staging

# Déploiement en production
./scripts/deploy.sh production --force
```

### Option 2: Docker Compose manuel

```bash
# Développement
make up

# Staging
docker compose -f docker-compose.yml -f docker-compose.staging.yml up -d

# Production
docker compose -f docker-compose.yml -f docker-compose.production.yml up -d
```

## 🧪 Tests et Validation

### Tests d'intégration

```bash
# Tests complets
./scripts/test-integration.sh

# Tests avec paramètres personnalisés
./scripts/test-integration.sh --backend-url http://localhost:8000 --timeout 60

# Tests après déploiement
make test
```

### Tests manuels

```bash
# Santé des services
curl http://localhost:8000/health
curl http://localhost:8501

# Test de prédiction
curl -X POST -H "Content-Type: application/json" \
  -d '{"image": [0.1, 0.2, ...]}' \
  http://localhost:8000/api/v1/predict
```

## 📊 Monitoring

### Accès aux dashboards

- **Application** : http://localhost:8501
- **API Backend** : http://localhost:8000/docs
- **Prometheus** : http://localhost:9090
- **Grafana** : http://localhost:3000 (admin/admin)

### Métriques surveillées

- Performance API (latence, throughput)
- Santé des services
- Utilisation ressources (CPU, mémoire)
- Prédictions ML (précision, temps d'inférence)

## 🔄 Workflow CI/CD

### Déclenchement automatique

**Backend (`mnist-backend`) :**

```
Push → Tests → Build → Security → Deploy Staging → Deploy Production
 ↓       ↓       ↓        ↓           ↓               ↓
Code   Unit    Docker   Trivy      Auto sur       Auto sur
Check  Tests   Image    Scan       develop        main
```

**Frontend (`mnist-frontend`) :**

```
Push → Tests → Lint → Build → A11y → Deploy Staging → Deploy Production
 ↓       ↓      ↓      ↓       ↓           ↓               ↓
Code   Unit   ESLint Docker  Access.    Auto sur       Auto sur
Check  Tests         Image   Tests      develop        main
```

**Orchestration (`mnist-deployment`) :**

```
Manual/API → Integration → Security → Deploy → Notify
    ↓           ↓           ↓          ↓        ↓
Workflow    E2E Tests   Compliance  Target   Teams
Dispatch                           Env
```

### Variables d'environnement GitHub

Configure ces secrets dans ton repo :

```bash
# Registry GitHub
GHCR_TOKEN=<github_token>

# Environnements
STAGING_HOST=<staging_server>
PRODUCTION_HOST=<production_server>

# Monitoring
GRAFANA_PASSWORD=<secure_password>

# Notifications
SLACK_WEBHOOK=<webhook_url>
```

## 📁 Structure du Projet

```
mnist-deployment/
├── .github/
│   └── workflows/
│       └── deploy.yml          # Pipeline d'orchestration
├── scripts/
│   ├── deploy.sh              # Script de déploiement
│   └── test-integration.sh    # Tests d'intégration
├── monitoring/
│   ├── prometheus-staging.yml # Config Prometheus staging
│   └── prometheus-prod.yml    # Config Prometheus production
├── docker-compose.yml         # Base configuration
├── docker-compose.staging.yml # Override staging
├── docker-compose.production.yml # Override production
├── Makefile                   # Commandes simplifiées
└── README.md                  # Cette documentation
```

## 🛠️ Commandes Makefile

```bash
make build     # Build des images
make up        # Démarrage des services
make down      # Arrêt des services
make logs      # Affichage des logs
make test      # Tests d'intégration
make status    # Status des services
make clean     # Nettoyage complet
```

## 🌍 Environnements

### Développement (dev)

- Debugging activé
- Hot reload
- Accès direct aux services
- Logs détaillés

### Staging

- Configuration proche production
- Tests automatisés
- Monitoring basique
- Déploiement automatique sur `develop`

### Production

- Optimisations performance
- High availability (replicas)
- Monitoring complet
- SSL/HTTPS
- Déploiement automatique sur `main`

## 🔐 Sécurité

### Scans automatiques

- **Trivy** : Vulnérabilités containers
- **SAST** : Analyse statique du code
- **Secrets** : Détection de secrets exposés
- **Dependencies** : Audit des dépendances

### Bonnes pratiques

- Images non-root
- Secrets via environment
- Network isolation
- Resource limits
- Health checks

## 📈 Performance

### Optimisations implémentées

- Multi-stage Docker builds
- Layer caching
- Resource limits appropriées
- Load balancing (production)
- Connection pooling

### Métriques cibles

- **API Latency** : < 200ms (p95)
- **Availability** : > 99.9%
- **Resource Usage** : < 80% CPU/RAM
- **Error Rate** : < 0.1%

## 🚨 Troubleshooting

### Problèmes courants

**Services ne démarrent pas :**

```bash
docker compose logs
./scripts/test-integration.sh --timeout 60
```

**Ports déjà utilisés :**

```bash
# Arrêt des services existants
docker compose down -v
lsof -i :8000,8501
```

**Images non trouvées :**

```bash
# Rebuild local
make build

# Pull depuis registry
docker compose pull
```

**Tests échouent :**

```bash
# Debug mode
DEBUG=1 ./scripts/test-integration.sh

# Tests individuels
curl -v http://localhost:8000/health
```

## 📞 Support

### Logs et debugging

```bash
# Logs en temps réel
make logs

# Logs spécifiques
docker compose logs mnist-backend
docker compose logs mnist-frontend

# Status détaillé
make status
```

### Escalation

1. Vérifier logs applicatifs
2. Consulter monitoring Grafana
3. Analyser métriques Prometheus
4. Contacter l'équipe DevOps

---

## 🎯 Prochaines étapes

### Améliorations prévues

- [ ]  Kubernetes manifests
- [ ]  Blue/Green deployment
- [ ]  Automated rollback
- [ ]  Advanced monitoring
- [ ]  Performance testing
- [ ]  Security hardening

### Intégrations futures

- [ ]  Database persistence
- [ ]  API rate limiting
- [ ]  User authentication
- [ ]  Model versioning
- [ ]  A/B testing
- [ ]  Multi-region deployment


# 🚀 MNIST Deployment: Le Hub d'Opérations MLOps

Ce dépôt est le centre de contrôle opérationnel de notre projet MNIST. Il ne contient ni le code du backend, ni celui du frontend, mais l'ensemble des outils et configurations qui permettent de les assembler, de les déployer et de les opérer de manière fiable et automatisée. C'est ici que le "Ops" de "MLOps" prend tout son sens.

## Rôle Central dans l'Architecture MLOps

Ce dépôt est le pivot qui assure notre maturité MLOps de **niveau 2**, en orchestrant le cycle de vie complet de l'application, du build à la production. Sa mission se décompose en quatre piliers fondamentaux :

### 1. 🏗️ Orchestration des Services

Ce n'est pas juste un conteneur, mais un écosystème de services. Les fichiers `docker-compose.*.yml` agissent comme le chef d'orchestre :

- Ils définissent comment les services (`mnist-backend`, `mnist-frontend`, `prometheus`, `grafana`) communiquent et coexistent.
- Ils gèrent les configurations spécifiques à chaque environnement (`dev`, `staging`, `prod`), garantissant que le comportement en local reflète fidèlement celui de la production.

### 2. ⚙️ Infrastructure as Code (IaC)

Nous ne cliquons pas sur des boutons pour déployer. L'infrastructure est définie par le code :

- Les `Dockerfile.*` et les fichiers `docker-compose` spécifient de manière déclarative l'environnement d'exécution.
- **Résultat** : Des déploiements reproductibles, cohérents et moins sujets aux erreurs humaines. N'importe quel membre de l'équipe peut recréer l'environnement complet avec quelques commandes.

### 3. 🔄 Automatisation CI/CD

Ce dépôt est le moteur de notre usine logicielle.

- Le `Makefile` expose des commandes de haut niveau (`make build`, `make up`, `make test`) pour simplifier les actions répétitives.
- Ces commandes sont les briques de base de nos pipelines CI/CD (ex: GitHub Actions). À chaque `push`, les workflows automatisés prennent le relais pour tester, construire les images Docker, les scanner pour des vulnérabilités et les déployer sur l'environnement cible.

### 4. 📊 Monitoring et Observabilité

Déployer c'est bien, savoir ce qui se passe c'est mieux.

- Le dossier `monitoring/` contient la configuration de Prometheus pour scraper les métriques essentielles de nos services.
- Nous passons d'une approche réactive ("ça ne marche plus !") à une approche proactive ("la latence augmente, nous devons scaler avant que les utilisateurs ne soient impactés"). C'est crucial pour la maintenance d'un service ML en production.

## Fichiers clés

- `docker-compose.yml`: Configuration de base pour tous les environnements.
- `docker-compose.dev.yml`: Surcharges pour l'environnement de développement (ex: hot-reloading).
- `docker-compose.staging.yml`: Configuration pour l'environnement de pré-production.
- `docker-compose.production.yml`: Configuration pour l'environnement de production (plus robuste).
- `Dockerfile.serving`: Dockerfile pour le service d'inférence (`mnist-backend`).
- `Dockerfile.training`: Dockerfile pour l'exécution du pipeline d'entraînement Kedro.
- `Makefile`: Raccourcis pour les commandes `docker-compose`.
- `monitoring/`: Configuration pour Prometheus.
