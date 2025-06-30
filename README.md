# 🐳 MNIST Deployment

Repo de déploiement Docker pour l'application MNIST (Backend FastAPI + Frontend Streamlit).

## 🚀 Démarrage rapide

```bash
# Aide
make help

# Construction et démarrage
make build
make up

# Voir les logs
make logs

# Arrêter
make down
```

## 🎯 URLs d'accès

- **Frontend** : http://localhost:8501
- **Backend API** : http://localhost:8000
- **Documentation API** : http://localhost:8000/docs

## 📋 Prérequis

Les repos `mnist-backend` et `mnist-frontend` doivent être au même niveau :

```
parent-directory/
├── mnist-backend/     # Repo du backend FastAPI
├── mnist-frontend/    # Repo du frontend Streamlit  
└── mnist-deployment/  # Ce repo (orchestration)
```
