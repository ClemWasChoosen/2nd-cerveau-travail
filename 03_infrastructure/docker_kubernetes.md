# Docker et Kubernetes : conteneurisation et orchestration

> **Dernière mise à jour : 12 octobre 2026**  
> **Niveau :** intermédiaire  
> **Domaine :** infrastructure, architecture, déploiement, MLOps  
> **Tags :** `#docker` `#kubernetes` `#containers` `#orchestration` `#devops` `#mlops`

## Table des matières

0. [Avant de commencer](#0-avant-de-commencer)  
1. [Le problème : rendre une application portable et exploitable](#1-le-problème--rendre-une-application-portable-et-exploitable)  
2. [Le modèle mental des conteneurs](#2-le-modèle-mental-des-conteneurs)  
3. [Docker : images, conteneurs et cycle de vie](#3-docker--images-conteneurs-et-cycle-de-vie)  
4. [Construire une image avec un Dockerfile](#4-construire-une-image-avec-un-dockerfile)  
5. [Stockage, réseau et configuration avec Docker](#5-stockage-réseau-et-configuration-avec-docker)  
6. [Docker Compose pour les environnements multi-services](#6-docker-compose-pour-les-environnements-multi-services)  
7. [Distribuer, sécuriser et diagnostiquer des images](#7-distribuer-sécuriser-et-diagnostiquer-des-images)  
8. [Pourquoi Kubernetes ?](#8-pourquoi-kubernetes-)  
9. [Architecture et modèle déclaratif de Kubernetes](#9-architecture-et-modèle-déclaratif-de-kubernetes)  
10. [Les objets Kubernetes essentiels](#10-les-objets-kubernetes-essentiels)  
11. [Configuration, secrets, stockage et santé](#11-configuration-secrets-stockage-et-santé)  
12. [Ressources, scheduling et mise à l'échelle](#12-ressources-scheduling-et-mise-à-léchelle)  
13. [Exposer une application et gérer les déploiements](#13-exposer-une-application-et-gérer-les-déploiements)  
14. [Fil rouge : de l'API locale au cluster](#14-fil-rouge--de-lapi-locale-au-cluster)  
15. [Guide de choix](#15-guide-de-choix)  
16. [Pense-bête des commandes](#16-pense-bête-des-commandes)  
17. [Pièges courants](#17-pièges-courants)  
18. [Exercices d'auto-évaluation](#18-exercices-dauto-évaluation)  
19. [Sources et références](#19-sources-et-références)  
20. [Liens connexes](#20-liens-connexes)

---

## 0. Avant de commencer

### Prérequis

- **Linux et ligne de commande :** savoir naviguer dans un terminal, gérer des processus et lire des logs.
- **Réseau :** comprendre la différence entre une adresse IP, un port, un nom DNS et une route HTTP.
- **Python ou un langage applicatif :** savoir lancer une petite API localement.
- **Git :** comprendre le principe d'un dépôt, d'un commit et d'une version reproductible.
- **Architecture logicielle :** distinguer une application, une base de données, un service et une configuration.

Ce cours ne suppose pas de connaître un fournisseur cloud particulier. Les commandes Kubernetes supposent toutefois l'existence d'un cluster local ou distant, par exemple avec `kind`, `minikube`, Docker Desktop ou un cluster administré.

### Objectifs d'apprentissage

| Niveau | Objectif |
|---|---|
| **Comprendre** | Expliquer la différence entre une image, un conteneur, un Pod, un Service et un Deployment. |
| **Appliquer** | Écrire un Dockerfile reproductible et lancer une application avec Docker Compose. |
| **Appliquer** | Déployer une API sur Kubernetes avec un Deployment, un Service, une ConfigMap et des probes. |
| **Analyser** | Diagnostiquer une image qui ne démarre pas, un Pod non prêt ou un Service inaccessible. |
| **Évaluer** | Choisir entre Docker seul, Docker Compose et Kubernetes selon le contexte opérationnel. |
| **Créer** | Concevoir le chemin complet d'une application, du code local à un déploiement orchestré. |

### TL;DR en 3 phrases

> Docker empaquette une application et ses dépendances dans une image exécutable de manière reproductible. Kubernetes orchestre des conteneurs regroupés dans des Pods, maintient l'état désiré et fournit déploiement, réseau, configuration et mise à l'échelle. Le workflow pratique consiste à construire une image, la tester localement, la publier dans un registre, puis la référencer dans des manifests Kubernetes déclaratifs.

---

## 1. Le problème : rendre une application portable et exploitable

### 1.1 Analogie : transporter une application comme un colis standardisé

Imaginez une application de prédiction qui fonctionne sur votre ordinateur mais échoue sur un serveur. Il manque une version de Python, une librairie système, une variable d'environnement ou un service auxiliaire. Le code est identique, mais son environnement ne l'est pas.

- **Approche naïve :** installer manuellement chaque dépendance sur chaque machine.
- **Approche conteneurisée :** construire un artefact versionné qui contient l'application, ses dépendances et les instructions de démarrage.
- **Approche orchestrée :** demander à une plateforme de maintenir plusieurs instances de cet artefact, de les rendre accessibles et de les remplacer en cas de panne.

Docker répond principalement au deuxième besoin. Kubernetes répond principalement au troisième.

### 1.2 Le problème formalisé

Une application déployable doit répondre à plusieurs contraintes :

| Contrainte | Question à résoudre |
|---|---|
| Reproductibilité | Le même code produit-il le même environnement ? |
| Isolation | Les dépendances de deux services peuvent-elles coexister sans conflit ? |
| Distribution | Comment transporter l'application vers une autre machine ? |
| Disponibilité | Que se passe-t-il si un processus ou un serveur tombe ? |
| Configuration | Comment changer un secret ou une URL sans reconstruire l'image ? |
| Montée en charge | Comment lancer plusieurs instances ? |
| Diagnostic | Comment savoir pourquoi une instance ne répond pas ? |

### 1.3 Le parcours complet

```text
Code source
    │
    ▼
Dockerfile + contexte de build
    │ docker build
    ▼
Image versionnée
    │ docker push
    ▼
Registre d'images
    │ pull
    ▼
Conteneur local ou runtime de nœud
    │
    ├── Docker Compose : plusieurs services sur une machine
    │
    └── Kubernetes : Pods, Services, contrôle de l'état désiré
```

> 📌 **Points clés, section 1**
> - Docker traite principalement le packaging et l'exécution d'applications.
> - Kubernetes traite principalement l'orchestration et la convergence vers un état désiré.
> - Une image est un artefact distribué, pas un processus en fonctionnement.

---

## 2. Le modèle mental des conteneurs

### 2.1 Intuition

Un conteneur est un processus isolé lancé à partir d'une image. Il ne s'agit pas d'une machine virtuelle complète : le conteneur partage le noyau de l'hôte, ce qui le rend généralement plus léger, mais implique une frontière de sécurité différente.

```text
Image immuable
      │ docker run
      ▼
Conteneur = processus + configuration d'exécution + couche writable
      │
      ├── suppression du conteneur : perte de la couche writable
      └── volume ou stockage externe : conservation des données
```

### 2.2 Image, conteneur, registre

| Concept | Rôle | Exemple mental |
|---|---|---|
| **Image** | Artefact immuable contenant le système de fichiers et le point d'entrée | Une recette empaquetée |
| **Conteneur** | Instance en fonctionnement d'une image | Un plat préparé |
| **Registre** | Service qui stocke et distribue des images | Un entrepôt |
| **Tag** | Nom lisible qui référence une version | `my-api:1.4` |
| **Digest** | Identifiant cryptographique du contenu exact | `sha256:...` |
| **Volume** | Stockage persistant géré hors de la couche éphémère | Un disque externe |

Un tag est pratique mais peut être déplacé vers une autre image. Un digest référence le contenu exact. Pour une production reproductible, il est préférable de contrôler les tags et, lorsque le contexte l'exige, de verrouiller les images par digest.

### 2.3 Les standards sous-jacents

Il faut distinguer les produits et les standards :

```text
Docker / BuildKit
    └── construit et distribue des images
OCI
    ├── décrit le format des images
    ├── décrit leur distribution
    └── décrit l'exécution d'un bundle
containerd / CRI-O
    └── gèrent le cycle de vie des conteneurs sur un nœud
runc / runtime OCI
    └── exécutent effectivement les processus isolés
CRI
    └── interface entre kubelet et runtime de conteneurs
Kubernetes
    └── orchestre les Pods, Services, volumes et états désirés
```

Dire que « Kubernetes utilise Docker » est trop vague. Kubernetes dialogue avec un runtime compatible CRI, souvent `containerd` ou `CRI-O`. Docker peut construire l'image utilisée ensuite par Kubernetes, mais Docker Engine n'est pas l'orchestrateur Kubernetes.

> 📌 **Points clés, section 2**
> - L'image est l'artefact, le conteneur est le processus.
> - La couche writable d'un conteneur est éphémère par défaut.
> - OCI, containerd, CRI, runc, Docker et Kubernetes désignent des couches différentes.

---

## 3. Docker : images, conteneurs et cycle de vie

### 3.1 Le cycle minimal

Le cycle Docker courant est :

```text
pull ou build
      │
      ▼
create / run
      │
      ├── logs / inspect / exec / stats
      │
      ▼
stop / restart
      │
      ▼
rm
```

Les commandes fondamentales sont :

```bash
# Vérifier la connexion au daemon Docker
docker version

# Télécharger une image
docker pull python:3.12-slim

# Lancer un conteneur en arrière-plan, avec publication d'un port
docker run --detach --name demo-api --publish 8080:8000 python:3.12-slim

# Lister les conteneurs en fonctionnement
docker ps

# Lister tous les conteneurs, y compris ceux arrêtés
docker ps --all

# Lire les logs
docker logs --follow demo-api

# Arrêter puis supprimer le conteneur
docker stop demo-api
docker rm demo-api
```

### 3.2 Processus principal et signaux

Un conteneur doit avoir un processus principal clair. Lorsque ce processus se termine, le conteneur se termine généralement aussi. Pour les applications longues, le processus doit recevoir correctement les signaux `SIGTERM` et `SIGINT`, afin de pouvoir fermer ses connexions et écrire son état avant l'arrêt.

Préférer la forme exec dans un Dockerfile :

```dockerfile
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

La forme shell suivante ajoute une couche d'interprétation et peut compliquer la transmission des signaux :

```dockerfile
CMD uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### 3.3 Ports : `EXPOSE` n'est pas une publication

`EXPOSE` documente le port écouté par l'application dans l'image. Il ne rend pas automatiquement ce port accessible depuis l'hôte.

```dockerfile
EXPOSE 8000
```

La publication est décidée au lancement :

```bash
docker run --publish 8080:8000 my-api:1.0
```

La syntaxe signifie :

```text
port_hôte:port_conteneur
8080:8000
```

> 📌 **Points clés, section 3**
> - `docker run` crée et démarre un conteneur, mais ne rend pas les données persistantes.
> - Les logs et l'inspection sont les premiers outils de diagnostic.
> - `EXPOSE` documente un port, `--publish` le rend accessible depuis l'hôte.

---

## 4. Construire une image avec un Dockerfile

### 4.1 Intuition

Un Dockerfile est une recette de construction. Chaque instruction contribue à l'image finale et peut participer au cache de build. L'objectif n'est pas seulement de « faire marcher » l'image, mais de produire un artefact petit, lisible, reproductible, sécurisé et facile à diagnostiquer.

### 4.2 Exemple fil rouge : une API de prédiction minimale

Organisation proposée :

```text
ml-api/
├── app/
│   ├── __init__.py
│   └── main.py
├── requirements.txt
├── Dockerfile
├── .dockerignore
└── compose.yaml
```

Application minimale :

```python
# app/main.py
from fastapi import FastAPI

app = FastAPI(title="ML API")


@app.get("/")
def root() -> dict[str, str]:
    """Réponse fonctionnelle minimale pour tester le service."""
    return {"service": "ml-api", "status": "ok"}


@app.get("/healthz")
def healthz() -> dict[str, str]:
    """Indique que le processus répond encore aux requêtes."""
    return {"status": "alive"}


@app.get("/readyz")
def readyz() -> dict[str, str]:
    """Indique que les dépendances nécessaires sont disponibles."""
    return {"status": "ready"}
```

Dépendances :

```text
fastapi==0.115.6
uvicorn[standard]==0.34.0
```

Les versions exactes doivent être adaptées au projet et verrouillées dans un fichier de dépendances versionné.

### 4.3 Dockerfile de production simple

```dockerfile
# syntax=docker/dockerfile:1

FROM python:3.12-slim

# Évite les fichiers .pyc et permet de voir les logs immédiatement.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Installer les dépendances avant le code pour maximiser le cache.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copier uniquement le code nécessaire à l'exécution.
COPY app/ ./app/

# Exécuter l'application avec un utilisateur non privilégié.
RUN useradd --create-home --uid 10001 appuser
USER 10001

# Documenter le port interne, sans le publier automatiquement.
EXPOSE 8000

# Utiliser la forme exec pour une gestion correcte des signaux.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 4.4 Le contexte de build et `.dockerignore`

Le contexte de build est l'ensemble de fichiers transmis au builder. Un contexte trop large ralentit le build et risque d'inclure des secrets ou des artefacts inutiles.

```text
.git
.gitignore
__pycache__/
*.pyc
.venv/
.env
.env.*
.pytest_cache/
.mypy_cache/
.ipynb_checkpoints/
models/
data/
logs/
```

Construire l'image :

```bash
docker build --tag ml-api:1.0 .
```

Le point final indique que le contexte de build est le répertoire courant. Pour vérifier la configuration du build lorsque la version de Docker le permet :

```bash
docker build --check .
```

### 4.5 Cache et ordre des instructions

Le cache est invalidé à partir de l'étape qui change. Les fichiers de dépendances changent généralement moins souvent que le code applicatif. Il est donc logique de les copier séparément avant le code.

```text
Mauvais ordre :
COPY . .
RUN pip install ...

Bon ordre :
COPY requirements.txt .
RUN pip install ...
COPY app/ ./app/
```

### 4.6 Multi-stage build

Un build multi-stage utilise plusieurs images intermédiaires et ne conserve dans l'image finale que les artefacts nécessaires. C'est surtout utile lorsque la compilation nécessite des outils qui ne doivent pas être présents à l'exécution.

```dockerfile
FROM golang:1.26 AS build
WORKDIR /src
COPY . .
RUN go build -o /out/app ./cmd/app

FROM gcr.io/distroless/static-debian12
COPY --from=build /out/app /app
ENTRYPOINT ["/app"]
```

Les bénéfices sont la réduction de taille, une surface d'attaque plus petite et une séparation claire entre build et runtime. Les images minimales rendent toutefois le diagnostic interactif plus difficile. Il faut donc prévoir des images de debug ou des outils externes adaptés.

### 4.7 `ARG`, `ENV`, `CMD` et `ENTRYPOINT`

| Instruction | Moment principal | Usage | Erreur fréquente |
|---|---|---|---|
| `ARG` | Build | Paramètre de construction | Y placer un secret |
| `ENV` | Build et runtime | Configuration disponible dans l'image ou le conteneur | Croire qu'une valeur est secrète |
| `CMD` | Runtime | Valeur par défaut remplaçable | L'utiliser comme mécanisme de configuration |
| `ENTRYPOINT` | Runtime | Exécutable principal | Empêcher les usages de debug sans le prévoir |
| `USER` | Runtime | Réduire les privilèges | Laisser l'application en root sans raison |
| `HEALTHCHECK` | Runtime | Décrire un test de santé Docker | Confondre santé et disponibilité métier |

Les secrets de build ne doivent pas être injectés dans `ARG` ou `ENV`. Utiliser les mécanismes de secrets du builder ou une gestion externe des secrets.

> 📌 **Points clés, section 4**
> - Le Dockerfile doit être optimisé pour la reproductibilité, le cache et la sécurité.
> - `.dockerignore`, dépendances verrouillées, utilisateur non root et forme exec sont des bases solides.
> - Un secret ne doit pas se retrouver dans une couche ou une variable persistante de l'image.

---

## 5. Stockage, réseau et configuration avec Docker

### 5.1 Couche writable, volumes et bind mounts

Par défaut, les écritures réalisées dans le système de fichiers du conteneur appartiennent à sa couche writable. Elles disparaissent avec le conteneur.

| Mécanisme | Contrôle principal | Usage recommandé |
|---|---|---|
| Couche writable | Docker et cycle de vie du conteneur | Cache temporaire, fichiers éphémères |
| Volume nommé | Docker | Données persistantes gérées par Docker |
| Bind mount | Hôte | Développement local, partage explicite d'un dossier |
| `tmpfs` | Mémoire de l'hôte | Données temporaires sensibles ou rapides |

Volume nommé :

```bash
# Créer un volume
docker volume create model-cache

# Monter le volume dans un conteneur
docker run --detach \
  --name api \
  --mount source=model-cache,target=/app/cache \
  ml-api:1.0

# Inspecter le volume
docker volume inspect model-cache
```

Bind mount de développement :

```bash
docker run --rm --interactive --tty \
  --mount type=bind,source="$PWD",target=/workspace \
  python:3.12-slim bash
```

Un bind mount dépend fortement de la structure et des permissions de l'hôte. Il est pratique en développement, mais un volume ou un service de stockage externe est généralement plus adapté pour la persistance en production.

### 5.2 Réseaux Docker

Les conteneurs présents sur un même réseau Docker user-defined peuvent se joindre par nom de service. Il n'est généralement pas nécessaire de coder en dur leurs adresses IP.

```bash
# Créer un réseau
docker network create app-net

# Lancer un service sur ce réseau
docker run --detach --name redis --network app-net redis:7

# Le second conteneur peut utiliser redis comme nom DNS
docker run --rm --network app-net redis:7 redis-cli -h redis ping
```

La publication de ports sert à rendre un service accessible depuis l'extérieur du réseau Docker. La communication interne entre services peut rester non publiée.

### 5.3 Variables d'environnement

```bash
docker run --rm \
  --env MODEL_NAME=classifier-v1 \
  --env LOG_LEVEL=info \
  ml-api:1.0
```

Pour plusieurs variables :

```text
# .env.local, ne pas committer si le fichier contient des secrets
MODEL_NAME=classifier-v1
LOG_LEVEL=info
```

```bash
docker run --rm --env-file .env.local ml-api:1.0
```

Une variable d'environnement est une configuration pratique, pas un coffre-fort. Pour un secret critique, utiliser un gestionnaire de secrets ou un mécanisme de secret adapté à la plateforme.

> 📌 **Points clés, section 5**
> - Les données importantes ne doivent pas dépendre de la couche writable éphémère.
> - Les volumes et bind mounts ont des usages différents.
> - Les services doivent communiquer par nom, pas par IP codée en dur.

---

## 6. Docker Compose pour les environnements multi-services

### 6.1 Intuition

Une application réelle comprend souvent plusieurs services : API, base de données, cache, worker, observabilité. Lancer chaque conteneur avec une longue commande `docker run` devient vite fragile. Compose décrit l'ensemble dans un fichier YAML versionné.

Compose est particulièrement utile pour :

- reproduire un environnement local ;
- démarrer plusieurs services avec une seule commande ;
- partager une configuration de développement ;
- tester les interactions entre services ;
- préparer un environnement de démonstration ou d'intégration.

Compose n'est pas un remplacement général de Kubernetes pour les besoins de haute disponibilité multi-nœuds.

### 6.2 Exemple `compose.yaml`

```yaml
services:
  api:
    build:
      context: .
    image: ml-api:dev
    ports:
      - "8000:8000"
    environment:
      MODEL_NAME: classifier-v1
      REDIS_URL: redis://redis:6379/0
    depends_on:
      redis:
        condition: service_healthy
    healthcheck:
      test: ["CMD", "python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8000/healthz')"]
      interval: 10s
      timeout: 3s
      retries: 5

  redis:
    image: redis:7
    volumes:
      - redis-data:/data
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 5s
      timeout: 3s
      retries: 10

volumes:
  redis-data:
```

L'application doit utiliser `redis` comme nom d'hôte, car Compose fournit une résolution DNS entre services.

### 6.3 Commandes Compose essentielles

```bash
# Construire et démarrer les services
docker compose up --build --detach

# Voir l'état des services
docker compose ps

# Suivre les logs de l'API
docker compose logs --follow api

# Exécuter une commande dans l'API
docker compose exec api sh

# Arrêter les services sans supprimer les volumes
docker compose stop

# Arrêter et supprimer les conteneurs et réseaux créés
docker compose down

# Arrêter et supprimer aussi les volumes déclarés
# Attention aux données persistantes
docker compose down --volumes
```

### 6.4 `depends_on` et readiness

`depends_on` exprime un ordre de démarrage ou une condition configurée. Il ne prouve pas qu'une dépendance est réellement prête dans tous les scénarios. Les healthchecks et les retry applicatifs restent importants.

```text
Démarré ≠ prêt à recevoir du trafic

processus lancé
      │
      ├── port ouvert
      │
      ├── dépendances connectées
      │
      └── application réellement prête
```

> 📌 **Points clés, section 6**
> - Compose décrit un environnement multi-services local et reproductible.
> - Le nom du service sert de nom DNS entre conteneurs.
> - L'ordre de démarrage ne suffit pas à garantir la disponibilité fonctionnelle.

---

## 7. Distribuer, sécuriser et diagnostiquer des images

### 7.1 Registres, tags et digests

Un registre stocke et distribue des images. Le chemin d'une image suit généralement ce modèle :

```text
registry.example.com/organisation/projet:version
```

Commandes :

```bash
# Se connecter à un registre
docker login registry.example.com

# Ajouter un tag compatible avec le registre
docker tag ml-api:1.0 registry.example.com/team/ml-api:1.0

# Publier l'image
docker push registry.example.com/team/ml-api:1.0

# Tirer l'image sur une autre machine
docker pull registry.example.com/team/ml-api:1.0
```

Bonnes pratiques :

- éviter `latest` pour les déploiements reproductibles ;
- utiliser des tags immuables ou des digests ;
- associer une image à un commit ou une version de release ;
- conserver les métadonnées de build et les attestations utiles ;
- scanner les dépendances et l'image avant publication ;
- limiter les droits d'écriture sur le registre.

### 7.2 Sécurité de l'image

Une image sûre dépend du Dockerfile, de l'image de base, des dépendances et du runtime. Les mesures importantes sont :

1. utiliser une image de base maintenue et adaptée au besoin ;
2. réduire le nombre de paquets installés ;
3. exécuter l'application sans privilèges root ;
4. ne jamais inclure de secret dans le contexte ou les couches ;
5. limiter les capacités et les montages lorsque c'est possible ;
6. contrôler les images par scan et par provenance ;
7. protéger l'accès au socket Docker, qui peut donner des privilèges élevés sur l'hôte.

Le mode rootless peut réduire certains risques, mais il introduit aussi des contraintes de configuration et de compatibilité. Il ne remplace pas une politique complète de sécurité.

### 7.3 Diagnostic Docker

```bash
# Résumé détaillé d'un conteneur
docker inspect api

# Statistiques de ressources
docker stats api

# Processus visibles dans le conteneur
docker top api

# Événements Docker en temps réel
docker events

# Espace disque utilisé
docker system df

# Afficher l'historique des couches d'une image
docker history ml-api:1.0

# Ouvrir un shell dans un conteneur en fonctionnement
docker exec --interactive --tty api sh
```

Nettoyage prudent :

```bash
# Supprimer les conteneurs arrêtés, réseaux inutilisés et caches non référencés
docker system prune
```

Ne pas exécuter aveuglément `docker system prune --all --volumes` sur une machine importante. Cette commande peut supprimer des images et volumes utiles.

> 📌 **Points clés, section 7**
> - Un tag lisible n'est pas nécessairement une référence immuable.
> - Les secrets doivent rester hors des images et du contexte de build.
> - `inspect`, `logs`, `stats`, `events` et `system df` forment une première boîte à outils de diagnostic.

---

## 8. Pourquoi Kubernetes ?

### 8.1 La limite de Docker seul

Docker seul fonctionne très bien pour un service local ou quelques conteneurs sur une machine. Les problèmes apparaissent lorsque l'on doit gérer plusieurs machines et plusieurs versions :

- maintenir plusieurs réplicas ;
- remplacer automatiquement un processus défaillant ;
- répartir les Pods sur plusieurs nœuds ;
- exposer un service malgré le remplacement de ses instances ;
- effectuer un rolling update ;
- appliquer des limites de ressources ;
- centraliser la configuration et les secrets ;
- diagnostiquer des workloads distribués.

Kubernetes ajoute une API, un control plane, des contrôleurs, un scheduler et des abstractions déclaratives pour résoudre ces problèmes.

### 8.2 Le modèle déclaratif

L'utilisateur décrit un état souhaité. Kubernetes observe l'état courant et tente de réduire l'écart entre les deux.

```text
spec : état désiré
        │
        ▼
API Kubernetes
        │
        ▼
Controllers + Scheduler
        │
        ▼
État courant des Pods et des Services
        │
        └── boucle de réconciliation continue
```

Un manifeste n'est donc pas simplement une suite de commandes « lancer ceci puis cela ». Il décrit ce qui doit rester vrai.

### 8.3 Quand Kubernetes est pertinent

Kubernetes devient pertinent lorsque l'équipe doit gérer de façon durable :

- plusieurs services ou workloads ;
- plusieurs nœuds ;
- déploiements progressifs et rollbacks ;
- contraintes de disponibilité ;
- isolation par namespaces ;
- quotas, limites et politiques d'accès ;
- scaling et scheduling spécialisés, par exemple GPU.

Il ajoute toutefois une forte complexité opérationnelle. Un cluster Kubernetes mal observé ou mal configuré peut être plus difficile à exploiter qu'un simple déploiement Docker.

> 📌 **Points clés, section 8**
> - Kubernetes n'est pas nécessaire pour chaque application conteneurisée.
> - Sa valeur vient de l'orchestration, de la réconciliation et des abstractions opérationnelles.
> - La complexité de Kubernetes doit être justifiée par les besoins de la plateforme.

---

## 9. Architecture et modèle déclaratif de Kubernetes

### 9.1 Architecture d'un cluster

```text
                        kubectl / clients / contrôleurs externes
                                      │
                                      ▼
┌──────────────────────────────────────────────────────────────┐
│                         Control plane                       │
│                                                              │
│  kube-apiserver  ◄──►  etcd                                 │
│        │              état persistant du cluster             │
│        ├── controllers                                        │
│        └── scheduler                                           │
└──────────────────────────────┬───────────────────────────────┘
                               │
              décisions, état désiré, événements
                               │
        ┌──────────────────────┴──────────────────────┐
        │                                             │
┌───────▼────────┐                            ┌───────▼────────┐
│ Node           │                            │ Node           │
│ kubelet        │                            │ kubelet        │
│ runtime CRI    │                            │ runtime CRI    │
│ réseau         │                            │ réseau         │
│ Pods           │                            │ Pods           │
└────────────────┘                            └────────────────┘
```

| Composant | Responsabilité |
|---|---|
| `kube-apiserver` | Façade de l'API, validation et point d'entrée des objets |
| `etcd` | Stockage de l'état du cluster |
| `scheduler` | Association des Pods non planifiés à des nœuds |
| Controllers | Convergence entre état désiré et état courant |
| `kubelet` | Exécution des PodSpecs sur un nœud |
| Runtime CRI | Gestion des conteneurs sur un nœud |
| Plugin réseau | Connectivité entre Pods et Services |

### 9.2 Objet Kubernetes : `metadata`, `spec`, `status`

Un objet Kubernetes possède généralement :

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ml-api
  namespace: ml-platform
spec:
  # État désiré
  replicas: 2
status:
  # État observé, rempli par Kubernetes
  availableReplicas: 2
```

Le champ `spec` est fourni par l'utilisateur. Le champ `status` est observé et mis à jour par Kubernetes. Il ne faut pas modifier manuellement `status` pour essayer de « réparer » un workload.

### 9.3 Gestion déclarative

```bash
# Vérifier les changements avant application
kubectl diff --filename k8s/

# Appliquer tous les manifests du répertoire
kubectl apply --filename k8s/

# Observer l'état après application
kubectl get all --namespace ml-platform
```

Éviter de mélanger, pour les mêmes objets, plusieurs propriétaires de configuration sans convention claire : commandes impératives, `apply`, Helm et GitOps peuvent entrer en concurrence.

> 📌 **Points clés, section 9**
> - Le control plane ne lance pas directement chaque processus applicatif.
> - Les contrôleurs réconcilient en continu l'état courant et l'état désiré.
> - `spec` décrit l'intention, `status` décrit l'observation.

---

## 10. Les objets Kubernetes essentiels

### 10.1 Pod

Un Pod est la plus petite unité déployable de Kubernetes. Il contient un ou plusieurs conteneurs partageant notamment le réseau et certains volumes.

Le modèle recommandé est généralement un conteneur applicatif par Pod. Plusieurs conteneurs sont pertinents lorsqu'ils sont fortement couplés, par exemple une application et un sidecar de proxy ou de collecte.

Un Pod est éphémère. En cas de panne, Kubernetes crée un nouveau Pod, avec un nouvel identifiant. Un Pod ne doit pas être considéré comme une machine permanente.

### 10.2 Deployment

Un Deployment décrit un workload généralement stateless. Il gère des ReplicaSets qui maintiennent le nombre souhaité de Pods et pilotent les mises à jour progressives.

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ml-api
  namespace: ml-platform
  labels:
    app.kubernetes.io/name: ml-api
    app.kubernetes.io/component: api
spec:
  replicas: 2
  selector:
    matchLabels:
      app.kubernetes.io/name: ml-api
  template:
    metadata:
      labels:
        app.kubernetes.io/name: ml-api
    spec:
      containers:
        - name: api
          image: registry.example.com/team/ml-api:1.0.0
          ports:
            - name: http
              containerPort: 8000
```

Le selector du Deployment doit correspondre aux labels du template. Une modification du template crée une nouvelle révision du Deployment.

### 10.3 Service

Les Pods sont remplaçables et leurs adresses IP changent. Un Service fournit une adresse logique stable et sélectionne les Pods par labels.

```yaml
apiVersion: v1
kind: Service
metadata:
  name: ml-api
  namespace: ml-platform
spec:
  selector:
    app.kubernetes.io/name: ml-api
  ports:
    - name: http
      port: 80
      targetPort: http
  type: ClusterIP
```

```text
Client interne
     │ ml-api:80
     ▼
Service ClusterIP
     │ selector: app.kubernetes.io/name=ml-api
     ├── Pod A :8000
     └── Pod B :8000
```

### 10.4 Namespace

Un namespace fournit une portée logique pour les ressources. Il facilite l'isolation organisationnelle, les quotas et les politiques, mais ne constitue pas à lui seul une frontière de sécurité complète.

```yaml
apiVersion: v1
kind: Namespace
metadata:
  name: ml-platform
```

### 10.5 Labels et selectors

Les labels décrivent les objets. Les selectors permettent aux Deployments, Services et autres contrôleurs de les retrouver.

```text
Label du Pod : app.kubernetes.io/name=ml-api
                    │
                    ▼
Selector du Service : sélectionne ce label
```

Les labels doivent être stables, explicites et cohérents. Une faute de frappe dans un selector peut produire un Service sans endpoints ou un Deployment qui ne gère pas les Pods attendus.

> 📌 **Points clés, section 10**
> - Le Pod est l'unité d'exécution, mais rarement l'unité de gestion manuelle.
> - Le Deployment maintient des Pods et organise les mises à jour.
> - Le Service découple les clients des IP éphémères des Pods.
> - Les labels et selectors forment le mécanisme de liaison entre les objets.

---

## 11. Configuration, secrets, stockage et santé

### 11.1 ConfigMap

Une ConfigMap contient de la configuration non sensible : mode d'exécution, nom de modèle, URL interne ou paramètres fonctionnels.

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: ml-api-config
  namespace: ml-platform
data:
  MODEL_NAME: classifier-v1
  LOG_LEVEL: info
```

Injection sous forme de variables :

```yaml
envFrom:
  - configMapRef:
      name: ml-api-config
```

### 11.2 Secret

Un Secret représente une donnée sensible du point de vue de la configuration Kubernetes. Son encodage en base64 n'est pas un chiffrement suffisant. La protection dépend notamment du contrôle d'accès, du chiffrement au repos et de la gestion du cluster.

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: ml-api-secret
  namespace: ml-platform
type: Opaque
stringData:
  REDIS_PASSWORD: change-me-during-deployment
```

Pour une vraie production, éviter de committer des valeurs sensibles dans Git. Utiliser un gestionnaire externe, Sealed Secrets, External Secrets ou une solution équivalente selon l'architecture retenue.

### 11.3 Volumes Kubernetes

Un volume peut être temporaire ou persistant. Les données d'un Pod ne doivent pas être considérées comme durables simplement parce qu'elles sont visibles dans un conteneur.

| Besoin | Mécanisme possible |
|---|---|
| Fichiers temporaires partagés dans un Pod | `emptyDir` |
| Données persistantes d'une application | PersistentVolumeClaim |
| Modèle fourni par un système de stockage | PVC, objet distant ou volume spécialisé |
| Configuration | ConfigMap ou Secret monté |

Exemple de volume temporaire :

```yaml
volumes:
  - name: tmp
    emptyDir: {}
containers:
  - name: api
    volumeMounts:
      - name: tmp
        mountPath: /tmp/app
```

### 11.4 Probes : startup, readiness, liveness

Les trois probes répondent à des questions différentes :

| Probe | Question | Effet typique |
|---|---|---|
| **Startup** | L'application a-t-elle fini de démarrer ? | Protège une initialisation lente |
| **Readiness** | Peut-elle recevoir du trafic maintenant ? | Retire temporairement le Pod du Service |
| **Liveness** | Le processus est-il bloqué ou irrécupérable ? | Peut déclencher un redémarrage |

```yaml
startupProbe:
  httpGet:
    path: /healthz
    port: http
  failureThreshold: 30
  periodSeconds: 10

readinessProbe:
  httpGet:
    path: /readyz
    port: http
  initialDelaySeconds: 5
  periodSeconds: 5

livenessProbe:
  httpGet:
    path: /healthz
    port: http
  periodSeconds: 10
  failureThreshold: 3
```

Une readiness probe ne doit pas simplement vérifier que le processus existe. Elle doit refléter la capacité réelle à servir les requêtes, notamment lorsque l'application dépend d'un modèle, d'une base de données ou d'un service externe.

> 📌 **Points clés, section 11**
> - ConfigMap sert à la configuration non sensible, Secret à la configuration sensible.
> - Base64 n'est pas synonyme de chiffrement.
> - Readiness contrôle le trafic, liveness contrôle la récupération, startup protège l'initialisation.

---

## 12. Ressources, scheduling et mise à l'échelle

### 12.1 Requests et limits

Les `requests` servent notamment au placement du Pod et à la réservation attendue. Les `limits` bornent l'utilisation autorisée selon la ressource et la configuration du cluster.

```yaml
resources:
  requests:
    cpu: "250m"
    memory: "512Mi"
  limits:
    cpu: "1"
    memory: "1Gi"
```

Sans requests cohérentes, le scheduler et les outils d'exploitation ont une vision incomplète des besoins. Sans limits réfléchies, un Pod peut consommer trop de ressources ou subir des arrêts inattendus.

### 12.2 Scheduling

Le scheduler filtre et classe les nœuds selon les ressources et contraintes du Pod. Des mécanismes comme `nodeSelector`, `nodeAffinity`, les taints et tolerations permettent d'exprimer des contraintes.

Exemple de sélection simplifiée d'un nœud GPU :

```yaml
nodeSelector:
  accelerator: nvidia
```

En production, la disponibilité d'un GPU dépend aussi du device plugin, du runtime, des drivers, des quotas et du fournisseur d'infrastructure. Le label seul ne suffit pas à rendre un GPU utilisable.

### 12.3 Réplication et rolling update

```yaml
spec:
  replicas: 3
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxUnavailable: 1
      maxSurge: 1
```

Le rolling update remplace progressivement les anciens Pods. Une readiness probe correcte est importante, car elle empêche d'envoyer du trafic à une instance qui n'est pas prête.

### 12.4 Horizontal Pod Autoscaler

Le HPA ajuste le nombre de Pods selon des métriques. Un exemple minimal :

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: ml-api
  namespace: ml-platform
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: ml-api
  minReplicas: 2
  maxReplicas: 10
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 70
```

Le HPA n'est pas une solution magique. Il faut une API de métriques fonctionnelle, des requests cohérentes, une charge mesurable et une application réellement capable de supporter plusieurs réplicas.

> 📌 **Points clés, section 12**
> - Les requests influencent le placement, les limits encadrent l'exécution.
> - Le scheduling GPU nécessite une chaîne technique complète, pas seulement un label.
> - Le scaling horizontal suppose une application conçue pour être répliquée.

---

## 13. Exposer une application et gérer les déploiements

### 13.1 Types de Services

| Type | Accessibilité | Usage courant |
|---|---|---|
| `ClusterIP` | Interne au cluster | Communication entre services |
| `NodePort` | Port sur les nœuds | Test ou cas simple, rarement choix final |
| `LoadBalancer` | Externe via intégration cloud | Service public ou réseau interne fourni par la plateforme |
| `ExternalName` | Alias DNS | Intégration vers un service externe |

### 13.2 Ingress et Gateway API

L'API Ingress reste stable mais est gelée du point de vue des nouvelles fonctionnalités. Elle nécessite un Ingress Controller. Créer un objet Ingress sans contrôleur n'expose rien.

Gateway API est une orientation plus récente et plus expressive pour les nouveaux développements. Elle introduit notamment `GatewayClass`, `Gateway` et `HTTPRoute`. Elle reste dépendante d'un contrôleur compatible et de l'installation des CRD correspondantes.

Le choix dépend de la plateforme :

```text
Besoin HTTP simple + contrôleur déjà installé
    └── Ingress peut suffire

Nouveau design, séparation des rôles, routage avancé
    └── Évaluer Gateway API
```

### 13.3 Rollout et rollback

```bash
# Suivre le déploiement
kubectl rollout status deployment/ml-api --namespace ml-platform

# Voir l'historique
kubectl rollout history deployment/ml-api --namespace ml-platform

# Revenir à la révision précédente
kubectl rollout undo deployment/ml-api --namespace ml-platform

# Voir les révisions et la configuration du Deployment
kubectl describe deployment/ml-api --namespace ml-platform
```

Un rollback efficace suppose que les images, manifests et configurations soient versionnés. Un rollback du Deployment ne restaure pas nécessairement une migration de base de données déjà exécutée.

> 📌 **Points clés, section 13**
> - Un Service stable le trafic vers des Pods remplaçables.
> - Ingress et Gateway API nécessitent un contrôleur, ils ne suffisent pas seuls.
> - Un rollback applicatif ne remonte pas automatiquement les changements de données.

---

## 14. Fil rouge : de l'API locale au cluster

### 14.1 Étape 1, exécution locale

```bash
uvicorn app.main:app --reload --port 8000
curl http://localhost:8000/
curl http://localhost:8000/healthz
```

### 14.2 Étape 2, conteneur local

```bash
docker build --tag ml-api:1.0 .
docker run --detach --name ml-api --publish 8000:8000 ml-api:1.0
curl http://localhost:8000/
docker logs --follow ml-api
```

### 14.3 Étape 3, environnement Compose

```bash
docker compose up --build --detach
docker compose ps
curl http://localhost:8000/
```

Compose est ici le niveau de reproductibilité locale et d'intégration multi-services.

### 14.4 Étape 4, publication dans un registre

```bash
docker tag ml-api:1.0 registry.example.com/team/ml-api:1.0.0
docker push registry.example.com/team/ml-api:1.0.0
```

L'image doit être accessible par le cluster, avec des credentials configurés si le registre est privé.

### 14.5 Étape 5, manifests Kubernetes

Namespace :

```yaml
apiVersion: v1
kind: Namespace
metadata:
  name: ml-platform
```

Deployment :

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ml-api
  namespace: ml-platform
spec:
  replicas: 2
  selector:
    matchLabels:
      app.kubernetes.io/name: ml-api
  template:
    metadata:
      labels:
        app.kubernetes.io/name: ml-api
    spec:
      containers:
        - name: api
          image: registry.example.com/team/ml-api:1.0.0
          ports:
            - name: http
              containerPort: 8000
          readinessProbe:
            httpGet:
              path: /readyz
              port: http
            periodSeconds: 5
          livenessProbe:
            httpGet:
              path: /healthz
              port: http
            periodSeconds: 10
          resources:
            requests:
              cpu: "250m"
              memory: "512Mi"
            limits:
              cpu: "1"
              memory: "1Gi"
```

Service :

```yaml
apiVersion: v1
kind: Service
metadata:
  name: ml-api
  namespace: ml-platform
spec:
  selector:
    app.kubernetes.io/name: ml-api
  ports:
    - name: http
      port: 80
      targetPort: http
  type: ClusterIP
```

### 14.6 Étape 6, application et test

```bash
kubectl apply --filename k8s/
kubectl get pods --namespace ml-platform
kubectl get service --namespace ml-platform
kubectl rollout status deployment/ml-api --namespace ml-platform

# Tester localement sans configurer encore un Ingress
kubectl port-forward service/ml-api 8080:80 --namespace ml-platform
curl http://localhost:8080/
```

### 14.7 Étape 7, diagnostic systématique

```text
1. Le contexte kubectl pointe-t-il vers le bon cluster ?
2. Le namespace contient-il le Deployment attendu ?
3. Le Deployment a-t-il créé un ReplicaSet ?
4. Les Pods sont-ils Running et Ready ?
5. Les événements indiquent-ils une erreur d'image ou de scheduling ?
6. Le Service possède-t-il des endpoints ?
7. L'application écoute-t-elle sur 0.0.0.0 et le bon port ?
8. Les probes vérifient-elles le bon chemin ?
9. Les requests sont-elles compatibles avec les nœuds disponibles ?
```

Pour une application LLM, ce cours s'arrête au packaging et à l'orchestration. Les choix de moteur, KV cache, quantification, batching et tuning GPU relèvent de [serving_llm.md](./serving_llm.md).

> 📌 **Points clés, section 14**
> - Le même artefact doit être testé localement avant d'être déployé dans le cluster.
> - `port-forward` permet de tester un Service sans exposer immédiatement l'application publiquement.
> - Le diagnostic suit une chaîne, du contexte Kubernetes jusqu'aux endpoints du Service.

---

## 15. Guide de choix

### 15.1 Tableau comparatif

| Critère | Docker seul | Docker Compose | Kubernetes |
|---|---|---|---|
| Cible principale | Processus ou service local | Plusieurs services sur une machine | Workloads sur un cluster |
| Complexité | Faible | Faible à moyenne | Élevée |
| Multi-nœuds | Non | Non, sauf extensions externes | Oui |
| Réplication native | Limitée | Limitée | Oui |
| Rolling update | À construire | Limité | Oui |
| Configuration | Variables, fichiers, secrets externes | YAML, variables, secrets | ConfigMap, Secret, gestion externe |
| Diagnostic | Logs, inspect, stats | Logs et état des services | Events, logs, describe, probes |
| Cas typique | Développement, job, service simple | Environnement local d'une application | Plateforme de production multi-services |

### 15.2 Arbre de décision

```text
Un seul service sur une machine ?
    ├── Oui → Docker seul
    └── Non
        Plusieurs services sur une machine pour développement/test ?
            ├── Oui → Docker Compose
            └── Non
                Plusieurs nœuds, réplication, rollouts ou scheduling ?
                    ├── Oui → Kubernetes
                    └── Non → Commencer simple, ne pas ajouter Kubernetes par défaut
```

### 15.3 Frontières avec les autres cours

| Sujet | Cours de référence |
|---|---|
| Architecture globale d'un système IA, MLOps, monitoring | [infra_ia.md](./infra_ia.md) |
| Moteurs d'inférence LLM et optimisation GPU | [serving_llm.md](./serving_llm.md) |
| Modèles, Transformers et cycle de vie LLM | [transformers_llm_cours_complet.md](../02_deep_learning/03_llm/transformers_llm_cours_complet.md) |
| Bases de données et SQL | [04_databases/sql/INDEX.md](../04_databases/sql/INDEX.md) |

> 📌 **Points clés, section 15**
> - Il faut choisir le plus petit niveau d'orchestration qui satisfait les contraintes.
> - Compose est d'abord un outil local multi-services, Kubernetes une plateforme de cluster.
> - Les cours doivent se compléter : architecture globale, plateforme et performance spécialisée sont trois niveaux distincts.

---

## 16. Pense-bête des commandes

Cette section est conçue pour une consultation rapide. Les commandes destructrices sont signalées par ⚠️.

### 16.1 Docker, diagnostic de base

```bash
# Version et informations générales
docker version
docker info

# Images
docker images
docker image ls
docker pull IMAGE:TAG
docker image inspect IMAGE:TAG
docker history IMAGE:TAG
docker image rm IMAGE:TAG

# Construire et taguer
docker build --tag NAME:TAG .
docker build --file Dockerfile.dev --tag NAME:dev .
docker tag NAME:TAG REGISTRY/ORG/NAME:TAG

# Conteneurs
docker ps
docker ps --all
docker run --detach --name NAME IMAGE:TAG
docker run --interactive --tty --rm IMAGE:TAG sh
docker start NAME
docker stop NAME
docker restart NAME
docker rm NAME

# Ports, variables et volumes
docker run --publish 8080:8000 IMAGE:TAG
docker run --env KEY=value IMAGE:TAG
docker run --env-file .env IMAGE:TAG
docker run --mount source=VOL,target=/data IMAGE:TAG

# Logs et inspection
docker logs NAME
docker logs --follow --tail 100 NAME
docker inspect NAME
docker exec --interactive --tty NAME sh
docker stats NAME
docker top NAME
docker events
```

### 16.2 Docker, réseau et stockage

```bash
# Réseaux
docker network ls
docker network create NETWORK
docker network inspect NETWORK
docker network connect NETWORK CONTAINER
docker network disconnect NETWORK CONTAINER
docker network rm NETWORK

# Volumes
docker volume ls
docker volume create VOLUME
docker volume inspect VOLUME
docker volume rm VOLUME

# Espace et nettoyage
docker system df
docker builder prune
# ⚠️ Supprime les ressources inutilisées
docker system prune
# ⚠️ Plus agressif, à utiliser avec discernement
docker system prune --all
```

### 16.3 Docker Compose

```bash
# Démarrer
docker compose up
docker compose up --build
docker compose up --detach

# Observer
docker compose ps
docker compose logs
docker compose logs --follow SERVICE
docker compose config

# Exécuter
docker compose exec SERVICE sh
docker compose run --rm SERVICE COMMAND

# Arrêter et nettoyer
docker compose stop
docker compose restart SERVICE
docker compose down
# ⚠️ Supprime aussi les volumes déclarés
docker compose down --volumes
```

### 16.4 Kubernetes, contexte et découverte

```bash
# Version et contexte
kubectl version
kubectl cluster-info
kubectl config get-contexts
kubectl config current-context
kubectl config use-context CONTEXT

# Ressources disponibles
kubectl api-resources
kubectl explain deployment.spec.template.spec.containers
kubectl get namespaces
kubectl get nodes
kubectl get pods --all-namespaces
```

### 16.5 Kubernetes, application et inspection

```bash
# Mode déclaratif
kubectl diff --filename k8s/
kubectl apply --filename k8s/
kubectl delete --filename k8s/

# Ressources dans un namespace
kubectl get all --namespace NAMESPACE
kubectl get pods --namespace NAMESPACE --output wide
kubectl get deployment SERVICE --namespace NAMESPACE
kubectl get service SERVICE --namespace NAMESPACE
kubectl get endpointslice --namespace NAMESPACE

# Détails et événements
kubectl describe pod POD --namespace NAMESPACE
kubectl describe deployment DEPLOYMENT --namespace NAMESPACE
kubectl get events --namespace NAMESPACE --sort-by=.lastTimestamp

# Logs et shell
kubectl logs POD --namespace NAMESPACE
kubectl logs --follow POD --namespace NAMESPACE
kubectl logs POD --container CONTAINER --namespace NAMESPACE
kubectl exec --interactive --tty POD --namespace NAMESPACE -- sh

# Test local
kubectl port-forward service/SERVICE 8080:80 --namespace NAMESPACE
```

### 16.6 Kubernetes, rollouts et scaling

```bash
# Rollout
kubectl rollout status deployment/DEPLOYMENT --namespace NAMESPACE
kubectl rollout history deployment/DEPLOYMENT --namespace NAMESPACE
kubectl rollout pause deployment/DEPLOYMENT --namespace NAMESPACE
kubectl rollout resume deployment/DEPLOYMENT --namespace NAMESPACE
kubectl rollout undo deployment/DEPLOYMENT --namespace NAMESPACE

# Scaling manuel
kubectl scale deployment/DEPLOYMENT --replicas=3 --namespace NAMESPACE

# HPA et ressources
kubectl get hpa --namespace NAMESPACE
kubectl describe hpa HPA --namespace NAMESPACE
kubectl top pods --namespace NAMESPACE
kubectl top nodes

# Autorisations
kubectl auth can-i get pods --namespace NAMESPACE
kubectl auth can-i create deployments --namespace NAMESPACE
```

### 16.7 Diagnostic express d'un Pod en erreur

```bash
kubectl get pod POD --namespace NAMESPACE --output wide
kubectl describe pod POD --namespace NAMESPACE
kubectl logs POD --namespace NAMESPACE --all-containers
kubectl get events --namespace NAMESPACE --sort-by=.lastTimestamp
```

Interprétation rapide :

| Symptôme | Première hypothèse |
|---|---|
| `ImagePullBackOff` | Nom, tag, registre ou credentials incorrects |
| `CrashLoopBackOff` | Processus qui quitte, configuration ou dépendance invalide |
| `Pending` | Ressources, taints, affinity, PVC ou scheduler |
| `Running` mais `0/1 Ready` | Readiness probe ou dépendance non disponible |
| Service sans endpoint | Selector qui ne correspond pas aux labels |
| `OOMKilled` | Limite mémoire trop basse ou fuite mémoire |
| Timeout externe | Service, Ingress/Gateway, NetworkPolicy ou application |

---

## 17. Pièges courants

**Piège 1 : confondre image et conteneur**  
Une image est un artefact immuable, un conteneur est une instance en fonctionnement. **Versionner l'image ne rend pas automatiquement les données du conteneur persistantes.**

**Piège 2 : utiliser `latest` en production**  
Le tag peut pointer vers une autre image plus tard. **Utiliser un tag de release immuable ou un digest lorsque la reproductibilité est importante.**

**Piège 3 : mettre un secret dans `ARG`, `ENV` ou Git**  
Les couches d'image, l'historique et les fichiers de configuration peuvent conserver la valeur. **Utiliser un mécanisme de secrets adapté au build et au runtime.**

**Piège 4 : croire que `EXPOSE` publie un port**  
`EXPOSE` documente un port interne. **La publication vient de `--publish`, Compose ou d'un Service Kubernetes.**

**Piège 5 : confondre conteneur démarré et application prête**  
Un processus peut être lancé tout en étant incapable de recevoir du trafic. **Utiliser des healthchecks, readiness probes et retry applicatifs.**

**Piège 6 : oublier `0.0.0.0` dans une API conteneurisée**  
Une application qui écoute seulement sur `127.0.0.1` peut être inaccessible depuis le réseau du conteneur. **Écouter sur l'interface appropriée, souvent `0.0.0.0`.**

**Piège 7 : selector Service et labels différents**  
Le Service existe mais ne possède aucun endpoint. **Comparer exactement `spec.selector` avec les labels du template de Pod.**

**Piège 8 : utiliser un Pod nu pour une application durable**  
Un Pod créé seul n'offre pas le même cycle de gestion qu'un Deployment. **Utiliser un contrôleur adapté au type de workload.**

**Piège 9 : croire que base64 chiffre un Secret Kubernetes**  
Base64 est un encodage. **Protéger l'accès, activer le chiffrement au repos et envisager une solution de secrets externe.**

**Piège 10 : ajouter Kubernetes trop tôt**  
Kubernetes apporte une puissance importante, mais aussi une charge cognitive et opérationnelle. **Commencer par Docker ou Compose lorsque le besoin ne justifie pas un cluster.**

---

## 18. Exercices d'auto-évaluation

### Questions de révision

1. Quelle est la différence entre une image Docker et un conteneur Docker ?
2. Pourquoi la couche writable d'un conteneur ne doit-elle pas être utilisée comme stockage persistant ?
3. Quelle différence y a-t-il entre `ARG` et `ENV` dans un Dockerfile ?
4. Pourquoi l'ordre des instructions d'un Dockerfile influence-t-il le temps de build ?
5. Dans quel cas Docker Compose est-il plus adapté que Kubernetes ?
6. Quel problème le Service Kubernetes résout-il par rapport à l'utilisation directe des IP de Pods ?
7. Quelle différence y a-t-il entre readiness, liveness et startup probes ?
8. Pourquoi `requests` et `limits` influencent-ils le scheduling et la stabilité ?
9. Que signifie le statut `Pending` d'un Pod, et quelles commandes utiliseriez-vous en premier ?
10. Pourquoi un rollback du Deployment ne suffit-il pas nécessairement à annuler une migration de base de données ?

### Exercice applicatif

Vous devez déployer une API de classification avec les contraintes suivantes :

- l'image est `registry.example.com/team/classifier-api:2.1.0` ;
- l'API écoute sur le port 8000 ;
- le déploiement doit maintenir 3 réplicas ;
- le trafic ne doit être envoyé qu'après le succès de `/readyz` ;
- un fichier de configuration non sensible doit fournir `MODEL_NAME` ;
- chaque Pod doit demander `250m` CPU et `512Mi` de mémoire ;
- le Service doit être accessible seulement à l'intérieur du cluster ;
- le déploiement doit pouvoir être annulé.

Produisez :

1. un Deployment ;
2. un Service ;
3. une ConfigMap ;
4. les commandes d'application, d'observation du rollout et de rollback ;
5. une procédure de diagnostic si le Service n'a aucun endpoint.

*Solution attendue :* les manifests doivent utiliser un selector identique aux labels des Pods, `ClusterIP`, une readiness probe HTTP, des requests, une ConfigMap injectée dans le conteneur et un Deployment géré par `kubectl rollout`.

<details>
<summary>Réponses aux questions de révision</summary>

1. L'image est un artefact immuable, le conteneur est une instance de cette image avec une configuration d'exécution et un processus.
2. La couche writable appartient au cycle de vie du conteneur et disparaît généralement avec lui. Les données persistantes doivent être externalisées.
3. `ARG` sert principalement pendant le build. `ENV` peut rester dans l'image et être disponible à l'exécution. Aucun des deux ne doit servir à stocker un secret.
4. Le cache est invalidé à partir d'une instruction modifiée. Placer les dépendances stables avant le code permet de réutiliser davantage le cache.
5. Compose est adapté à plusieurs services sur une machine, surtout en développement et intégration. Kubernetes est adapté à une plateforme de cluster avec réplication, rollouts et scheduling.
6. Le Service fournit une adresse logique stable et sélectionne les Pods par labels, alors que les IP de Pods sont éphémères.
7. Startup vérifie la fin du démarrage, readiness la capacité à recevoir du trafic et liveness la nécessité éventuelle d'un redémarrage.
8. Les requests participent au placement et les limits bornent la consommation. Des valeurs incohérentes peuvent provoquer un mauvais scheduling ou des OOMKills.
9. `Pending` signifie que le Pod n'est pas encore exécuté comme attendu. Commencer par `kubectl describe pod`, puis lire les événements, vérifier les ressources, les contraintes, le PVC et l'image.
10. Le rollback modifie la version du workload, mais une migration de données déjà exécutée demande sa propre stratégie de compatibilité ou de restauration.

</details>

<details>
<summary>Trame de correction de l'exercice applicatif</summary>

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: classifier-config
  namespace: ml-platform
data:
  MODEL_NAME: classifier-v2
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: classifier-api
  namespace: ml-platform
spec:
  replicas: 3
  selector:
    matchLabels:
      app.kubernetes.io/name: classifier-api
  template:
    metadata:
      labels:
        app.kubernetes.io/name: classifier-api
    spec:
      containers:
        - name: api
          image: registry.example.com/team/classifier-api:2.1.0
          ports:
            - name: http
              containerPort: 8000
          envFrom:
            - configMapRef:
                name: classifier-config
          readinessProbe:
            httpGet:
              path: /readyz
              port: http
          resources:
            requests:
              cpu: "250m"
              memory: "512Mi"
---
apiVersion: v1
kind: Service
metadata:
  name: classifier-api
  namespace: ml-platform
spec:
  type: ClusterIP
  selector:
    app.kubernetes.io/name: classifier-api
  ports:
    - name: http
      port: 80
      targetPort: http
```

```bash
kubectl apply --filename classifier.yaml
kubectl rollout status deployment/classifier-api --namespace ml-platform
kubectl rollout undo deployment/classifier-api --namespace ml-platform
kubectl get endpointslice --namespace ml-platform
kubectl describe service/classifier-api --namespace ml-platform
kubectl get pods --show-labels --namespace ml-platform
```

</details>

---

## 19. Sources et références

### Documentation officielle Docker

- [Docker overview](https://docs.docker.com/get-started/docker-overview/)
- [What is an image?](https://docs.docker.com/get-started/docker-concepts/the-basics/what-is-an-image/)
- [What is a container?](https://docs.docker.com/get-started/docker-concepts/the-basics/what-is-a-container/)
- [Dockerfile reference](https://docs.docker.com/reference/dockerfile/)
- [Docker build overview](https://docs.docker.com/build/concepts/overview/)
- [Building best practices](https://docs.docker.com/build/building/best-practices/)
- [Multi-stage builds](https://docs.docker.com/build/building/multi-stage/)
- [Build secrets](https://docs.docker.com/build/building/secrets/)
- [Docker Compose](https://docs.docker.com/compose/)
- [Compose startup order](https://docs.docker.com/compose/how-tos/startup-order/)
- [Docker networking overview](https://docs.docker.com/engine/network/)
- [Volumes](https://docs.docker.com/engine/storage/volumes/)
- [Bind mounts](https://docs.docker.com/engine/storage/bind-mounts/)
- [Docker Engine security](https://docs.docker.com/engine/security/)
- [Rootless mode](https://docs.docker.com/engine/security/rootless/)
- [Docker registries](https://docs.docker.com/get-started/docker-concepts/the-basics/what-is-a-registry/)

### Documentation officielle Kubernetes

- [Cluster architecture](https://kubernetes.io/docs/concepts/architecture/)
- [Objects in Kubernetes](https://kubernetes.io/docs/concepts/overview/working-with-objects/)
- [Pods](https://kubernetes.io/docs/concepts/workloads/pods/)
- [Pod lifecycle](https://kubernetes.io/docs/concepts/workloads/pods/pod-lifecycle/)
- [Deployments](https://kubernetes.io/docs/concepts/workloads/controllers/deployment/)
- [Services](https://kubernetes.io/docs/concepts/services-networking/service/)
- [ConfigMaps](https://kubernetes.io/docs/concepts/configuration/configmap/)
- [Secrets](https://kubernetes.io/docs/concepts/configuration/secret/)
- [Volumes](https://kubernetes.io/docs/concepts/storage/volumes/)
- [Persistent volumes](https://kubernetes.io/docs/concepts/storage/persistent-volumes/)
- [Resource management](https://kubernetes.io/docs/concepts/configuration/manage-resources-containers/)
- [Assigning Pods to nodes](https://kubernetes.io/docs/concepts/scheduling-eviction/assign-pod-node/)
- [Horizontal Pod Autoscaling](https://kubernetes.io/docs/tasks/run-application/horizontal-pod-autoscale/)
- [Ingress](https://kubernetes.io/docs/concepts/services-networking/ingress/)
- [Gateway API](https://kubernetes.io/docs/concepts/services-networking/gateway/)
- [kubectl quick reference](https://kubernetes.io/docs/reference/kubectl/quick-reference/)
- [Debugging running Pods](https://kubernetes.io/docs/tasks/debug/debug-application/debug-running-pod/)
- [Debugging Services](https://kubernetes.io/docs/tasks/debug/debug-application/debug-service/)
- [Using RBAC authorization](https://kubernetes.io/docs/reference/access-authn-authz/rbac/)
- [Container runtimes](https://kubernetes.io/docs/setup/production-environment/container-runtimes/)

### Standards et projets

- [Open Container Initiative](https://opencontainers.org/)
- [OCI Image Specification](https://github.com/opencontainers/image-spec)
- [OCI Runtime Specification](https://github.com/opencontainers/runtime-spec)
- [OCI Distribution Specification](https://github.com/opencontainers/distribution-spec)
- [containerd](https://containerd.io/)
- [Container Runtime Interface](https://kubernetes.io/docs/concepts/containers/cri/)
- [Cloud Native Computing Foundation](https://www.cncf.io/projects/)

---

## 20. Liens connexes

- [Infrastructure et déploiement de systèmes IA](./infra_ia.md)
- [Serving et optimisation d'inférence LLM](./serving_llm.md)
- [Cours complet Transformers et LLM](../02_deep_learning/03_llm/transformers_llm_cours_complet.md)
- [Bases SQL](../04_databases/sql/INDEX.md)
- [Structure de référence des cours](../_non_merge/05_methodologie/structure_cours.md)
- [README du second cerveau](../README.md)

---

*Ce cours présente des fondamentaux durables. Les commandes, API et recommandations dépendantes d'une version doivent être vérifiées dans la documentation officielle correspondant au cluster et au runtime utilisés.*
