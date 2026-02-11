# Architecture d'Infrastructure Complète pour Projets IA

## Table des matières

1. Introduction et principes fondamentaux
2. Architecture Data (DL → DW → DM)
3. Module ETL et transformations
4. Bases de connaissances et règles métier
5. Module RAG (Retrieval-Augmented Generation)
6. Modules de modélisation (ML et LLM)
7. Orchestrateur IA
8. Modules de gouvernance et monitoring
9. Cas pratique : Classification d'emails
10. Mise en production et MLOps

---

## 1. Introduction et principes fondamentaux

### 1.1 Pourquoi une architecture IA spécifique ?

Un projet IA échoue rarement à cause du modèle, mais souvent à cause de l'infrastructure qui l'entoure. Selon des rapports industriels, environ 95% des projets IA n'atteignent jamais la production ou échouent en production.

**Les causes principales d'échec :**
- Qualité des données insuffisante ("garbage in, garbage out")
- Absence de traçabilité et d'auditabilité
- Confusion des responsabilités entre modules
- Absence de contrats entre composants
- Non-maîtrise de la dérive (data drift, concept drift)
- Couplage fort entre composants

**Principe fondamental :**

> Un bon système IA n'est pas un système "intelligent", c'est un système maîtrisé, testé, traçable et gouverné.

### 1.2 Les rôles dans une architecture IA

Une architecture IA bien conçue sépare clairement les responsabilités :

| Composant | Rôle | Ce qu'il NE fait PAS |
|-----------|------|----------------------|
| **LLM** | Raisonner, expliquer, reformuler | Calculer des faits métier, décider seul |
| **ML classique** | Classifier, scorer, prédire (rapide) | Générer du texte, raisonner |
| **ETL** | Nettoyer, normaliser, calculer des faits | Raisonner, générer |
| **Orchestrateur** | Coordonner, router, valider | Calculer des faits métier |
| **RAG** | Récupérer du contexte documentaire | Calculer, décider |

### 1.3 Principe architectural : un module = une responsabilité

**Règle d'or :**

> On ne découpe pas une architecture par type de données, mais par responsabilités métier stables.

Les modules communiquent via des **contrats JSON stables**, jamais par interprétation implicite.

**Chaque contrat doit être :**
- **versionné** (contract_version)
- **documenté** (champs, sémantique, règles)
- **associé à un owner** (responsable)
- **testé automatiquement** (compatibilité ascendante si possible)

**Exemple de contrat JSON :**

```json
{
  "contract_version": "v1.2.0",
  "owner": "equipe_data",
  "schema": {
    "client_vip": {"type": "boolean", "required": true},
    "anciennete_contrat_mois": {"type": "integer", "min": 0}
  },
  "sla": {
    "freshness_max_hours": 24,
    "quality_threshold": 0.95
  }
}
```

---

## 2. Architecture Data (DL → DW → DM)

### 2.1 Vue d'ensemble : l'architecture médaillon (Bronze → Silver → Gold)

L'architecture médaillon (popularisée par Databricks) organise les données en couches de qualité croissante.

**Principe :**

> Les données progressent à travers des couches de raffinement : bronze (brut) → silver (validé) → gold (enrichi/agrégé).

**Source :** Microsoft Azure Databricks, "What is the medallion lakehouse architecture?" (https://learn.microsoft.com/en-us/azure/databricks/lakehouse/medallion)
```
[Sources]
    ↓
[Bronze/Raw] ← Données brutes, append-only, immuables
    ↓
[Validation + Quarantaine] ← Publish gate
    ↓
[Silver/DWH] ← Données nettoyées, modélisées
    ↓
[Service de projection IA]
    ↓
[Gold/DM] ← Données optimisées pour consommation (BI, ML, IA)
```

### 2.2 Entrepôt 1 : Zone brute (Bronze / Data Lake)

**Rôle :** Conserver la source de vérité brute, immuable.

**Deux patterns valides :**

#### Pattern A : Landing zone temporaire

- Les données arrivent en zone tampon
- Elles sont validées et transférées vers le DWH
- La landing zone est purgée après transfert réussi

**Quand l'utiliser :**
- Le SIO conserve les données sources
- Le DWH devient la vérité analytique
- Besoin de minimiser les coûts de stockage

#### Pattern B : Data Lake permanent

- Les données arrivent et sont conservées indéfiniment
- Le Data Lake est immuable (append-only)
- Permet de rejouer les traitements à tout moment

**Quand l'utiliser :**
- Le SIO purge les données
- Besoin d'audit historique complet
- Nécessité de reprocesser les données brutes

**Contenu typique :**
- Données brutes dans leur format d'origine (JSON, CSV, Parquet, Avro, etc.)
- Métadonnées d'ingestion (source, timestamp, run_id)
- Organisation par source/date/partition

**Principes clés (selon Microsoft Azure)** :
- Minimal data cleanup
- Append-only
- Retain all historical data (source de vérité)
- Store most fields as string/binary to protect against schema changes

**Exemple de structure :**
```
/bronze/
/emails_raw/
/source=outlook/
/date=2026-02-11/
emails_batch_001.eml
emails_batch_002.eml
/source=gmail/
/date=2026-02-11/
emails_batch_001.eml
```
### 2.3 Validation, conformité et quarantaine (Publish gate)

**Principe du publish gate :**

> Un dataset n'est publié vers DW/DM que si les contrôles de qualité et de conformité passent. Sinon : quarantaine.

**Source :** Pattern "quarantine bad data" décrit dans DQX framework (https://databrickslabs.github.io/dqx/docs/motivation/)

**Contrôles typiques :**

1. **Contrôles de schéma (hard-fail)**
   - Colonnes manquantes
   - Types incompatibles
   - Renommages inattendus

2. **Contrôles de qualité critique (hard-fail)**
   - Clés nulles
   - Doublons sur identifiants uniques
   - Violations de contraintes FK

3. **Contrôles statistiques (soft/hard selon seuil)**
   - Chute brutale de volume (ex: -80%)
   - Explosion de volume (ex: +300%)
   - Dérive de distributions
   - Retard d'arrivée

**Processus de quarantaine :**

1. **Détection automatique** (via règles DQ)
2. **Isolation** (marquage ou déplacement physique)
3. **Génération rapport** :
   - batch_id, partition
   - Règles violées avec échantillons
   - Métadonnées (source, date, volumétrie)
   - Gravité (bloquant/warning)
4. **Alerting** (équipe data/métier)
5. **Investigation & correction**
6. **Retraitement** (avec version corrigée)
7. **Levée de quarantaine** (tracée dans audit trail)

**Exemple de structure quarantaine :**

```
/conformance/
/emails/
/input/       ← Données en cours de validation
/output/      ← Données validées
/quarantine/  ← Données rejetées avec rapport
/2026-02-11/
batch_001_report.json
batch_001_data.parquet
```

**Pourquoi la quarantaine est critique :**
- Empêche la propagation d'erreurs en aval
- Permet investigation sans impacter la production
- Trace l'historique des anomalies (conformité)

### 2.4 Entrepôt 2 : Data Warehouse (Silver)

**Rôle :** Représentation structurée, validée et modélisée des données.

**Contenu :**
- **Faits métier** (événements, transactions, mesures)
- **Dimensions** (clients, produits, temps, géographie)
- **Indicateurs** (pré-agrégations)
- **Historisation** (SCD Type 1 ou Type 2)

**Modélisation dimensionnelle (star schema) :**

Le schéma en étoile est l'approche standard pour modéliser un DWH analytique.

**Source :** Microsoft Power BI, "Understand star schema" (https://learn.microsoft.com/en-us/power-bi/guidance/star-schema)

**Principes :**
- **Dimensions** (Type 1 : dernière valeur, Type 2 : historisation complète)
- **Faits** (mesures quantitatives, clés vers dimensions)
- **Dénormalisation** des dimensions (snowflake → star)

**Exemple : DWH pour emails**

Dimensions :
- **dim_date** (date_id, date, annee, mois, jour, semaine)
- **dim_expediteur** (expediteur_id, email, domaine, type)
- **dim_destinataire** (destinataire_id, email, domaine, type)
- **dim_categorie** (categorie_id, categorie, sous_categorie)

Faits :
- **fact_emails** (email_id, date_id, expediteur_id, destinataire_id, categorie_id, longueur_sujet, longueur_corps, nb_pieces_jointes, timestamp)

**Transformations DWH (depuis bronze) :**

1. **Parsing et extraction structurée**
   - Parsing MIME (.eml) → colonnes structurées
   - Extraction headers (from, to, cc, subject, date, message_id)
   - Extraction corps (text, html)
   - Extraction pièces jointes (métadonnées)

2. **Nettoyage et normalisation**
   - Décodage encodages (quoted-printable, base64, charset → UTF-8)
   - Canonisation emails (lowercase, suppression alias)
   - Nettoyage HTML (suppression scripts/styles)
   - Extraction URLs

3. **Enrichissement**
   - Détection langue
   - Extraction domaines
   - Indicateurs sécurité (SPF/DKIM/DMARC)
   - Thread detection (via Message-ID, In-Reply-To)

4. **Calculs métier déterministes**
   - Classification domaine (interne/externe)
   - Détection patterns (urgence, spam, phishing)
   - Agrégations (emails/jour/expéditeur)

**Historisation (SCD Type 2) :**

Pour les dimensions qui évoluent (ex: classification d'un expéditeur change), on utilise le Type 2.

**Exemple dim_expediteur avec SCD Type 2 :**

| expediteur_key | email | classification | valid_from | valid_to | is_current |
|----------------|-------|----------------|------------|----------|------------|
| 1 | user@example.com | normal | 2025-01-01 | 2026-01-15 | false |
| 2 | user@example.com | suspect | 2026-01-16 | 9999-12-31 | true |

**Pourquoi le DWH est la vérité métier :**
- Données validées et nettoyées
- Modélisation alignée sur les besoins métier
- Historisation pour analyse temporelle
- Base pour construire les datamarts

### 2.5 Service de projection IA (Serving Data)

**Rôle :** Projeter des vues contractuelles, stables, versionnées, prêtes à injecter dans les modules IA.

**Principe :**

> Le service de projection IA prépare le terrain, le LLM joue le match.

**Ce service ne fait PAS :**
- Nettoyage (déjà fait dans DWH)
- Calculs lourds (déjà faits dans DWH)
- Décisions métier

**Ce service FAIT :**
- Projection de vues selon contrats stables
- Sérialisation optimisée (JSON, Parquet, etc.)
- Versionning des vues

**Trois types de sorties :**

#### A. Faits métier (pré-calculés, déterministes)

Tout ce qui est binaire, calculable, basé sur des dates/seuils connus.

```json
{
  "email_uid": "abc123",
  "est_interne": true,
  "contient_pieces_jointes": true,
  "nb_pieces_jointes": 3,
  "expediteur_suspect": false,
  "urls_externes": 2,
  "longueur_sujet": 45,
  "longueur_corps": 1200,
  "langue_detectee": "fr",
  "urgent": false
}
```

#### B. Contexte structuré (descriptif, non décisionnel)

```json
{
  "expediteur": "contact@example.com",
  "domaine_expediteur": "example.com",
  "destinataires": ["user1@internal.com", "user2@internal.com"],
  "sujet": "Demande de résiliation",
  "timestamp": "2026-02-11T10:00:00Z",
  "thread_id": "thread_xyz"
}
```

#### C. Texte canonique pour modèles NLP

```json
{
  "text_for_model": "Objet: Demande de résiliation\n\nBonjour,\n\nJe souhaite résilier mon contrat...",
  "text_cleaned": "demande resiliation souhait resilier contrat",
  "embedding_source": "text_for_model"
}
```

**Quand appeler ce service ?**

**Stratégie recommandée : projection batch (post-DWH)**

- **Déclenchement :** après écriture dans DWH (automatique, asynchrone)
- **Fréquence :** même cadence que l'ETL métier (quotidien, horaire, temps réel)
- **Avantages :**
  - Aucune latence au moment de la requête utilisateur
  - Cohérence des vues IA
  - Optimisation possible (calculs groupés, cache)
- **Inconvénients :**
  - Données potentiellement non à jour (selon fréquence ETL)
  - Stockage additionnel

**Schéma temporel :**

T0: Donnée brute arrive (SIO)
T1: ETL métier (nettoyage, RGPD, calculs)
T2: Écriture DWH
T3: Service de projection IA (asynchrone)
T4: Stockage des vues IA-ready
─────────────────────────────────────
T5: Question utilisateur (LLM)
T6: Injection immédiate des vues IA

**Le LLM ne déclenche jamais la projection lourde.**

### 2.6 Entrepôt 3 : Data Mart (Gold)

**Rôle :** Données optimisées pour consommation (BI, ML, IA applicative).

**Source :** Microsoft Azure, "Curated layer (gold)" (https://learn.microsoft.com/en-us/azure/cloud-adoption-framework/scenarios/cloud-scale-analytics/best-practices/data-lake-zones)

**Contenu :**
- Agrégats métier (ex: emails/jour/catégorie)
- Features ML (ex: embeddings pré-calculés)
- Datasets entraînement/validation/test
- Vues BI (ex: dashboards)

**Organisation par domaine/usage :**

/gold/
/bi_dashboards/
emails_par_categorie_jour.parquet
top_expediteurs_suspect.parquet
/ml_classification/
emails_training_set_v2.parquet
emails_validation_set_v2.parquet
emails_test_set_v2.parquet
/ai_inference/
emails_to_classify_realtime.json

**Exemple : dataset ML pour classification d'emails**

| email_uid | text_for_model | label | split | dataset_version | created_at |
|-----------|----------------|-------|-------|-----------------|------------|
| email_001 | "Demande résiliation..." | resiliation | train | v2.1.0 | 2026-02-10 |
| email_002 | "Question facturation..." | facturation | train | v2.1.0 | 2026-02-10 |
| email_003 | "Modification contrat..." | modification | val | v2.1.0 | 2026-02-10 |

**Features additionnelles (optionnelles) :**

| email_uid | longueur_sujet | nb_urls | domaine_interne | heure_envoi | jour_semaine |
|-----------|----------------|---------|-----------------|-------------|--------------|
| email_001 | 45 | 0 | false | 10 | lundi |
| email_002 | 32 | 2 | true | 14 | mardi |

**Principe de qualité Gold :**

> Les données Gold sont alignées sur la logique métier et optimisées pour la performance.

### 2.7 Traçabilité (lineage) DL → DW → DM

**Principe :**

> La traçabilité permet de répondre : "D'où vient ce champ ?", "Quel code l'a produit ?", "Si je change une source, quels DM sont impactés ?"

**Trois niveaux de lineage :**

1. **Dataset-level** (rapide à mettre en place)
   - table source → job → table cible

2. **Column-level** (gouvernance)
   - colonne source → transformation → colonne cible

3. **Run-level** (opérationnel/audit)
   - run_id, timestamp, durée, statut, volumes, version code

**Métadonnées minimales par exécution (run-level) :**

```json
{
  "run_id": "run_20260211_100532",
  "pipeline_name": "emails_bronze_to_silver",
  "task_name": "parse_mime_emails",
  "start_time": "2026-02-11T10:05:32Z",
  "end_time": "2026-02-11T10:12:45Z",
  "duration_seconds": 433,
  "status": "SUCCESS",
  "input_datasets": [
    {
      "uri": "s3://bronze/emails_raw/date=2026-02-11/",
      "partition": "date=2026-02-11",
      "row_count": 12500
    }
  ],
  "output_datasets": [
    {
      "uri": "dwh.emails.fact_emails",
      "partition": "date=2026-02-11",
      "row_count": 12489,
      "rejected_count": 11
    }
  ],
  "git_commit": "a3f2b1c",
  "code_version": "v2.3.1",
  "quarantine_count": 11,
  "error_message": null
}
```

**Pourquoi la traçabilité est critique :**
- Debugging rapide (identifier la source d'une erreur)
- Impact analysis (quels DM sont affectés par un changement)
- Audit et conformité (RGPD, réglementation)
- Reproductibilité (rejouer une exécution exacte)

---

## 3. Module ETL et transformations

### 3.1 Responsabilités du module ETL

**Principe fondamental :**

> Tout ce qui est déterministe, stable et indépendant de la question utilisateur doit être pré-calculé dans l'ETL, jamais dans le LLM.

**Rôle de l'ETL :**
- Intégrer la donnée
- Appliquer les règles déterministes
- Produire des datasets publiables (DW/DM)
- Produire des vues contractuelles IA-ready
- Traçabilité (run_id, code_version, inputs/outputs, volumes, rejets)

**Ce qui doit être dans l'ETL (jamais dans le LLM) :**
- Calculs de dates (ancienneté, délais, échéances)
- Calculs de seuils (dépassement, limites)
- Flags métier (éligibilité, statut, urgence)
- Agrégations déterministes (sommes, moyennes, comptages)

### 3.2 Orchestration ETL

**Composants requis :**
- **Scheduler** (Airflow, Prefect, Dagster)
- **DAG** (Directed Acyclic Graph) pour dépendances
- **Gestion des reprises** (idempotence, checkpoints)
- **Monitoring** (durée, volumétrie, erreurs)

**Principes d'orchestration :**

1. **Idempotence :** même entrée → même sortie (reproductibilité)
2. **Incrémentalité :** traiter uniquement les données nouvelles/modifiées
3. **Parallélisation :** tâches indépendantes en parallèle
4. **Isolation :** échec d'une tâche ne bloque pas les autres

**Exemple de DAG Airflow (simplifié) :**

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime

def ingest_emails():
    # Ingestion depuis sources
    pass

def parse_emails():
    # Parsing MIME
    pass

def validate_quality():
    # Contrôles DQ
    pass

def load_dwh():
    # Chargement DWH
    pass

def project_ai_views():
    # Service de projection IA
    pass

with DAG('emails_etl', start_date=datetime(2026, 2, 11), schedule_interval='@daily') as dag:
    
    task_ingest = PythonOperator(task_id='ingest', python_callable=ingest_emails)
    task_parse = PythonOperator(task_id='parse', python_callable=parse_emails)
    task_validate = PythonOperator(task_id='validate', python_callable=validate_quality)
    task_load = PythonOperator(task_id='load_dwh', python_callable=load_dwh)
    task_project = PythonOperator(task_id='project_ai', python_callable=project_ai_views)
    
    task_ingest >> task_parse >> task_validate >> task_load >> task_project
```

### 3.3 Versioning des transformations

**Pourquoi versionner :**
- Reproductibilité (rejouer une transformation passée)
- Audit (quelle version a produit quelle donnée)
- Rollback (revenir à une version antérieure en cas d'erreur)
- A/B testing (comparer deux versions de transformation)

**Ce qui doit être versionné :**
1. Code de transformation (Git commit hash)
2. Paramètres de configuration (seuils, règles métier)
3. Schémas de données (input/output)
4. Dépendances (versions des bibliothèques)

**Modèle de données (table etl_transformation_registry) :**

| transformation_id | version | git_commit | config_json | schema_version | deployed_at |
|-------------------|---------|------------|-------------|----------------|-------------|
| parse_mime_emails | v2.3.1 | a3f2b1c | {"encoding": "utf-8"} | v1.2 | 2026-02-10 |
| classify_emails | v1.5.0 | b4e3d2f | {"threshold": 0.7} | v1.1 | 2026-02-05 |

**Pattern de déploiement :**

1. **Développement** (branche feature)
2. **Tests** (environnement staging, données de test)
3. **Validation** (comparaison résultats vs version précédente)
4. **Déploiement progressif** (canary : 10% → 100%)
5. **Monitoring post-déploiement** (alertes sur métriques)

---

## 4. Bases de connaissances et règles métier

### 4.1 Distinction : règles déterministes vs connaissances documentaires

**Règles déterministes (exécutables) :**
- Conditions if/then claires
- Testables automatiquement
- Versionnées avec dates d'effet
- Exemples : "Si ancienneté < 12 mois alors non éligible", "Si urgence = true alors priorité = haute"

**Connaissances documentaires (citables) :**
- Procédures, FAQ, manuels
- Non exécutables directement
- Utilisées via RAG
- Exemples : guides utilisateur, documentation produit

**Tableau comparatif (règles métier vs RAG vs Fine-tuning) :**

| Critère | Règles métier | RAG | Fine-tuning |
|---------|---------------|-----|-------------|
| Nature du savoir | Déterministe | Textuel/documentaire | Statistique |
| Fréquence de changement | Élevée | Moyenne | Faible |
| Explicabilité | ★★★★★ | ★★★★ | ★ |
| Traçabilité | Totale | Bonne | Faible |
| Coût initial | Faible | Moyen | Élevé |
| Coût récurrent | Faible | Moyen | Élevé |
| Risque d'hallucination | Nul | Faible | Moyen |
| Temps réel | Excellent | Bon | Bon |
| Maintenance | Simple | Modérée | Complexe |
| Audit/conformité | Excellente | Bonne | Difficile |

**Règle d'or d'architecture :**

> Ce qui est calculable → règles  
> Ce qui est documentable → RAG  
> Ce qui est implicite → fine-tuning

### 4.2 Versioning des règles métier

**Nécessité :**
- Les règles métier changent (nouvelles lois, politiques internes)
- Une décision IA doit être reproductible (audit)
- Besoin de savoir quelle version de règle a été appliquée

**Modèle de données (table rules_registry) :**

| rule_id | version | valid_from | valid_to | status | rule_text | tags_json | embedding | created_by | approved_by | git_commit |
|---------|---------|------------|----------|--------|-----------|-----------|-----------|------------|-------------|------------|
| R001 | v1.0.0 | 2025-01-01 | 2026-01-15 | DEPRECATED | "Si client déménage..." | {"domaine": "demenagement"} | [0.12, ...] | user1 | manager1 | abc123 |
| R001 | v2.0.0 | 2026-01-16 | 9999-12-31 | ACTIVE | "Si client déménage..." | {"domaine": "demenagement"} | [0.14, ...] | user2 | manager1 | def456 |

**Règles de gestion :**
1. Une seule version ACTIVE par rule_id à un instant T
2. Pas de chevauchement (valid_from de v2 = valid_to de v1)
3. Historisation complète (jamais de suppression physique)
4. Audit trail (qui a modifié quoi et quand)

**Utilisation dans le RAG :**
- Filtrage par `valid_from <= now() AND valid_to > now()`
- Stockage de la version utilisée dans les logs (`rule_id + version`)

**Migration de règles :**

1. **Création nouvelle version** (status=DRAFT)
2. **Validation métier** (tests, revue)
3. **Activation** (status=ACTIVE, mise à jour valid_from/valid_to)
4. **Dépréciation ancienne version** (status=DEPRECATED)
5. **Archivage** après période de grâce (status=ARCHIVED)

### 4.3 Tags JSON pour filtrage

**Pourquoi les tags sont utiles :**
- Filtrage rapide avant RAG
- Classification sémantique
- Audit et maintenance
- Extensible

**Exemple JSON pour une règle métier :**

```json
{
  "rule_id": "R001",
  "version": "v2.0.0",
  "texte": "Si un client déménage, cela est gratuit si aucun déménagement dans les 12 derniers mois",
  "tags": {
    "domaine": ["demenagement", "resiliation"],
    "type_client": ["particulier"],
    "produit": ["fibre", "adsl"],
    "condition": ["gratuite", "delai"]
  }
}
```

**Bonnes pratiques pour les tags :**
- Format JSON ou table relationnelle séparée
- Standardisation des tags (éviter synonymie)
- Granularité modérée
- Hiérarchie ou catégories
- Indexation côté BDD (ex: Oracle JSON_EXISTS)

**Architecture filtrage + RAG :**

```
Question + Faits client
          │
          ▼
Filtrage tags JSON/SQL  ← Règles avec tags
          │
          ▼
Top N règles + embeddings → RAG / LLM
          │
          ▼
Réponse finale
```

---

## 5. Module RAG (Retrieval-Augmented Generation)

### 5.1 Rôle et périmètre du RAG

**Rôle :** Enrichir le contexte du LLM avec des connaissances documentaires pertinentes.

**Périmètre d'application :**
- Règles métier volumineuses (>100 règles)
- Documentation technique (manuels, procédures)
- FAQ métier (questions/réponses pré-établies)
- Base de connaissances évolutive

**Exclusions (ne pas utiliser RAG pour) :**
- Données clients temps réel (utiliser ETL + faits métier)
- Calculs dynamiques (utiliser module calculs métier)
- Décisions critiques (utiliser règles déterministes)

### 5.2 Architecture RAG : deux pipelines distincts

**Pipeline 1 : Indexation (asynchrone, batch)**

**Déclenchement :** création/modification de documents  
**Fréquence :** quotidienne ou event-driven

**Étapes :**

1. **Extraction de contenu** (parsing PDF/Word/HTML)
2. **Nettoyage** (suppression bruit, headers, footers)
3. **Chunking** (découpage sémantique)
4. **Génération embeddings** (modèle text-embedding)
5. **Enrichissement métadonnées** (tags, version, source, date)
6. **Indexation dans vector store** (upsert avec deduplication)
7. **Contrôles qualité** :
   - Vérification complétude
   - Détection doublons
   - Validation métadonnées

**Pipeline 2 : Requêtage (synchrone, temps réel)**

**Déclenchement :** question utilisateur nécessitant contexte documentaire

**Étapes :**

1. **Analyse de la question** (query understanding)
   - Détection intention
   - Extraction mots-clés métier
   - Reformulation si nécessaire

2. **Filtrage métadonnées** (pré-retrieval)
   - Filtres obligatoires : version ACTIVE, domaine métier
   - Filtres conditionnels : produit, type client, langue

3. **Recherche hybride** (recommandé)
   a. **Dense retrieval** (vecteurs sémantiques)
      - Calcul similarité cosinus
      - Top-K candidats (ex: K=20)
   b. **Sparse retrieval** (keywords, BM25)
      - Recherche mots-clés exacts
      - Top-M candidats (ex: M=20)
   c. **Fusion** (Reciprocal Rank Fusion)
      - Combinaison scores dense + sparse
      - Top-N final (ex: N=10)

4. **Reranking** (optionnel mais recommandé)
   - Modèle cross-encoder
   - Re-scoring des N candidats
   - Sélection Top-P (ex: P=5)
   - Gain typique : +20-35% précision

5. **Assemblage contexte**
   - Concaténation P chunks sélectionnés
   - Ajout métadonnées (source, version, date)
   - Génération citations
   - Vérification taille totale (limite tokens LLM)

6. **Caching** (optimisation performance/coût)
   - Cache requêtes (query → résultats, TTL=1h)
   - Cache embeddings (questions fréquentes)
   - Taux de hit attendu : 60-80% en production

### 5.3 Stratégie de chunking

**Pourquoi chunker :**
- Une règle trop longue dilue la pertinence sémantique
- Une règle trop courte perd le contexte
- Objectif : unité sémantique cohérente et autonome

**Stratégies de chunking (par ordre de préférence) :**

1. **Chunking sémantique** (recommandé)
   - Découper par paragraphe logique
   - Une règle = un chunk
   - Avantage : pertinence maximale

2. **Chunking par délimiteur**
   - Découper sur marqueurs (titres, numérotation)
   - Exemple : "Article 3.2.1" = nouveau chunk
   - Avantage : respect structure documentaire

3. **Chunking fixed-size** (dernier recours)
   - Découper tous les N tokens (ex: 512 tokens)
   - Avec overlap (ex: 50 tokens)
   - Inconvénient : peut couper au milieu d'une règle

**Métadonnées par chunk (essentielles pour filtrage) :**

```json
{
  "chunk_id": "R123_v1.0.0_chunk_0",
  "rule_id": "R123",
  "version": "v1.0.0",
  "chunk_index": 0,
  "chunk_text": "Si un client déménage, cela est gratuit si aucun déménagement dans les 12 derniers mois.",
  "tags_json": {
    "domaine": "demenagement",
    "condition": "gratuite",
    "delai": "12_mois"
  },
  "embedding": [0.123, -0.456, ...],
  "char_count": 92,
  "token_count": 25,
  "created_at": "2026-02-01T10:00:00Z"
}
```

### 5.4 Évaluation RAG (qualité)

**Métriques offline (jeux de test annotés) :**

| Métrique | Description | Seuil attendu |
|----------|-------------|---------------|
| Context Precision | Pertinence des chunks récupérés | >0.7 |
| Context Recall | Complétude des chunks pertinents | >0.8 |
| Faithfulness | Réponse LLM fidèle au contexte | >0.85 |
| Answer Relevance | Réponse répond à la question | >0.9 |

**Métriques online (production) :**

| Métrique | Description | Objectif |
|----------|-------------|----------|
| Latence retrieval | Temps de récupération | P95 <500ms |
| Latence reranking | Temps de re-scoring | P95 <200ms |
| Taux contexte vide | % requêtes sans résultat | <5% |
| Feedback utilisateur | Thumbs up/down | >80% positif |

**Outils d'évaluation :**
- RAGAS (framework open-source)
- LangSmith (LangChain)
- Arize AI (plateforme monitoring)

**Source :** Framework RAGAS pour évaluation RAG

### 5.5 Sécurité RAG : permission-aware retrieval

**Principe :**

> Les documents récupérés doivent respecter les droits de l'utilisateur/du service.

**Implémentation :**

1. **Filtrage par tags** (métadonnées de sécurité)
   - classification (public, confidentiel, secret)
   - groupes autorisés
   - rôles métier

2. **Politiques d'accès** (ABAC/RBAC)
   - Attribute-Based Access Control
   - Role-Based Access Control

**Exemple de filtrage sécurité :**

```python
def retrieve_with_security(query, user_id, user_groups):
    # Récupération profil utilisateur
    user_clearance = get_user_clearance(user_id)
    
    # Filtrage par classification
    filter_conditions = {
        "classification": {"$lte": user_clearance},
        "authorized_groups": {"$in": user_groups}
    }
    
    # Requête RAG avec filtres sécurité
    results = vector_store.search(
        query_embedding=embed(query),
        filter=filter_conditions,
        top_k=10
    )
    
    return results
```

**Pourquoi c'est critique :**
- Conformité RGPD et réglementaire
- Prévention fuites de données
- Audit trail des accès

---

## 6. Modules de modélisation (ML et LLM)

### 6.1 Module Machine Learning classique

**Rôle :** Modèles prédictifs sur données structurées (alternative rapide et déterministe au LLM).

**Cas d'usage typiques :**
- Classification (emails, tickets, sentiments)
- Scoring (risque, churn, propension)
- Régression (prédiction montants, durées)
- Détection d'anomalies (fraude, comportements atypiques)
- Ranking (priorisation, recommandation)

**Avantages vs LLM :**

| Critère | ML classique | LLM |
|---------|--------------|-----|
| Latence | <10ms | 500ms-2s |
| Coût | Quasi-nul | Élevé (tokens) |
| Explicabilité | Élevée (SHAP, LIME) | Faible |
| Déterminisme | Total | Probabiliste |
| Conformité | Facile à auditer | Complexe |

**Cycle de vie ML (MLOps) :**

1. **Développement** (environnement DEV)
   - Feature engineering (depuis DM ou Feature Store)
   - Entraînement (GPU/CPU)
   - Validation (hold-out, cross-validation)
   - Hyperparameter tuning (Grid Search, Bayesian Opt)
   - Experiment tracking (MLflow, Weights & Biases)

2. **Qualification** (environnement STAGING)
   - Tests sur données de production (shadow mode)
   - Validation performance (vs baseline, vs modèle précédent)
   - Tests de robustesse (données bruitées, edge cases)
   - Tests de biais (fairness, disparate impact)
   - Validation métier (revue avec experts domaine)

3. **Déploiement** (environnement PROD)
   - Registry : enregistrement dans model registry (MLflow)
   - Promotion : passage STAGING → PRODUCTION (avec approbation)
   - Déploiement :
     - Online (API REST/gRPC, faible latence)
     - Batch (scoring planifié, volumétries importantes)
     - Streaming (Kafka, événements temps réel)
   - Stratégies de déploiement :
     - Canary : 10% trafic → validation → 100%
     - Blue/Green : deux environnements, switch instantané
     - Shadow : nouveau modèle en parallèle, sans impact

4. **Monitoring** (continu)
   - Performance métier (Accuracy, Precision, Recall, F1, AUC-ROC)
   - Data drift (distribution features : KS test, PSI, Wasserstein)
   - Concept drift (dégradation performance)
   - Latence (P50, P95, P99)
   - Métriques métier (taux de conversion, etc.)

5. **Retraining** (automatisé ou manuel)
   - Déclencheurs :
     - Drift détecté
     - Performance dégradée
     - Planning régulier (ex: mensuel)
     - Nouvelles données disponibles
   - Processus : extraction → réentraînement → validation → déploiement
   - Rollback : si nouveau modèle moins performant

**Exemple : classification d'emails avec ML classique**

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

# Feature engineering
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))

# Modèle
classifier = RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42)

# Pipeline
pipeline = Pipeline([
    ('tfidf', vectorizer),
    ('classifier', classifier)
])

# Entraînement
pipeline.fit(X_train, y_train)

# Prédiction
y_pred = pipeline.predict(X_test)

# Évaluation
print(classification_report(y_test, y_pred))

# Sauvegarde
import mlflow
mlflow.sklearn.log_model(pipeline, "email_classifier_v1")
```

**Monitoring du drift :**

```python
from scipy.stats import ks_2samp

def detect_data_drift(reference_data, current_data, threshold=0.05):
    """Détecte le drift sur les features numériques."""
    drift_report = {}
    
    for feature in reference_data.columns:
        # Test de Kolmogorov-Smirnov
        statistic, p_value = ks_2samp(
            reference_data[feature], 
            current_data[feature]
        )
        
        drift_detected = p_value < threshold
        drift_report[feature] = {
            'statistic': statistic,
            'p_value': p_value,
            'drift_detected': drift_detected
        }
    
    return drift_report

# Exemple d'utilisation
drift_report = detect_data_drift(X_train, X_production)
drifted_features = [f for f, r in drift_report.items() if r['drift_detected']]

if drifted_features:
    print(f"Drift détecté sur : {drifted_features}")
    # Déclencher retraining
```

### 6.2 Module Deep Learning / LLM

**Rôle :** Raisonnement, explication, génération de texte.

**Principe fondamental :**

> Le LLM doit raisonner sur des entrées contrôlées : faits structurés, contexte sélectionné, règles/documents citables. Il ne calcule jamais de faits métier.

**Types de LLM :**

| Type | Données | Ce qui est appris | Coût | Cas d'usage | Usage entreprise |
|------|---------|-------------------|------|-------------|------------------|
| LLM pré-entraîné | Corpus massif public | Connaissance générale | Faible (inférence) | Chatbot, Q&A | Très courant |
| LLM fine-tuning | Dataset ciblé (exemples métier) | Comportement, style, formats | Moyen à élevé | Extraction, classification, ton | Cas ciblés |
| LLM from scratch | Centaines de milliards de tokens | Langage, syntaxe, raisonnement | Très élevé | Modèle généraliste | Jamais |

**Registry et gestion des modèles LLM :**

**Modèle de données (table llm_registry) :**

| model_id | model_family | model_size | deployment_type | endpoint_url | context_window | cost_per_1k_tokens | latency_p95 | status | recommended_use_cases |
|----------|--------------|------------|-----------------|--------------|----------------|--------------------| ------------|--------|-----------------------|
| mistral-7b-v0.2 | Mistral | 7B | local | localhost:8000 | 32k | 0 | 800ms | ACTIVE | ["chatbot", "classification"] |
| gpt-4o-mini | OpenAI | unknown | API externe | api.openai.com | 128k | 0.15 | 1200ms | ACTIVE | ["extraction", "raisonnement"] |

**Stratégie de sélection du modèle (dans orchestrateur) :**

1. **Par cas d'usage** :
   - Chatbot conversationnel : Mistral 7B ou DeepSeek 7B
   - Extraction structurée : GPT-4o-mini ou Mistral Small
   - Classification : modèle light ou ML classique

2. **Par contraintes** :
   - Latence stricte (<500ms) : modèle light local
   - Coût limité : modèle open-source auto-hébergé
   - Qualité maximale : modèle large (GPT-4, Claude 3.5)

**Logging (obligatoire pour chaque appel LLM) :**

```json
{
  "call_id": "llm_call_001",
  "model_id": "mistral-7b-v0.2",
  "model_version": "v0.2",
  "prompt_tokens": 1250,
  "completion_tokens": 180,
  "total_tokens": 1430,
  "latency_ms": 850,
  "cost_usd": 0.0,
  "prompt_hash": "a3f2b1c...",
  "response_hash": "d4e5f6a...",
  "created_at": "2026-02-11T10:30:00Z"
}
```

**Gestion des erreurs LLM :**

**Types d'erreurs :**

1. **Erreurs transitoires** (retry possible)
   - Timeout réseau
   - Rate limit (429)
   - Erreur serveur (500, 503)

2. **Erreurs permanentes** (retry inutile)
   - Authentification invalide (401)
   - Prompt trop long (400)
   - Contenu bloqué par modération (400)

**Stratégie de retry :**

```python
import time
import random

def call_llm_with_retry(prompt, max_retries=3):
    """Appelle le LLM avec retry exponentiel."""
    for attempt in range(max_retries):
        try:
            response = llm_api.call(prompt)
            return response
        except TransientError as e:
            if attempt < max_retries - 1:
                wait_time = (2 ** attempt) + random.uniform(0, 1)
                time.sleep(wait_time)
            else:
                raise
        except PermanentError as e:
            # Pas de retry
            raise
```

**Stratégie de fallback :**

1. **Fallback vers modèle secondaire** (ex: GPT-4 échoue → Mistral)
2. **Réponse dégradée** (si acceptable)
3. **Escalade humaine** (si critique)

**Circuit breaker :**

```python
class CircuitBreaker:
    def __init__(self, failure_threshold=0.5, timeout=30):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failures = 0
        self.successes = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
    
    def call(self, func, *args, **kwargs):
        if self.state == "OPEN":
            if time.time() - self.last_failure_time > self.timeout:
                self.state = "HALF_OPEN"
            else:
                raise CircuitBreakerOpenError("Circuit ouvert")
        
        try:
            result = func(*args, **kwargs)
            self.on_success()
            return result
        except Exception as e:
            self.on_failure()
            raise
    
    def on_success(self):
        self.successes += 1
        if self.state == "HALF_OPEN":
            self.state = "CLOSED"
            self.failures = 0
    
    def on_failure(self):
        self.failures += 1
        self.last_failure_time = time.time()
        
        total = self.failures + self.successes
        if total > 0 and self.failures / total > self.failure_threshold:
            self.state = "OPEN"
```

---

## 7. Orchestrateur IA

### 7.1 Rôle central de l'orchestrateur

**Principe :**

> L'orchestrateur est le cerveau non-IA. Le LLM n'est qu'un moteur de raisonnement.

**Responsabilités clés :**

| Domaine | Rôle |
|---------|------|
| Flux | Gérer les étapes |
| Sécurité | Filtrer données |
| Qualité | Bloquer incohérences |
| Performance | Éviter appels inutiles |
| Gouvernance | Tracer tout |

**Pipeline type orchestré :**

```
Question client
    ↓
Analyse intention
    ↓
Récupération faits métier (ETL)
    ↓
Décision RAG (oui / non)
    ↓
Construction prompt
    ↓
Appel LLM
    ↓
Validation réponse
    ↓
Sortie + logs
```

**Décisions prises par l'orchestrateur (PAS le LLM) :**
- Faut-il du RAG ?
- Quel modèle LLM ?
- Quelle version de règles ?
- Résumé ou contexte complet ?
- Réponse bloquante ou informative ?

### 7.2 Architecture interne de l'orchestrateur

L'orchestrateur n'est pas un monolithe. Il est composé de sous-modules spécialisés.

**Sous-modules (composants internes) :**

1. **Request Handler** (entrée)
   - Réception requête (HTTP, WebSocket, MQ)
   - Validation format (schema JSON, auth)
   - Extraction métadonnées (user_id, session_id, channel)
   - Génération request_id (traçabilité)

2. **Context Loader** (données)
   - Récupération faits métier (depuis ETL/DM)
   - Récupération contexte client (depuis DM)
   - Récupération historique conversation (si chatbot)
   - Assemblage structure unifiée

3. **Intent Analyzer** (classification légère)
   - Détection intention principale (via règles ou ML light)
   - Détection sous-intentions
   - Détection urgence, ton, sentiment
   - Sortie : intent_json (structuré)

4. **Strategy Selector** (décision)
   - Choix du pipeline (classification, RAG, LLM, ML)
   - Décision RAG (nécessaire ou non)
   - Sélection modèle LLM (selon cas d'usage, budget, latence)
   - Sortie : execution_plan (DAG de tâches)

5. **RAG Controller** (si RAG nécessaire)
   - Filtrage métadonnées (pré-retrieval)
   - Appel module RAG (retrieval + reranking)
   - Sélection Top-P chunks
   - Injection dans contexte

6. **Prompt Builder** (construction)
   - Template de prompt (selon cas d'usage)
   - Injection faits métier (format contractuel)
   - Injection contexte client
   - Injection règles RAG (si applicable)
   - Injection instructions système
   - Validation taille (tokens < context_window)

7. **LLM Gateway** (appel modèle)
   - Routage vers bon modèle
   - Gestion retry/fallback (si erreur)
   - Circuit breaker (protection)
   - Logging appel (prompt, réponse, latence, coût)

8. **Response Validator** (post-traitement)
   - Validation cohérence (réponse vs faits métier)
   - Détection hallucination
   - Validation format (si JSON attendu)
   - Filtrage contenu inapproprié
   - Enrichissement (ajout citations, sources)

9. **Response Formatter** (sortie)
   - Formatage selon canal (JSON, texte, audio)
   - Ajout métadonnées (confidence, sources, règles appliquées)
   - Masquage données sensibles (RGPD)

10. **Logger & Metrics** (observabilité)
    - Logging complet (request, execution_plan, réponse, erreurs)
    - Métriques (latence par étape, taux de succès, coût)
    - Traces distribuées (OpenTelemetry)
    - Export vers monitoring (Prometheus, Grafana)

**Architecture logique (DAG d'exécution) :**

```
Request
    ↓
[Request Handler] → request_id, metadata
    ↓    
[Context Loader] → faits_metier, contexte_client
    ↓
[Intent Analyzer] → intent_json
    ↓
[Strategy Selector] → execution_plan
    ↓
[RAG Controller] (optionnel) → contexte_enrichi
    ↓
[Prompt Builder] → prompt_final
    ↓
[LLM Gateway] → reponse_brute
    ↓
[Response Validator] → reponse_validee
    ↓
[Response Formatter] → reponse_finale
    ↓
[Logger & Metrics] → stockage + alertes
```

### 7.3 Exemple de logique d'orchestration (pseudo-code)

```python
def orchestrate(question, client_id, user_id):
    # 1. Request handling
    request_id = generate_request_id()
    metadata = extract_metadata(user_id, channel="web")
    
    # 2. Context loading
    facts = get_facts(client_id)  # Depuis DM
    context = get_context(client_id)  # Depuis DM
    
    # 3. Intent analysis
    intent = analyze_intent(question)  # ML light ou règles
    
    # 4. Strategy selection
    if facts["client_vip"]:
        priority = "HIGH"
        model_id = "gpt-4o-mini"
    else:
        priority = "NORMAL"
        model_id = "mistral-7b-v0.2"
    
    need_rag = detect_need_rag(question, intent)
    
    # 5. RAG (si nécessaire)
    rules = []
    if need_rag:
        rules = retrieve_rules(
            query=question,
            context=context,
            user_groups=get_user_groups(user_id)
        )
    
    # 6. Prompt building
    prompt = build_prompt(
        question=question,
        facts=facts,
        context=context,
        rules=rules,
        intent=intent
    )
    
    # 7. LLM call
    response = call_llm(
        model_id=model_id,
        prompt=prompt,
        max_retries=3
    )
    
    # 8. Response validation
    validated = validate_response(response, facts)
    
    if not validated["is_valid"]:
        # Fallback ou escalade
        response = generate_fallback_response(question, facts)
    
    # 9. Logging
    log_all(
        request_id=request_id,
        question=question,
        facts=facts,
        rules=rules,
        response=response,
        validated=validated,
        model_id=model_id,
        latency_ms=calculate_latency(),
        cost_usd=calculate_cost(response)
    )
    
    # 10. Response formatting
    formatted = format_response(
        response=validated["response"],
        sources=rules,
        confidence=validated["confidence"]
    )
    
    return formatted
```

### 7.4 Validation post-LLM (obligatoire)

**Exemples de validations :**
- Incohérence faits / réponse
- Règle ignorée
- Hallucination (réponse contredit faits)
- Non-respect du format

```python
def validate_response(response, facts):
    """Valide la cohérence de la réponse du LLM."""
    is_valid = True
    errors = []
    
    # Validation 1 : Format JSON (si attendu)
    try:
        response_json = json.loads(response)
    except json.JSONDecodeError:
        is_valid = False
        errors.append("Format JSON invalide")
        return {"is_valid": is_valid, "errors": errors}
    
    # Validation 2 : Cohérence avec faits métier
    if "eligible_gratuite" in response_json:
        if response_json["eligible_gratuite"] == True and facts["eligible_gratuite"] == False:
            is_valid = False
            errors.append("Incohérence : réponse dit 'éligible' mais faits disent 'non éligible'")
    
    # Validation 3 : Présence champs obligatoires
    required_fields = ["decision", "justification"]
    for field in required_fields:
        if field not in response_json:
            is_valid = False
            errors.append(f"Champ obligatoire manquant : {field}")
    
    return {
        "is_valid": is_valid,
        "errors": errors,
        "response": response_json,
        "confidence": calculate_confidence(response_json)
    }
```

### 7.5 Patterns de résilience

- **Timeout** : chaque étape a un timeout max
- **Retry** : erreurs transitoires (3 tentatives max)
- **Fallback** : si LLM échoue, réponse dégradée
- **Circuit breaker** : si module défaillant, le désactiver temporairement
- **Bulkhead** : isoler ressources (pool de connexions séparé par module)

---

## 8. Modules de gouvernance et monitoring

### 8.1 Module Résultats et Historique (Output Store)

**Rôle :** Stocker toutes les interactions IA pour audit, conformité et amélioration continue.

**Schéma de données (table ai_interaction_log) :**

| Colonne | Type | Description |
|---------|------|-------------|
| interaction_id | UUID | Identifiant unique |
| request_id | UUID | Lien avec logs techniques |
| user_id | VARCHAR | Utilisateur (anonymisé si RGPD) |
| session_id | VARCHAR | Conversation (si chatbot) |
| channel | VARCHAR | web, mobile, call_center, email, api |
| timestamp | TIMESTAMP | Date/heure précise |
| intent | VARCHAR | Intention détectée |
| question_text | TEXT | Question originale (peut être masquée selon RGPD) |
| context_used_json | JSON | Contexte injecté (faits, règles RAG, historique) |
| execution_plan_json | JSON | Stratégie appliquée (modules appelés) |
| response_text | TEXT | Réponse générée |
| response_confidence | FLOAT | Score de confiance |
| models_used | JSON | Liste des modèles (LLM, ML, règles) |
| latency_total_ms | INT | Latence totale |
| cost_usd | FLOAT | Coût (si LLM payant) |
| status | VARCHAR | SUCCESS, FAILED, TIMEOUT, PARTIAL |
| error_message | TEXT | Si échec |
| feedback | VARCHAR | thumbs_up/down (si collecté) |
| retention_until | DATE | Date de suppression (RGPD) |

**Rétention des données (RGPD) :**

**Règles de rétention :**

1. **Logs techniques** (ai_interaction_log)
   - Conservation : 90 jours (sauf obligation légale)
   - Anonymisation : après 30 jours (user_id remplacé par hash)
   - Suppression : après 90 jours (sauf marqué "audit")

2. **Décisions critiques** (ai_decision_audit)
   - Conservation : selon réglementation (ex: 3 ans bancaire, 5 ans assurance)
   - Anonymisation : dès que possible (user_id pseudonymisé)

3. **Feedback utilisateurs**
   - Conservation : 1 an (amélioration continue)
   - Anonymisation : immédiate (dissocie du user_id)

**Processus de purge automatisé :**

```python
from datetime import datetime, timedelta

def purge_logs():
    """Purge automatique des logs expirés."""
    today = datetime.now()
    
    # Anonymisation après 30 jours
    cutoff_anonymize = today - timedelta(days=30)
    db.execute("""
        UPDATE ai_interaction_log
        SET user_id = SHA256(user_id),
            question_text = '[ANONYMIZED]'
        WHERE timestamp < %s
          AND user_id NOT LIKE 'anon_%'
    """, [cutoff_anonymize])
    
    # Suppression après 90 jours
    cutoff_delete = today - timedelta(days=90)
    db.execute("""
        DELETE FROM ai_interaction_log
        WHERE timestamp < %s
          AND status != 'AUDIT_REQUIRED'
    """, [cutoff_delete])
    
    # Archivage cold storage (si obligation légale)
    archive_to_s3_glacier(cutoff_delete)
```

### 8.2 Module Logs / Traces / Monitoring

**Séparer strictement :**

1. **Output Store** (résultats métier)
   - Décisions, scores, justifications, références
   - Consultable pour KPI et audit métier
   - Rétention alignée sur besoin métier et conformité

2. **Observability Store** (logs/traces techniques)
   - Traces d'exécution, latences, erreurs, timeouts, fallbacks, métriques
   - Accès restreint (sécurité), rétention courte par défaut
   - Corrélation via correlation_id / request_id

**Architecture monitoring :**
```
[Application]
    ↓ (logs structurés JSON)
[Collector] (Fluentd, Logstash)
    ↓
[Storage] (Elasticsearch, Loki)
    ↓
[Visualization] (Kibana, Grafana)
    ↓
[Alerting] (Prometheus Alertmanager)
```

**Métriques clés à monitorer :**

| Métrique | Description | Objectif |
|----------|-------------|----------|
| Latence totale | Temps de bout en bout | P95 <2s |
| Latence par étape | Identifier goulots d'étranglement | - |
| Taux de succès | % requêtes réussies | >99% |
| Taux d'erreur | % requêtes échouées | <1% |
| Coût par requête | Tokens LLM + compute | Budgété |
| Taux d'utilisation RAG | % requêtes nécessitant RAG | - |

**Exemple de dashboard Grafana (requête Prometheus) :**

```promql
# Latence P95 par étape
histogram_quantile(0.95, 
  sum(rate(orchestrator_step_duration_seconds_bucket[5m])) by (le, step)
)

# Taux d'erreur par modèle
sum(rate(llm_call_errors_total[5m])) by (model_id)
/ 
sum(rate(llm_calls_total[5m])) by (model_id)

# Coût par heure
sum(rate(llm_cost_usd_total[1h]))
```

### 8.3 Module Auto-validation / Qualité IA

**Principe :**

> C'est l'équivalent des tests unitaires et de non-régression pour l'IA.

**Jeux de tests IA :**

```json
{
  "test_id": "TEST-RESI-001",
  "question": "Je déménage, est-ce gratuit ?",
  "faits_metier": {
    "demenagement_12_mois": true
  },
  "expected": {
    "service": "Résiliation",
    "eligibilite_gratuite": false
  }
}
```

**Tests automatisés :**

**Types de tests :**

1. **Tests de non-régression**
   - Mêmes entrées → mêmes sorties attendues
   - À chaque changement de prompt / modèle

2. **Tests de robustesse**
   - Reformulations
   - Bruit
   - Phrases incomplètes

3. **Tests métier**
   - Cas limites
   - Règles contradictoires
   - Priorités

```python
import pytest

class TestEmailClassification:
    
    def test_classification_resiliation(self):
        """Test classification email résiliation."""
        email = {
            "sujet": "Demande de résiliation",
            "corps": "Je souhaite résilier mon contrat"
        }
        
        result = orchestrate(email, client_id="client_001")
        
        assert result["categorie"] == "resiliation"
        assert result["confidence"] > 0.8
    
    def test_classification_robustesse(self):
        """Test robustesse avec reformulations."""
        emails = [
            "Je veux résilier",
            "Résiliation svp",
            "Annuler mon contrat"
        ]
        
        for email in emails:
            result = orchestrate({"sujet": email}, client_id="client_001")
            assert result["categorie"] == "resiliation"
```

**Métriques de qualité :**

| Indicateur | Objectif |
|------------|----------|
| Taux de réponses valides | >99% |
| Respect des faits métier | 100% |
| Stabilité intention | >95% |
| Latence P95 | <2s |
| Hallucinations | 0 tolérance |

**Boucle d'amélioration continue :**

```
Résultats IA
    ↓
Feedback humain
    ↓
Jeux de tests enrichis
    ↓
Auto-validation
    ↓
Ajustement prompts / règles
```

### 8.4 Module Gouvernance & Sécurité IA

**Rôle :**
- Contrôle des accès
- Masquage des données sensibles, anonymisation
- RGPD
- Traçabilité des accès
- Gestion des droits par rôle
- Audit trails

**Sécurité technique (obligatoire) :**
- Chiffrement données au repos
- Chiffrement en transit (TLS 1.3)
- Contrôle d'accès par rôle (RBAC)
- Séparation environnements (POC / PROD)
- Détection d'anomalies ou d'usage hors règles
- Gestion des secrets et clés (rotation, séparation des rôles)
- Segmentation réseau (zones de confiance)

**RGPD – principes IA-safe :**

| Type | Règle |
|------|-------|
| Données client | Jamais dans embeddings (sauf scopes contrôlés) |
| Logs | Pseudonymisés |
| Prompts | Pas de données sensibles |
| Résultats | Traçables |

**Masquage avant LLM :**

```json
{
  "nom": "[CLIENT]",
  "telephone": "[MASKED]",
  "email": "[MASKED]",
  "adresse": "[MASKED]"
}
```

**Auditabilité :**

> Tu dois pouvoir répondre : "Pourquoi l'IA a donné cette réponse le 12/03 à 14h32 ?"

Requiert : prompts, règles, version modèle, faits utilisés.

**Contrôles d'accès (exemple) :**

```python
from functools import wraps

def require_clearance(clearance_level):
    """Décorateur pour contrôle d'accès."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            user_clearance = get_user_clearance(kwargs['user_id'])
            if user_clearance < clearance_level:
                raise PermissionDeniedError(
                    f"Clearance {clearance_level} requise"
                )
            return func(*args, **kwargs)
        return wrapper
    return decorator

@require_clearance(clearance_level=3)
def access_sensitive_data(user_id, data_id):
    """Accès données sensibles."""
    return db.query("SELECT * FROM sensitive_data WHERE id = %s", [data_id])
```

---

## 9. Cas pratique : Classification d'emails

### 9.1 Vue d'ensemble du cas d'usage

**Objectif :** Classifier automatiquement des emails entrants selon des catégories métier (résiliation, facturation, technique, commercial, etc.).

**Données d'entrée :** Emails bruts au format MIME (.eml).

**Données de sortie :** Catégorie + confidence + justification.

**Architecture complète DL → DW → DM → IA :**

```
[Emails .eml]
    ↓
[Bronze/DL] ← Stockage brut .eml
    ↓
[Parsing MIME] ← Extraction headers + corps
    ↓
[Validation/Quarantaine] ← Contrôles qualité
    ↓
[Silver/DWH] ← Modélisation dimensionnelle
    ↓
[Service de projection IA] ← Vues contractuelles
    ↓
[Gold/DM] ← Datasets ML/IA
    ↓
[Orchestrateur IA]
    ├─ [ML Classifier] ← Classif rapide
    └─ [LLM] ← Raisonnement + explication
```

### 9.2 Étape 1 : Bronze / Data Lake

**Ingestion des emails .eml :**

```python
import hashlib
from datetime import datetime

def ingest_email(eml_file, source="outlook"):
    """Ingeste un email .eml dans le Data Lake."""
    # Lecture du fichier
    with open(eml_file, 'rb') as f:
        eml_content = f.read()
    
    # Génération identifiant unique
    email_uid = hashlib.sha256(eml_content).hexdigest()
    
    # Métadonnées d'ingestion
    metadata = {
        "email_uid": email_uid,
        "source": source,
        "ingestion_timestamp": datetime.now().isoformat(),
        "file_size_bytes": len(eml_content)
    }
    
    # Stockage dans Data Lake
    output_path = f"s3://bronze/emails_raw/source={source}/date={datetime.now().date()}/{email_uid}.eml"
    s3_client.put_object(
        Bucket="bronze",
        Key=output_path,
        Body=eml_content,
        Metadata=metadata
    )
    
    # Enregistrement dans manifest
    db.execute("""
        INSERT INTO dl_emails_ingest_manifest 
        (email_uid, path, source, ingestion_timestamp, file_size_bytes, status)
        VALUES (%s, %s, %s, %s, %s, %s)
    """, [email_uid, output_path, source, metadata["ingestion_timestamp"], 
          metadata["file_size_bytes"], "INGESTED"])
    
    return email_uid
```

### 9.3 Étape 2 : Parsing MIME et transformation vers Silver/DWH

**Parsing MIME (.eml) :**

```python
import email
from email import policy
from email.parser import BytesParser

def parse_mime_email(eml_path):
    """Parse un email .eml et extrait les informations structurées."""
    with open(eml_path, 'rb') as f:
        msg = BytesParser(policy=policy.default).parse(f)
    
    # Extraction headers
    headers = {
        "message_id": msg.get("Message-ID"),
        "from": msg.get("From"),
        "to": msg.get("To"),
        "cc": msg.get("Cc"),
        "subject": msg.get("Subject"),
        "date": msg.get("Date"),
        "in_reply_to": msg.get("In-Reply-To"),
        "references": msg.get("References")
    }
    
    # Extraction corps
    body_text = None
    body_html = None
    
    if msg.is_multipart():
        for part in msg.walk():
            content_type = part.get_content_type()
            if content_type == "text/plain":
                body_text = part.get_content()
            elif content_type == "text/html":
                body_html = part.get_content()
    else:
        body_text = msg.get_content()
    
    # Extraction pièces jointes
    attachments = []
    for part in msg.iter_attachments():
        attachments.append({
            "filename": part.get_filename(),
            "content_type": part.get_content_type(),
            "size_bytes": len(part.get_content())
        })
    
    return {
        "headers": headers,
        "body_text": body_text,
        "body_html": body_html,
        "attachments": attachments
    }
```

**Nettoyage et normalisation :**

```python
import re
from bs4 import BeautifulSoup

def clean_email_body(body_text, body_html):
    """Nettoie et normalise le corps de l'email."""
    # Extraction texte depuis HTML si nécessaire
    if body_text is None and body_html is not None:
        soup = BeautifulSoup(body_html, 'html.parser')
        body_text = soup.get_text()
    
    if body_text is None:
        return None
    
    # Nettoyage
    text = body_text
    
    # Suppression signatures
    text = re.sub(r'--\s*\n.*', '', text, flags=re.DOTALL)
    
    # Suppression citations anciennes ("On ... wrote:")
    text = re.sub(r'On .* wrote:.*', '', text, flags=re.DOTALL)
    
    # Suppression espaces multiples
    text = re.sub(r'\s+', ' ', text)
    
    # Normalisation
    text = text.strip()
    
    return text

def extract_urls(text):
    """Extrait les URLs d'un texte."""
    url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
    urls = re.findall(url_pattern, text)
    return urls

def detect_language(text):
    """Détecte la langue d'un texte."""
    from langdetect import detect
    try:
        return detect(text)
    except:
        return None
```

**Calculs métier déterministes :**

```python
def compute_email_features(parsed_email, cleaned_body):
    """Calcule des features métier déterministes."""
    features = {}
    
    # Longueurs
    features["longueur_sujet"] = len(parsed_email["headers"]["subject"] or "")
    features["longueur_corps"] = len(cleaned_body or "")
    
    # Nombre de pièces jointes
    features["nb_pieces_jointes"] = len(parsed_email["attachments"])
    features["contient_pieces_jointes"] = features["nb_pieces_jointes"] > 0
    
    # URLs
    urls = extract_urls(cleaned_body or "")
    features["nb_urls"] = len(urls)
    features["contient_urls"] = len(urls) > 0
    
    # Langue
    features["langue_detectee"] = detect_language(cleaned_body)
    
    # Domaine expéditeur
    from_email = parsed_email["headers"]["from"]
    if from_email:
        domain = from_email.split("@")[-1].strip(">")
        features["domaine_expediteur"] = domain
        features["est_interne"] = domain in INTERNAL_DOMAINS
    else:
        features["domaine_expediteur"] = None
        features["est_interne"] = False
    
    # Urgence (heuristique simple)
    urgent_keywords = ["urgent", "immédiat", "rapidement", "asap"]
    features["urgent"] = any(kw in (cleaned_body or "").lower() for kw in urgent_keywords)
    
    return features
```

**Chargement dans DWH (Silver) :**

```python
def load_to_dwh(email_uid, parsed_email, cleaned_body, features):
    """Charge l'email parsé dans le DWH."""
    # Insertion dans fact_emails
    db.execute("""
        INSERT INTO dwh.fact_emails (
            email_uid, message_id, from_email, to_email, subject, 
            body_text, body_html, body_cleaned,
            timestamp, longueur_sujet, longueur_corps,
            nb_pieces_jointes, contient_pieces_jointes,
            nb_urls, contient_urls, langue_detectee,
            domaine_expediteur, est_interne, urgent
        ) VALUES (
            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
        )
    """, [
        email_uid,
        parsed_email["headers"]["message_id"],
        parsed_email["headers"]["from"],
        parsed_email["headers"]["to"],
        parsed_email["headers"]["subject"],
        parsed_email["body_text"],
        parsed_email["body_html"],
        cleaned_body,
        parsed_email["headers"]["date"],
        features["longueur_sujet"],
        features["longueur_corps"],
        features["nb_pieces_jointes"],
        features["contient_pieces_jointes"],
        features["nb_urls"],
        features["contient_urls"],
        features["langue_detectee"],
        features["domaine_expediteur"],
        features["est_interne"],
        features["urgent"]
    ])
    
    # Insertion pièces jointes dans dim_attachments
    for att in parsed_email["attachments"]:
        db.execute("""
            INSERT INTO dwh.dim_attachments (
                email_uid, filename, content_type, size_bytes
            ) VALUES (%s, %s, %s, %s)
        """, [email_uid, att["filename"], att["content_type"], att["size_bytes"]])
```

### 9.4 Étape 3 : Validation et quarantaine

**Contrôles de qualité :**

```python
def validate_email(parsed_email, cleaned_body):
    """Valide un email et décide de la quarantaine."""
    errors = []
    
    # Contrôle 1 : Headers obligatoires
    required_headers = ["from", "date", "subject"]
    for header in required_headers:
        if not parsed_email["headers"].get(header):
            errors.append(f"Header obligatoire manquant : {header}")
    
    # Contrôle 2 : Corps non vide
    if not cleaned_body or len(cleaned_body) < 10:
        errors.append("Corps vide ou trop court (<10 caractères)")
    
    # Contrôle 3 : Langue supportée
    lang = detect_language(cleaned_body)
    if lang not in SUPPORTED_LANGUAGES:
        errors.append(f"Langue non supportée : {lang}")
    
    # Contrôle 4 : Taille aberrante
    if len(cleaned_body or "") > 1000000:  # 1 MB
        errors.append("Corps trop long (>1MB)")
    
    if errors:
        return {"is_valid": False, "errors": errors}
    else:
        return {"is_valid": True, "errors": []}

def quarantine_email(email_uid, validation_result):
    """Met un email en quarantaine."""
    # Stockage en quarantaine
    db.execute("""
        INSERT INTO quarantine.emails (
            email_uid, errors_json, quarantine_timestamp
        ) VALUES (%s, %s, %s)
    """, [email_uid, json.dumps(validation_result["errors"]), datetime.now()])
    
    # Alerte
    send_alert(
        subject="Email mis en quarantaine",
        body=f"Email {email_uid} en quarantaine : {validation_result['errors']}"
    )
```

### 9.5 Étape 4 : Service de projection IA (vers Gold/DM)

**Projection vues IA-ready :**

```python
def project_ai_views(email_uid):
    """Projette les vues IA-ready depuis le DWH."""
    # Récupération depuis DWH
    email = db.query("""
        SELECT * FROM dwh.fact_emails WHERE email_uid = %s
    """, [email_uid])[0]
    
    # A. Faits métier (pré-calculés)
    faits_metier = {
        "email_uid": email_uid,
        "est_interne": email["est_interne"],
        "contient_pieces_jointes": email["contient_pieces_jointes"],
        "nb_pieces_jointes": email["nb_pieces_jointes"],
        "contient_urls": email["contient_urls"],
        "nb_urls": email["nb_urls"],
        "longueur_sujet": email["longueur_sujet"],
        "longueur_corps": email["longueur_corps"],
        "langue_detectee": email["langue_detectee"],
        "urgent": email["urgent"]
    }
    
    # B. Contexte structuré
    contexte = {
        "expediteur": email["from_email"],
        "domaine_expediteur": email["domaine_expediteur"],
        "destinataires": email["to_email"].split(","),
        "sujet": email["subject"],
        "timestamp": email["timestamp"].isoformat()
    }
    
    # C. Texte canonique pour modèles NLP
    text_for_model = f"Objet: {email['subject']}\n\n{email['body_cleaned']}"
    
    # Stockage dans DM
    db.execute("""
        INSERT INTO dm.emails_ai_ready (
            email_uid, faits_metier_json, contexte_json, text_for_model
        ) VALUES (%s, %s, %s, %s)
    """, [email_uid, json.dumps(faits_metier), json.dumps(contexte), text_for_model])
    
    return {
        "faits_metier": faits_metier,
        "contexte": contexte,
        "text_for_model": text_for_model
    }
```

### 9.6 Étape 5 : Classification via orchestrateur IA

**Orchestration complète :**

```python
def classify_email(email_uid):
    """Classifie un email via l'orchestrateur IA."""
    # 1. Chargement contexte
    ai_views = db.query("""
        SELECT * FROM dm.emails_ai_ready WHERE email_uid = %s
    """, [email_uid])[0]
    
    faits_metier = json.loads(ai_views["faits_metier_json"])
    contexte = json.loads(ai_views["contexte_json"])
    text_for_model = ai_views["text_for_model"]
    
    # 2. Intent analysis (ML light)
    intent = analyze_intent_ml(text_for_model)
    
    # 3. Décision stratégie
    if faits_metier["urgent"]:
        priority = "HIGH"
        use_llm = True  # Explication détaillée requise
    else:
        priority = "NORMAL"
        use_llm = False  # ML classique suffit
    
    # 4. Classification
    if use_llm:
        # LLM pour raisonnement + explication
        response = classify_with_llm(text_for_model, faits_metier, contexte)
    else:
        # ML classique pour rapidité
        response = classify_with_ml(text_for_model, faits_metier)
    
    # 5. Validation
    validated = validate_classification(response, faits_metier)
    
    # 6. Logging
    log_classification(email_uid, response, validated)
    
    return validated

def classify_with_ml(text, faits_metier):
    """Classification rapide via ML classique."""
    # Chargement modèle
    model = mlflow.sklearn.load_model("models:/email_classifier/production")
    
    # Prédiction
    prediction = model.predict([text])[0]
    proba = model.predict_proba([text])[0]
    
    return {
        "categorie": prediction,
        "confidence": float(max(proba)),
        "justification": "Classification automatique ML"
    }

def classify_with_llm(text, faits_metier, contexte):
    """Classification avec raisonnement via LLM."""
    # Construction prompt
    prompt = f"""
TU ES : Un assistant de classification d'emails.

RÈGLES ABSOLUES :
1. Tu classes l'email dans UNE des catégories suivantes : resiliation, facturation, technique, commercial, autre
2. Tu justifies en 1 phrase maximum
3. Tu ne réponds QUE au format JSON

FAITS MÉTIER (VÉRITÉS ABSOLUES) :
{json.dumps(faits_metier, indent=2, ensure_ascii=False)}

CONTEXTE EMAIL :
Expéditeur : {contexte['expediteur']}
Domaine : {contexte['domaine_expediteur']}
Sujet : {contexte['sujet']}

TEXTE EMAIL :
{text}

FORMAT DE RÉPONSE OBLIGATOIRE :
{{
  "categorie": "resiliation|facturation|technique|commercial|autre",
  "confidence": 0.0-1.0,
  "justification": "raison en 1 phrase"
}}
"""
    
    # Appel LLM
    response = llm_api.call(
        model_id="mistral-7b-v0.2",
        prompt=prompt,
        temperature=0.1,
        max_tokens=150
    )
    
    # Parsing JSON
    try:
        result = json.loads(response)
        return result
    except json.JSONDecodeError:
        return {
            "categorie": "autre",
            "confidence": 0.0,
            "justification": "Erreur parsing réponse LLM"
        }

def validate_classification(response, faits_metier):
    """Valide la classification."""
    is_valid = True
    errors = []
    
    # Validation format
    required_fields = ["categorie", "confidence", "justification"]
    for field in required_fields:
        if field not in response:
            is_valid = False
            errors.append(f"Champ manquant : {field}")
    
    # Validation catégorie
    valid_categories = ["resiliation", "facturation", "technique", "commercial", "autre"]
    if response.get("categorie") not in valid_categories:
        is_valid = False
        errors.append(f"Catégorie invalide : {response.get('categorie')}")
    
    # Validation confidence
    if not (0 <= response.get("confidence", -1) <= 1):
        is_valid = False
        errors.append("Confidence hors bornes [0, 1]")
    
    return {
        "is_valid": is_valid,
        "errors": errors,
        "categorie": response.get("categorie"),
        "confidence": response.get("confidence"),
        "justification": response.get("justification")
    }
```

### 9.7 Étape 6 : Monitoring et amélioration continue

**Métriques de classification :**

```python
def compute_classification_metrics(period_days=7):
    """Calcule les métriques de classification."""
    # Récupération des logs
    logs = db.query("""
        SELECT 
            categorie, 
            confidence, 
            is_valid, 
            feedback,
            latency_ms
        FROM ai_classification_log
        WHERE timestamp > NOW() - INTERVAL %s DAY
    """, [period_days])
    
    metrics = {
        "total_classifications": len(logs),
        "taux_succes": sum(1 for l in logs if l["is_valid"]) / len(logs),
        "confidence_moyenne": sum(l["confidence"] for l in logs) / len(logs),
        "latence_p95": np.percentile([l["latency_ms"] for l in logs], 95),
        "feedback_positif": sum(1 for l in logs if l["feedback"] == "thumbs_up") / len(logs)
    }
    
    # Répartition par catégorie
    from collections import Counter
    metrics["repartition_categories"] = dict(
        Counter(l["categorie"] for l in logs)
    )
    
    return metrics

# Alerting
def check_metrics_and_alert(metrics):
    """Vérifie les métriques et alerte si nécessaire."""
    if metrics["taux_succes"] < 0.99:
        send_alert(
            subject="Alerte : Taux de succès < 99%",
            body=f"Taux actuel : {metrics['taux_succes']:.2%}"
        )
    
    if metrics["latence_p95"] > 2000:
        send_alert(
            subject="Alerte : Latence P95 > 2s",
            body=f"Latence actuelle : {metrics['latence_p95']}ms"
        )
    
    if metrics["feedback_positif"] < 0.8:
        send_alert(
            subject="Alerte : Feedback positif < 80%",
            body=f"Taux actuel : {metrics['feedback_positif']:.2%}"
        )
```

---

## 10. Mise en production et MLOps

### 10.1 Pipeline CI/CD pour IA

**Principe :**

> Toute modification de modèle/prompt/règle/retriever déclenche une batterie de tests. Promotion vers production uniquement si les seuils sont atteints.

**Pipeline CI/CD (exemple GitLab CI) :**

```yaml
stages:
  - test
  - build
  - deploy

variables:
  MODEL_REGISTRY: "mlflow.company.com"
  DQ_THRESHOLD: "0.95"

test_model:
  stage: test
  script:
    # Tests unitaires
    - pytest tests/unit/
    
    # Tests de non-régression IA
    - python scripts/test_ai_non_regression.py
    
    # Tests de robustesse
    - python scripts/test_ai_robustness.py
    
    # Évaluation métriques
    - python scripts/evaluate_model.py --threshold $DQ_THRESHOLD
  
  artifacts:
    reports:
      junit: test-results.xml

build_model:
  stage: build
  script:
    # Enregistrement dans registry
    - mlflow models register --model-uri "runs:/${CI_COMMIT_SHA}/model" --name "email_classifier"
  
  only:
    - main

deploy_canary:
  stage: deploy
  script:
    # Déploiement canary (10%)
    - kubectl set image deployment/email-classifier classifier=email-classifier:${CI_COMMIT_SHA}
    - kubectl patch deployment email-classifier -p '{"spec":{"replicas":1}}'
    
    # Monitoring pendant 30 minutes
    - python scripts/monitor_canary.py --duration 1800
    
    # Si OK, rollout complet
    - kubectl scale deployment email-classifier --replicas=10
  
  only:
    - main
  
  environment:
    name: production
    on_stop: rollback_deployment

rollback_deployment:
  stage: deploy
  script:
    - kubectl rollout undo deployment/email-classifier
  when: manual
```

### 10.2 Déploiement progressif (canary)

**Stratégie canary :**

1. **Déploiement initial** (10% trafic)
2. **Monitoring** (métriques, logs, erreurs)
3. **Validation** (si métriques OK)
4. **Rollout complet** (100% trafic)
5. **Ou rollback** (si problème)

```python
def canary_deployment(new_model_id, canary_percentage=10, duration_minutes=30):
    """Déploie un nouveau modèle en canary."""
    # 1. Déploiement canary
    deploy_model(new_model_id, traffic_percentage=canary_percentage)
    
    # 2. Monitoring
    start_time = time.time()
    while time.time() - start_time < duration_minutes * 60:
        # Récupération métriques
        metrics_new = get_metrics(new_model_id)
        metrics_old = get_metrics(current_model_id)
        
        # Comparaison
        if metrics_new["error_rate"] > metrics_old["error_rate"] * 1.2:
            # Erreurs +20% → rollback
            rollback_model(new_model_id)
            return {"status": "ROLLED_BACK", "reason": "error_rate"}
        
        if metrics_new["latency_p95"] > metrics_old["latency_p95"] * 1.5:
            # Latence +50% → rollback
            rollback_model(new_model_id)
            return {"status": "ROLLED_BACK", "reason": "latency"}
        
        time.sleep(60)  # Check toutes les minutes
    
    # 3. Si OK, rollout complet
    deploy_model(new_model_id, traffic_percentage=100)
    
    return {"status": "SUCCESS"}
```

### 10.3 Monitoring production

**Dashboards Grafana (exemple) :**

**Dashboard 1 : Performance globale**

- Latence P50/P95/P99 (par étape : RAG, LLM, total)
- Taux de succès (%)
- Taux d'erreur (%)
- Coût par heure ($)

**Dashboard 2 : Qualité IA**

- Confidence moyenne par catégorie
- Taux de feedback positif (%)
- Taux de quarantaine (%)
- Distribution des catégories

**Dashboard 3 : Infrastructure**

- CPU/Mémoire par service
- Taux de hit cache (%)
- Nombre de requêtes par seconde
- File d'attente (longueur)

**Alerting (exemple Prometheus Alertmanager) :**

```yaml
groups:
  - name: ai_classification_alerts
    interval: 30s
    rules:
      - alert: HighErrorRate
        expr: rate(ai_classification_errors_total[5m]) > 0.05
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "Taux d'erreur élevé (>5%)"
          description: "Taux actuel : {{ $value }}"
      
      - alert: HighLatency
        expr: histogram_quantile(0.95, rate(ai_classification_duration_seconds_bucket[5m])) > 2
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "Latence P95 >2s"
          description: "Latence actuelle : {{ $value }}s"
      
      - alert: LowConfidence
        expr: avg(ai_classification_confidence) < 0.7
        for: 15m
        labels:
          severity: warning
        annotations:
          summary: "Confidence moyenne <0.7"
          description: "Confidence actuelle : {{ $value }}"
```

### 10.4 Gestion des incidents

**Runbook exemple (incident LLM) :**

**Symptôme :** Taux d'erreur LLM >10%

**Actions :**

1. **Vérifier status API externe** (si LLM cloud)
   - Consulter status page (ex: status.openai.com)
   - Vérifier rate limits

2. **Vérifier logs** (erreurs, timeouts)
   ```bash
   kubectl logs -l app=orchestrator --since=10m | grep ERROR

   Activer fallback (modèle secondaire)

    update_config("llm.fallback_enabled", True)

    Escalade (si problème persiste >15min)
        Notifier équipe SRE
        Activer mode dégradé (ML classique uniquement)
    Post-mortem (après résolution)
        Documenter incident
        Identifier cause racine
        Mettre à jour runbook

Conclusion : Principes d'or de l'architecture IA
Les 10 commandements de l'architecture IA industrielle

    Un module = une responsabilité claire
        Découper par responsabilités métier stables, pas par type de données
    Contrats JSON stables entre modules
        Versionner, documenter, tester
    Tout ce qui est calculable → hors LLM
        ETL fait les calculs, LLM raisonne
    Data quality : publish gate obligatoire
        Aucune donnée invalide ne doit atteindre le DWH/DM
    Traçabilité complète (lineage run-level)
        Pouvoir rejouer n'importe quelle exécution
    RAG = réduction, pas enrichissement
        Filtrer avant d'injecter, pas tout passer au LLM
    Validation post-LLM obligatoire
        Vérifier cohérence avec faits métier
    Tests IA = tests unitaires + non-régression
        Jeux de tests versionnés, exécution automatique
    Monitoring production = observabilité + qualité
        Métriques techniques + métriques métier
    Amélioration continue via feedback humain
        Boucle fermée : feedback → tests → ajustement

Ressources complémentaires
Articles et papiers scientifiques

    Data quality & quarantine patterns
        DQX framework documentation (https://databrickslabs.github.io/dqx/docs/motivation/)
        "Data Quality & Validation Checks in Azure Data platform" (https://piyash.hashnode.dev/data-quality-validation-checks-in-azure-data-platform)
    Architecture médaillon (Bronze/Silver/Gold)
        Microsoft, "What is the medallion lakehouse architecture?" (https://learn.microsoft.com/en-us/azure/databricks/lakehouse/medallion)
        Microsoft, "Data lake zones and containers" (https://learn.microsoft.com/en-us/azure/cloud-adoption-framework/scenarios/cloud-scale-analytics/best-practices/data-lake-zones)
    Star schema et modélisation dimensionnelle
        Microsoft Power BI, "Understand star schema" (https://learn.microsoft.com/en-us/power-bi/guidance/star-schema)
    Data contracts
        Open Data Contract Standard (ODCS) (https://github.com/bitol-io/open-data-contract-standard)
        "Data contracts: What are they and why do they matter?" (Thoughtworks)
    RAG et retrieval
        RAGAS framework pour évaluation RAG (https://github.com/explodinggradients/ragas)
        "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks" (Lewis et al., 2020)
    MLOps et déploiement
        "Hidden Technical Debt in Machine Learning Systems" (Sculley et al., 2015)
        "Continuous Delivery for Machine Learning" (Thoughtworks)

Outils et frameworks

    Data platform : Databricks, Snowflake, BigQuery, Azure Synapse
    ETL/Orchestration : Airflow, Prefect, Dagster
    Vector stores : Qdrant, Pinecone, Milvus, pgvector
    Model registry : MLflow, Weights & Biases, Vertex AI
    Monitoring : Prometheus, Grafana, Datadog, Arize AI
    LLM serving : vLLM, TGI, Ollama

Questions de vérification de compréhension

    Quelle est la différence entre la couche Bronze, Silver et Gold ?
    Pourquoi un "publish gate" est-il indispensable entre Bronze et Silver ?
    Quand utiliser le RAG vs les règles métier déterministes vs le fine-tuning ?
    Quel est le rôle exact du service de projection IA, et quand est-il exécuté ?
    Pourquoi l'orchestrateur ne doit-il JAMAIS calculer de faits métier ?
    Comment garantir la traçabilité (lineage) d'une transformation DL → DW → DM ?
    Quelles sont les 3 familles de déclencheurs de quarantaine ?
    Quelle est la différence entre data drift et concept drift en ML ?
    Pourquoi le LLM ne doit-il jamais être la seule source de décision pour un système critique ?
    Comment organiser les tests automatisés pour un système IA en production ?

Note finale : Ce cours est conçu pour être relu dans plusieurs années. Tous les concepts sont basés sur des références publiques ou des échanges documentés. Les principes architecturaux sont agnostiques des technologies et resteront valides même si les outils évoluent.
