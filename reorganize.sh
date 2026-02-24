#!/bin/bash

# Script de réorganisation de l'arborescence du 2nd-cerveau-travail
# Date: 2026-02-24
# Auteur: Vito Corleone

set -e  # Arrêt en cas d'erreur

echo "🚀 Début de la réorganisation de l'arborescence..."

# Créer une nouvelle branche
git checkout -b reorganisation-arborescence-2026-02-24

# Créer la nouvelle structure de dossiers
echo "📁 Création de la nouvelle structure..."

mkdir -p 01_machine_learning/01_fundamentals
mkdir -p 01_machine_learning/02_supervised
mkdir -p 01_machine_learning/03_unsupervised/dimension_reduction

mkdir -p 02_deep_learning/01_architectures
mkdir -p 02_deep_learning/02_fundamentals
mkdir -p 02_deep_learning/03_llm
mkdir -p 02_deep_learning/04_interpretabilite

mkdir -p 03_infrastructure

mkdir -p 04_databases/sql

mkdir -p 05_methodologie

# Déplacer et renommer les fichiers
echo "🔄 Déplacement des fichiers..."

# Machine Learning - Fundamentals
git mv General/PreprocessingData.md 01_machine_learning/01_fundamentals/preprocessing_data.md
git mv General/PostprocessingData.md 01_machine_learning/01_fundamentals/postprocessing_data.md

# Machine Learning - Supervised
git mv General/SVM.md 01_machine_learning/02_supervised/svm.md
git mv General/XGBoost.md 01_machine_learning/02_supervised/xgboost.md
git mv General/LightGBM.md 01_machine_learning/02_supervised/lightgbm.md

# Machine Learning - Unsupervised
git mv General/Reduction-dim/T-SNE.md 01_machine_learning/03_unsupervised/dimension_reduction/tsne.md

# Deep Learning - Architectures
git mv General/Réseaux-Neuronnaux/CNN.md 02_deep_learning/01_architectures/cnn.md
git mv General/Réseaux-Neuronnaux/ViT.md 02_deep_learning/01_architectures/vit.md
git mv General/Réseaux-Neuronnaux/Yolo.md 02_deep_learning/01_architectures/yolo.md

# Deep Learning - Fundamentals
git mv General/Réseaux-Neuronnaux/FonctionActivation.md 02_deep_learning/02_fundamentals/fonction_activation.md

# Deep Learning - LLM
git mv General/Réseaux-Neuronnaux/ApprentissageRenforcementLLM.md 02_deep_learning/03_llm/apprentissage_renforcement_llm.md
git mv General/Réseaux-Neuronnaux/DistillationLLM.md 02_deep_learning/03_llm/distillation_llm.md
git mv General/RAG.md 02_deep_learning/03_llm/rag.md

# Deep Learning - Interpretabilité
git mv General/Réseaux-Neuronnaux/Interpretabilité/Intagrated-Gradient.md 02_deep_learning/04_interpretabilite/integrated_gradients.md

# Infrastructure
git mv General/Infrastructure/Infra-IA.md 03_infrastructure/infra_ia.md

# Databases
git mv SQL/creation_tables.md 04_databases/sql/creation_tables.md

# Méthodologie
git mv General/Presenter-resultats-IA.md 05_methodologie/presenter_resultats_ia.md
git mv General/Presentation-comprhension/Comprendre-papier-scientifique.md 05_methodologie/comprendre_papier_scientifique.md

# Supprimer les anciens dossiers vides
echo "🗑️  Nettoyage des dossiers vides..."
rm -rf General
rm -rf SQL

# Commit des changements
echo "💾 Commit des modifications..."
git add .
git commit -m "♻️ Réorganisation de l'arborescence du repository

- Structure thématique claire: ML → DL → Infrastructure → Databases → Méthodologie
- Nomenclature normalisée en snake_case
- Hiérarchie logique avec numérotation pour la progression pédagogique
- Correction de typos (Intagrated → Integrated, Presentation-comprhension)
- Organisation alignée sur les standards de documentation technique

Détails:
- 01_machine_learning/: Fondamentaux, Supervisé, Non-supervisé
- 02_deep_learning/: Architectures, Fundamentals, LLM, Interprétabilité
- 03_infrastructure/: Déploiement et infrastructure IA
- 04_databases/: SQL et gestion de données
- 05_methodologie/: Compétences transverses (présentation, lecture papiers)
"

echo "✅ Réorganisation terminée!"
echo ""
echo "📌 Prochaines étapes:"
echo "1. Vérifiez les changements: git status"
echo "2. Poussez la branche: git push origin reorganisation-arborescence-2026-02-24"
echo "3. Créez une Pull Request sur GitHub"
echo ""
echo "🔗 URL pour créer la PR:"
echo "https://github.com/ClemWasChoosen/2nd-cerveau-travail/pull/new/reorganisation-arborescence-2026-02-24"
