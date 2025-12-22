# Guide Complet : Présentation des Résultats de Modèles d'IA
**Pense-bête professionnel pour des présentations claires et pertinentes**

---

## 📊 TABLEAU RÉCAPITULATIF GÉNÉRAL

### Vue d'ensemble par type de tâche

| **Type de Tâche** | **Métriques Principales** | **Visualisations Clés** | **Audience Direction** | **Audience Technique** |
|-------------------|---------------------------|-------------------------|------------------------|------------------------|
| **Classification Binaire** | Accuracy, Precision, Recall, F1-Score, AUC-ROC | Matrice de confusion, Courbe ROC, Courbe Precision-Recall | Taux de bonnes prédictions, Coût des erreurs | Seuil de décision, Trade-offs |
| **Classification Multi-classes** | Accuracy, F1-Score (macro/micro/weighted), Confusion Matrix | Matrice de confusion, F1 par classe, Courbe d'apprentissage | Performance globale et par catégorie | Déséquilibre de classes |
| **Régression** | MAE, RMSE, R², MAPE | Prédictions vs Réel, Résidus, Distribution des erreurs | Précision des prédictions en unités métier | Biais et variance |
| **Détection d'Objets (YOLO)** | mAP, Precision, Recall, IoU | Détections visuelles, Courbe Precision-Recall, Confusion Matrix | Taux de détection correcte, Exemples visuels | Performance par classe, seuils |
| **NLP - Classification de texte** | Accuracy, F1-Score, Confusion Matrix | Matrice de confusion, Exemples d'erreurs | Compréhension correcte des textes | Performance par catégorie |
| **NLP - Génération (LLM)** | Perplexité, BLEU, ROUGE, BERTScore | Exemples de génération, Comparaisons humaines | Qualité des textes générés | Métriques automatiques |
| **NLP - NER/Extraction** | F1-Score (entité), Precision, Recall | Exemples annotés, Confusion par type d'entité | Taux d'extraction correcte | Performance par entité |
| **Transversal** | Temps d'entraînement, Temps d'inférence, Taille du modèle | Courbes d'apprentissage (loss), Temps d'exécution | Coût et temps de traitement | Convergence, overfitting |

### Métriques détaillées par usage

| **Métrique** | **Formule** | **Quand l'utiliser** | **Valeur idéale** | **Attention** |
|--------------|-------------|----------------------|-------------------|---------------|
| **Accuracy** | $\frac{TP + TN}{TP + TN + FP + FN}$ | Classes équilibrées | 1.0 (100%) | Trompeuse si déséquilibre |
| **Precision** | $\frac{TP}{TP + FP}$ | Coût élevé des faux positifs | 1.0 (100%) | Peut être élevée avec beaucoup de FN |
| **Recall (Sensibilité)** | $\frac{TP}{TP + FN}$ | Coût élevé des faux négatifs | 1.0 (100%) | Peut être élevé avec beaucoup de FP |
| **F1-Score** | $2 \times \frac{Precision \times Recall}{Precision + Recall}$ | Équilibre Precision/Recall | 1.0 (100%) | Moyenne harmonique |
| **AUC-ROC** | Aire sous courbe ROC | Évaluer tous les seuils possibles | 1.0 (100%) | Moins pertinent si déséquilibre fort |
| **MAE** | $\frac{1}{n} \sum_{i=1}^{n} \|y_i - \hat{y}_i\|$ | Régression, erreurs en unités réelles | 0 | Sensible aux outliers (moins que RMSE) |
| **RMSE** | $\sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}$ | Régression, pénaliser les grandes erreurs | 0 | Très sensible aux outliers |
| **R²** | $1 - \frac{SS_{res}}{SS_{tot}}$ | Régression, qualité d'ajustement | 1.0 (100%) | Peut être négatif si très mauvais modèle |
| **mAP** | Moyenne des AP par classe | Détection d'objets | 1.0 (100%) | Dépend du seuil IoU choisi |
| **IoU** | $\frac{Aire(A \cap B)}{Aire(A \cup B)}$ | Détection/Segmentation | 1.0 (100%) | Seuil typique : 0.5 |

---

## 1. PRINCIPES GÉNÉRAUX DE PRÉSENTATION

### 1.1 Adapter le discours à l'audience

**Pourquoi c'est crucial ?**
Présenter des résultats d'IA à la direction n'est pas la même chose qu'en parler à un expert technique. Le même slide peut être incompréhensible pour l'un et trop simpliste pour l'autre.

**Les trois types d'audiences principales :**

#### 🎯 **Direction / Management (Non-technique)**
- **Ce qui compte :** Impact métier, ROI, fiabilité, coûts, risques
- **Langage :** Éviter le jargon, utiliser des analogies métier
- **Métriques prioritaires :** Accuracy (taux de réussite), exemples concrets, coût des erreurs
- **Visualisations :** Simples, colorées, annotées, avec peu de chiffres

**Exemple de formulation :**  
❌ "Le modèle atteint 0.87 d'AUC-ROC"  
✅ "Le modèle identifie correctement 87% des cas problématiques, ce qui réduit les contrôles manuels de 40%"

#### 🔬 **Experts IA / Data Scientists**
- **Ce qui compte :** Choix techniques, architecture, hyperparamètres, reproductibilité
- **Langage :** Jargon technique acceptable, précision mathématique
- **Métriques prioritaires :** F1, AUC-ROC, Precision-Recall, loss curves, convergence
- **Visualisations :** Détaillées, avec plusieurs métriques simultanées

**Exemple de formulation :**
✅ "F1-Score de 0.87 avec precision 0.91 et recall 0.83, optimisé via grid search sur le seuil de décision"

#### 👥 **Utilisateurs Finaux**
- **Ce qui compte :** Facilité d'utilisation, confiance, cas d'usage concrets
- **Langage :** Simple, rassurant, focus sur les bénéfices
- **Métriques prioritaires :** Taux de réussite en termes métier
- **Visualisations :** Exemples concrets, démonstrations visuelles

**Exemple de formulation :**
✅ "Sur 100 documents que vous soumettez, le système en traite correctement 92, et signale les 8 restants pour vérification humaine"

### 1.2 Structure d'une présentation efficace

**Le storytelling en 4 temps :**

1. **Contexte & Problème** (1-2 slides)
   - Quel problème métier résolvons-nous ?
   - Pourquoi l'IA est-elle pertinente ?

2. **Approche & Données** (1-2 slides)
   - Type de modèle choisi (et pourquoi ce choix)
   - Données utilisées (quantité, qualité, représentativité)

3. **Résultats** (3-5 slides) ← **Cœur de ce guide**
   - Métriques de performance
   - Visualisations clés
   - Comparaison avec baseline ou état de l'art

4. **Impact & Prochaines étapes** (1-2 slides)
   - Bénéfices métier quantifiés
   - Limites actuelles
   - Recommandations

---

## 2. CLASSIFICATION : MÉTRIQUES & VISUALISATIONS

### 2.1 Comprendre les bases : TP, TN, FP, FN

**Matrice de confusion - Le fondement de tout**

```
                      Prédiction
                  Positif    Négatif
Réalité  Positif   TP         FN
         Négatif   FP         TN
```

**Définitions :**
- **TP (True Positive)** : Cas positif correctement identifié
- **TN (True Negative)** : Cas négatif correctement identifié
- **FP (False Positive)** : Cas négatif incorrectement identifié comme positif (Erreur de Type I)
- **FN (False Negative)** : Cas positif incorrectement identifié comme négatif (Erreur de Type II)

**Pourquoi c'est important ?**
Toutes les métriques de classification dérivent de ces 4 valeurs. Comprendre ces concepts permet de choisir la bonne métrique selon le contexte métier.

### 2.2 Accuracy (Exactitude)

**Formule mathématique :**
$$ Accuracy = \frac{TP + TN}{TP + TN + FP + FN} = \frac{\text{Prédictions correctes}}{\text{Total des prédictions}} $$

**Quand l'utiliser ?**
- Classes équilibrées (environ 50/50 ou proche)
- Coût des FP et FN équivalent
- Communication grand public (métrique intuitive)

**Pourquoi cette métrique peut être trompeuse ?**

**Exemple concret :**
Détection de fraudes bancaires avec 1% de fraudes réelles.
Un modèle qui prédit "pas de fraude" pour TOUS les cas obtient 99% d'accuracy !

```python
# Exemple de calcul
TP = 5      # Fraudes détectées
TN = 9900   # Non-fraudes correctement identifiées
FP = 50     # Fausses alertes
FN = 45     # Fraudes manquées

accuracy = (TP + TN) / (TP + TN + FP + FN)
print(f"Accuracy: {accuracy:.2%}")  # 99.05%
# Mais on rate 90% des fraudes ! (FN=45 sur 50 fraudes réelles)
```

**Visualisation PowerPoint recommandée :**
- Jauge ou barre de progression (0-100%)
- Comparaison avec un modèle baseline
- Annotation du contexte (équilibré/déséquilibré)

**Comment présenter à la direction :**
"Le modèle prédit correctement 95% des cas, ce qui signifie qu'il se trompe sur 5% des décisions. Sur 1000 transactions, environ 50 nécessiteraient une vérification manuelle."

### 2.3 Precision (Précision)

**Formule mathématique :**
$$ Precision = \frac{TP}{TP + FP} = \frac{\text{Vrais positifs}}{\text{Tous les positifs prédits}} $$

**Interprétation :** "Parmi tous les cas que le modèle a identifiés comme positifs, quelle proportion l'est vraiment ?"

**Quand l'utiliser ?**
- **CoCoût élevé des faux positifs (FP)**
- Exemples métier :
  - Spam : Un vrai email classé comme spam (FP) = client mécontent
  - Recommandations produits : Mauvaise recommandation (FP) = perte de confiance
  - Alertes médicales : Fausse alerte (FP) = tests inutiles coûteux

**Pourquoi optimiser la Precision ?**
Dans certains contextes, il vaut mieux être sûr de ce qu'on détecte, quitte à en rater quelques-uns.

**Exemple concret :**
Système de recommandation de produits premium :
- TP = 80 recommandations pertinentes cliquées
- FP = 20 recommandations non pertinentes (irritent le client)
- Precision = 80/(80+20) = 80%

**Visualisation PowerPoint recommandée :**
- Graphique en barres comparant Precision/Recall
- Diagramme illustrant "Sur 100 alertes, combien sont vraies ?"
- Évolution de la Precision selon différents seuils

**Comment présenter à la direction :**
"Quand le système déclenche une alerte, elle est correcte 8 fois sur 10. Cela limite les interventions inutiles et optimise le temps des équipes."

### 2.4 Recall / Sensibilité (Rappel)

**Formule mathématique :**
$$ Recall = \frac{TP}{TP + FN} = \frac{\text{Vrais positifs}}{\text{Tous les positifs réels}} $$

**Interprétation :** "Parmi tous les cas réellement positifs, quelle proportion le modèle a-t-il détectée ?"

**Quand l'utiliser ?**
- **Coût élevé des faux négatifs (FN)**
- Exemples métier :
  - Détection de maladies graves : Rater un cancer (FN) = risque vital
  - Détection de fraudes : Rater une fraude (FN) = perte financière
  - Détection d'intrusions : Rater une cyberattaque (FN) = compromission du système

**Pourquoi optimiser le Recall ?**
Dans certains contextes, il est crucial de ne rien rater, même si cela génère des fausses alertes.

**Exemple concret :**
Détection de défauts critiques en production :
- TP = 90 défauts détectés
- FN = 10 défauts ratés (risque de livraison défectueuse)
- Recall = 90/(90+10) = 90%

**Trade-off Precision vs Recall :**
C'est souvent un compromis :
- ↑ Precision → ↓ Recall (être plus sélectif)
- ↑ Recall → ↓ Precision (détecter plus largement)

**Visualisation PowerPoint recommandée :**
- Graphique montrant le trade-off Precision-Recall
- Courbe Precision-Recall
- Illustration : "Sur 100 cas problématiques réels, combien sont détectés ?"

**Comment présenter à la direction :**
"Le système détecte 90% des cas problématiques. Sur 100 incidents réels, 10 passeraient inaperçus. C'est un choix conscient pour limiter les fausses alertes à 15%."

### 2.5 F1-Score

**Formule mathématique :**
$$ F1 = 2 \times \frac{Precision \times Recall}{Precision + Recall} = \frac{2 \times TP}{2 \times TP + FP + FN} $$

**Interprétation :** Moyenne harmonique entre Precision et Recall (équilibre entre les deux).

**Pourquoi la moyenne harmonique et pas arithmétique ?**
La moyenne harmonique pénalise les déséquilibres extrêmes.

Exemple :
- Precision = 90%, Recall = 10%
- Moyenne arithmétique = 50% (semble correct)
- F1-Score = 18% (reflète mieux le déséquilibre)

**Quand l'utiliser ?**
- Quand Precision et Recall sont tous deux importants
- Classes déséquilibrées
- Comparaison globale de modèles
- Standard dans la communauté ML

**Variantes du F1-Score :**

1. **F1 Macro** : Moyenne des F1 de chaque classe (toutes les classes ont le même poids)
   $$ F1_{macro} = \frac{1}{n} \sum_{i=1}^{n} F1_i $$

2. **F1 Micro** : F1 global calculé sur l'ensemble des TP, FP, FN
   $$ F1_{micro} = \frac{2 \times TP_{total}}{2 \times TP_{total} + FP_{total} + FN_{total}} $$

3. **F1 Weighted** : Moyenne pondérée par le nombre d'échantillons par classe
   $$ F1_{weighted} = \sum_{i=1}^{n} \frac{n_i}{N} F1_i $$

**Quand utiliser quelle variante ?**
- **Macro** : Toutes les classes ont la même importance (même les petites classes)
- **Micro** : Performance globale (biaisé vers les classes majoritaires)
- **Weighted** : Compromis, reflète la distribution réelle

**Visualisation PowerPoint recommandée :**
- Affichage du F1-Score avec Precision et Recall en contexte
- Graphique radar pour F1 par classe (multi-classes)
- Comparaison F1 de plusieurs modèles

**Comment présenter à la direction :**
"Le F1-Score de 85% indique un bon équilibre entre détecter les vrais cas (90% de détection) et éviter les fausses alertes (80% de précision)."

### 2.6 Matrice de Confusion

**Définition :**
Tableau croisé montrant les prédictions vs la réalité.

**Exemple Classification Binaire :**

```
                Prédit Négatif   Prédit Positif
Réel Négatif            850              50
Réel Positif            30               70
```


**Exemple Classification Multi-classes :**
```
            Prédit A   Prédit B   Prédit C
Réel A            120        10         5
Réel B            8          95         12
Réel C            3          15         132
```

**Pourquoi c'est la visualisation la plus importante ?**
1. Montre les types d'erreurs (où le modèle se trompe)
2. Identifie les classes confondues entre elles
3. Base de toutes les autres métriques

**Visualisation PowerPoint recommandée :**
- Heatmap avec échelle de couleur (vert = bon, rouge = erreurs)
- Annotations des valeurs principales
- Normalisation possible (en pourcentages par ligne)

**Bonnes pratiques de présentation :**
- **Pour la direction** : Simplifier, montrer uniquement les erreurs critiques
- **Pour les experts** : Version complète avec pourcentages et valeurs brutes
- Ajouter des annotations explicatives sur les erreurs notables

**Comment interpréter :**
- **Diagonale** : Prédictions correctes (plus c'est élevé, mieux c'est)
- **Hors diagonale** : Erreurs (analyser les confusions fréquentes)

**Exemple d'analyse :**
"Le modèle confond souvent les catégories B et C (15 cas), ce qui suggère qu'elles sont similaires. Nous pouvons envisager de collecter plus de données différenciatrices."

### 2.7 Courbe ROC et AUC-ROC

**ROC = Receiver Operating Characteristic**

**Principe :**
La courbe ROC visualise les performances du modèle pour tous les seuils de décision possibles.

**Axes :**
- **Axe X** : Taux de Faux Positifs (FPR) = $$ \frac{FP}{FP + TN} $$
- **Axe Y** : Taux de Vrais Positifs (TPR) = Recall = $$ \frac{TP}{TP + FN} $$

**AUC-ROC (Area Under the Curve) :**
- **AUC = 1.0** : Classificateur parfait
- **AUC = 0.5** : Classificateur aléatoire (ligne diagonale)
- **AUC < 0.5** : Pire qu'aléatoire (inverser les prédictions !)

**Pourquoi c'est utile ?**
1. Évalue la performance indépendamment du seuil choisi
2. Compare des modèles objectivement
3. Montre le trade-off entre TPR et FPR

**Interprétation intuitive :**
AUC = 0.85 signifie : "Si on prend un exemple positif et un négatif au hasard, il y a 85% de chances que le modèle assigne un score plus élevé au positif."

**Limites de la courbe ROC :**
- Moins informative si classes très déséquilibrées
- Dans ce cas, préférer la courbe Precision-Recall

**Visualisation PowerPoint recommandée :**
- Courbe ROC avec ligne de référence (diagonale)
- Affichage de l'AUC dans le titre ou légende
- Comparaison de plusieurs modèles sur le même graphique

**Comment présenter à la direction :**
"L'AUC de 0.87 indique que le modèle fait bien la distinction entre les cas positifs et négatifs. Plus on est proche de 1, meilleure est la performance. Un score aléatoire serait de 0.5."

### 2.8 Courbe Precision-Recall

**Principe :**
Similaire à la courbe ROC, mais affiche Precision vs Recall pour tous les seuils.

**Axes :**
- **Axe X** : Recall
- **Axe Y** : Precision

**Pourquoi préférer cette courbe à ROC dans certains cas ?**

**Classes déséquilibrées :**
Avec ROC, un modèle médiocre peut paraître bon car le TN (classe majoritaire) domine.
La courbe Precision-Recall se concentre uniquement sur la classe positive (minoritaire).

**Exemple :**
Détection de fraudes (1% de fraudes) :
- Courbe ROC : Peut afficher AUC = 0.90 (trompeur)
- Courbe Precision-Recall : Montrera mieux les difficultés réelles

**Interprétation :**
- Plus l'aire sous la courbe est grande, mieux c'est
- Ligne de référence = proportion de la classe positive

**Visualisation PowerPoint recommandée :**
- Courbe avec plusieurs seuils annotés
- Comparaison avec baseline
- Mise en évidence du point de fonctionnement choisi

**Comment présenter à la direction :**
"Cette courbe montre le compromis : plus on veut détecter de cas (Recall élevé), plus on aura de fausses alertes (Precision baisse). Le point rouge indique notre seuil de fonctionnement optimal."

### 2.9 Choix du seuil de décision

**Problème :**
Les modèles de classification produisent des probabilités (ex: 0.73), pas directement des classes. Il faut choisir un seuil (ex: 0.5) pour décider.

**Pourquoi 0.5 n'est pas toujours optimal ?**

**Exemple métier :**
Détection de clients à risque de churn :
- Coût de rétention : 50€ (action marketing)
- Valeur client perdue : 500€
- Ratio coût FP vs FN : 1:10
→ Seuil optimal : ~0.3 (plus sensible, accepte plus de FP)

**Comment choisir le seuil ?**

1. **Approche métier** : Quantifier les coûts des FP et FN
   $$ \text{Seuil optimal} \propto \frac{\text{Coût FP}}{\text{Coût FN}} $$

2. **Approche F1** : Maximiser le F1-Score

3. **Approche ROC** : Point le plus proche du coin supérieur gauche

**Visualisation PowerPoint recommandée :**
- Graphique montrant Precision, Recall et F1 en fonction du seuil
- Annotation du seuil choisi avec justification
- Tableau coût/bénéfice si disponible

**Comment présenter à la direction :**
"Nous avons ajusté le seuil de décision à 0.35 au lieu de 0.5 standard. Cela augmente les détections de 15% en acceptant 10% de fausses alertes supplémentaires, ce qui reste acceptable vu le coût élevé d'un cas manqué."

---

## 3. RÉGRESSION : MÉTRIQUES & VISUALISATIONS

### 3.1 MAE (Mean Absolute Error)

**Formule mathématique :**
$$ MAE = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i| $$

Où :
- $$ y_i $$ = valeur réelle
- $$ \hat{y}_i $$ = valeur prédite
- $$ n $$ = nombre d'observations

**Interprétation :**
Erreur moyenne en valeur absolue, dans les mêmes unités que la variable cible.

**Pourquoi l'utiliser ?**
- Facilement interprétable (même unité que les données)
- Moins sensible aux outliers que RMSE
- Toutes les erreurs ont le même poids

**Exemple concret :**
Prédiction de prix immobiliers :
- MAE = 25 000€
→ "En moyenne, le modèle se trompe de 25 000€ sur le prix prédit"

**Visualisation PowerPoint recommandée :**
- Affichage de la valeur avec unité métier
- Comparaison avec baseline (moyenne, médiane)
- Contexte : "X% du prix moyen"

**Comment présenter à la direction :**
"Le modèle prédit les prix avec une erreur moyenne de 25 000€, soit 8% du prix moyen d'un bien. C'est 30% plus précis que notre méthode actuelle."

### 3.2 RMSE (Root Mean Squared Error)

**Formule mathématique :**
$$ RMSE = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2} $$

**Interprétation :**
Racine carrée de la moyenne des erreurs au carré, dans les mêmes unités que la variable cible.

**Pourquoi l'utiliser ?**
- Pénalise davantage les grandes erreurs (grâce au carré)
- Utile quand les grandes erreurs sont particulièrement coûteuses
- Standard dans la communauté ML

**Différence MAE vs RMSE :**

{{{python
import numpy as np

# Exemple de prédictions
y_true = np.array([100, 100, 100, 100, 100])
y_pred1 = np.array([105, 105, 105, 105, 105])  # Erreurs constantes
y_pred2 = np.array([100, 100, 100, 100, 125])  # Une grosse erreur

mae1 = np.mean(np.abs(y_true - y_pred1))
rmse1 = np.sqrt(np.mean((y_true - y_pred1)**2))
print(f"Cas 1 - MAE: {mae1:.2f}, RMSE: {rmse1:.2f}")  # MAE: 5.00, RMSE: 5.00

mae2 = np.mean(np.abs(y_true - y_pred2))
rmse2 = np.sqrt(np.mean((y_true - y_pred2)**2))
print(f"Cas 2 - MAE: {mae2:.2f}, RMSE: {rmse2:.2f}")  # MAE: 5.00, RMSE: 11.18

# Même MAE mais RMSE plus élevé car une grosse erreur
}}}

**Quand préférer RMSE à MAE ?**
- Quand les grandes erreurs sont inacceptables
- Quand on veut être conservateur

**Quand préférer MAE à RMSE ?**
- Quand on veut une métrique robuste aux outliers
- Quand toutes les erreurs ont le même coût

**Visualisation PowerPoint recommandée :**
- Comparaison MAE vs RMSE (si RMSE >> MAE, présence d'outliers)
- Graphique avec unités métier

**Comment présenter à la direction :**
"Le RMSE de 35 000€ est supérieur au MAE de 25 000€, ce qui signifie que quelques prédictions ont des erreurs importantes. Nous travaillons à améliorer ces cas extrêmes."

### 3.3 R² (Coefficient de détermination)

**Formule mathématique :**
$$ R^2 = 1 - \frac{SS_{res}}{SS_{tot}} = 1 - \frac{\sum_{i=1}^{n} (y_i - \hat{y}_i)^2}{\sum_{i=1}^{n} (y_i - \bar{y})^2} $$

Où :
- $$ SS_{res} $$ = Somme des carrés des résidus (erreur du modèle)
- $$ SS_{tot} $$ = Somme des carrés totale (variance des données)
- $$ \bar{y} $$ = Moyenne des valeurs réelles

**Interprétation :**
Proportion de la variance expliquée par le modèle.

**Valeurs typiques :**
- **R² = 1** : Modèle parfait (explique 100% de la variance)
- **R² = 0** : Modèle aussi bon qu'une simple moyenne
- **R² < 0** : Modèle pire qu'une moyenne (rare, signe de gros problème)

**Pourquoi c'est utile ?**
- Sans unité (comparaison entre datasets différents)
- Intuitivement compréhensible (pourcentage de variance expliquée)

**Attention aux pièges :**
1. R² augmente toujours avec plus de features (même non pertinentes)
   → Utiliser R² ajusté pour pénaliser la complexité :
   $$ R^2_{adj} = 1 - (1 - R^2) \frac{n - 1}{n - p - 1} $$
   Où p = nombre de features

2. R² ne mesure pas la qualité de l'ajustement absolu, seulement relatif

**Exemple concret :**
Prédiction de consommation énergétique :
- R² = 0.85
→ "Le modèle explique 85% des variations de consommation. Les 15% restants sont dus à des facteurs non capturés ou à du bruit."

**Visualisation PowerPoint recommandée :**
- Jauge ou barre (0-100%)
- Graphique scatter plot avec droite de régression
- Comparaison avec modèles alternatifs

**Comment présenter à la direction :**
"Le R² de 85% signifie que le modèle capture les principaux facteurs explicatifs. C'est un bon niveau de performance pour ce type de problème, où il reste toujours une part d'imprévisibilité."

### 3.4 MAPE (Mean Absolute Percentage Error)

**Formule mathématique :**
$$ MAPE = \frac{100\%}{n} \sum_{i=1}^{n} \left| \frac{y_i - \hat{y}_i}{y_i} \right| $$

**Interprétation :**
Erreur moyenne en pourcentage.

**Pourquoi l'utiliser ?**
- Facilement interprétable par les non-experts
- Sans unité (comparaison entre différents problèmes)
- Intuitif pour la direction ("erreur de X%")

**Limites importantes :**
1. **Undefined si $$ y_i = 0 $$** (division par zéro)
2. **Asymétrique** : pénalise plus les sous-estimations que les sur-estimations
3. **Biaisé vers les petites valeurs**

**Exemple de biais :**
- Prédire 90 au lieu de 100 : erreur = 10%
- Prédire 110 au lieu de 100 : erreur = 9.1%
→ Asymétrie !

**Alternative symétrique : sMAPE**
$$ sMAPE = \frac{100\%}{n} \sum_{i=1}^{n} \frac{|y_i - \hat{y}_i|}{(|y_i| + |\hat{y}_i|)/2} $$

**Visualisation PowerPoint recommandée :**
- Affichage en pourcentage (très visuel)
- Distribution des erreurs par gamme de valeurs
- Comparaison avec tolérance métier acceptable

**Comment présenter à la direction :**
"Le modèle prédit avec une erreur moyenne de 8%, ce qui est inférieur à la tolérance de 10% fixée. Pour un budget de 100K€, l'erreur typique sera de 8K€."

### 3.5 Graphique Prédictions vs Réel

**Principe :**
Scatter plot avec :
- **Axe X** : Valeurs réelles
- **Axe Y** : Valeurs prédites
- **Diagonale** : Prédictions parfaites (y = x)

**Comment interpréter :**
- **Points sur la diagonale** : Prédictions exactes
- **Points au-dessus** : Sur-estimation
- **Points en-dessous** : Sous-estimation
- **Dispersion** : Variance de l'erreur

**Patterns à identifier :**

1. **Biais systématique** :
   - Points systématiquement au-dessus → Sur-estimation
   - Points systématiquement en-dessous → Sous-estimation

2. **Hétéroscédasticité** :
   - Dispersion qui augmente avec la valeur
   - Indique que l'erreur n'est pas constante

3. **Non-linéarité** :
   - Forme courbée
   - Le modèle linéaire ne capture pas la relation réelle

**Visualisation PowerPoint recommandée :**
- Scatter plot avec ligne de référence (y=x)
- Colorier par densité ou par erreur
- Annotations pour les outliers notables
- Ajouter R² sur le graphique

**Bonnes pratiques :**
- Limiter le nombre de points si trop dense (sous-échantillonner ou hexbin)
- Ajouter une ligne de tendance
- Identifier visuellement les zones problématiques

**Comment présenter à la direction :**
"Ce graphique compare les prédictions (axe vertical) aux valeurs réelles (axe horizontal). Plus les points sont proches de la ligne, plus les prédictions sont précises. On observe une légère sous-estimation pour les valeurs élevées."

### 3.6 Analyse des résidus

**Définition :**
Résidu = Erreur = $$ r_i = y_i - \hat{y}_i $$

**Graphiques essentiels :**

#### 1. Résidus vs Prédictions
- **Axe X** : Valeurs prédites
- **Axe Y** : Résidus

**Idéalement :**
- Résidus centrés autour de 0
- Pas de pattern (nuage de points aléatoire)
- Variance constante (homoscédasticité)

**Patterns problématiques :**
- **Entonnoir** : Variance augmente avec la prédiction
- **Courbe** : Relation non-linéaire non capturée
- **Décalage de 0** : Biais systématique

#### 2. Distribution des résidus
- Histogramme ou QQ-plot
- **Idéalement** : Distribution normale centrée sur 0

**Pourquoi c'est important ?**
- Valider les hypothèses du modèle (régression linéaire suppose des résidus normaux)
- Identifier des outliers
- Détecter des problèmes de modélisation

**Visualisation PowerPoint recommandée :**
- Graphique résidus vs prédictions
- Histogramme des résidus
- Statistiques : Moyenne (proche de 0 ?), Écart-type

**Comment présenter à la direction :**
"L'analyse des erreurs montre qu'elles sont bien réparties autour de zéro, sans biais systématique. Quelques valeurs extrêmes (outliers) ont été identifiées et nécessitent une investigation."

### 3.7 Distribution des erreurs

**Graphique :**
Histogramme ou boxplot des erreurs absolues.

**Pourquoi c'est utile ?**
- Complète MAE/RMSE en montrant la distribution
- Identifie les percentiles (médiane, P90, P95, etc.)
- Montre si quelques gros outliers tirent les métriques

**Métriques complémentaires :**
- **Erreur médiane** : Moins sensible aux outliers que la moyenne
- **P90 / P95** : "90% des prédictions ont une erreur inférieure à X"
- **Max error** : Pire cas

**Visualisation PowerPoint recommandée :**
- Histogramme avec annotations des percentiles
- Boxplot montrant médiane, quartiles, outliers
- Tableau récapitulatif (médiane, P90, P95, max)

**Comment présenter à la direction :**
"Si 90% des prédictions ont une erreur inférieure à 30K€, cela signifie que dans la grande majorité des cas, le modèle est fiable. Les 10% restants nécessitent une vérification humaine."

---

## 4. NLP : MÉTRIQUES & VISUALISATIONS SPÉCIFIQUES

### 4.1 Classification de texte (BERT, Transformers)

**Métriques identiques à la classification standard :**
- Accuracy, Precision, Recall, F1-Score
- Matrice de confusion
- Courbes ROC et Precision-Recall

**Spécificités NLP :**

#### Analyse des erreurs par type de texte
- Longueur du texte (courts vs longs)
- Présence de mots-clés spécifiques
- Ambiguïté linguistique

#### Exemples d'erreurs
**Crucial pour le debugging et la communication**

Tableau recommandé :

|                   Texte                  |  Vérité | Prédiction | Confiance |      Commentaire      |
|:----------------------------------------:|:-------:|:----------:|:---------:|:---------------------:|
| "Produit correct mais livraison tardive" | Négatif | Positif    | 0.72      | Sentiment mixte       |
| "Pas mal du tout !"                      | Positif | Négatif    | 0.65      | Négation non capturée |

**Visualisation PowerPoint recommandée :**
- Matrice de confusion avec exemples de textes
- Tableau des erreurs les plus confiantes (faux avec haute probabilité)
- Word clouds des mots associés aux erreurs

**Comment présenter à la direction :**
"Le modèle atteint 91% de F1-Score sur la classification de sentiments. Les erreurs principales concernent les avis mixtes et l'ironie, que nous améliorons avec des données supplémentaires."

### 4.2 NER (Named Entity Recognition) & Extraction d'informations

**Métriques principales :**
- **F1-Score par type d'entité** (PER, ORG, LOC, DATE, etc.)
- **Exact Match** : Entité détectée avec les bons délimiteurs
- **Partial Match** : Entité détectée mais délimiteurs incorrects

**Types d'erreurs spécifiques :**
1. **Span incorrect** : "New York City" détecté comme "York"
2. **Type incorrect** : "Apple" classé comme fruit au lieu d'ORG
3. **Entité manquée** : "Microsoft" non détecté
4. **Faux positif** : "Lundi" détecté comme une organisation

**Visualisation PowerPoint recommandée :**
- F1-Score par type d'entité (graphique en barres)
- Exemples annotés avec couleurs par entité
- Matrice de confusion des types d'entités

**Exemple de visualisation annotée :**
```
Texte original:
"Apple et Microsoft ont signé un accord à San Francisco."
Détection correcte:
[Apple]_ORG et [Microsoft]_ORG ont signé un accord à [San Francisco]_LOC.
Erreur typique:
[Apple]_PROD et [Microsoft]_ORG ont signé un accord à San Francisco.
```
**Comment présenter à la direction :**
"Le système extrait correctement 89% des noms d'entreprises et 92% des lieux. Les entités de type 'produit' sont plus difficiles (78%) car elles ressemblent à des noms communs."

### 4.3 Génération de texte (LLM)

**Métriques automatiques :**

#### BLEU (Bilingual Evaluation Understudy)
**Usage principal :** Traduction automatique, génération de texte

**Principe :**
Mesure le chevauchement de n-grams entre texte généré et référence(s).

**Formule simplifiée :**
$$ BLEU = BP \times \exp\left(\sum_{n=1}^{N} w_n \log p_n\right) $$

Où :
- $$ p_n $$ = précision des n-grams (unigrams, bigrams, trigrams, 4-grams)
- $$ BP $$ = Brevity Penalty (pénalise les textes trop courts)
- Typiquement : BLEU-4 (jusqu'à 4-grams)

**Interprétation :**
- Score entre 0 et 1 (souvent x100 pour 0-100)
- **> 50** : Excellente qualité
- **30-50** : Bonne qualité
- **< 30** : Qualité faible

**Limites :**
- Ne capture pas le sens sémantique
- Nécessite une référence exacte
- Biaisé vers les correspondances exactes

#### ROUGE (Recall-Oriented Understudy for Gisting Evaluation)
**Usage principal :** Résumés automatiques

**Variantes :**
- **ROUGE-N** : Chevauchement de n-grams (comme BLEU mais focus sur recall)
- **ROUGE-L** : Plus longue sous-séquence commune
- **ROUGE-W** : Sous-séquence pondérée
- **ROUGE-S** : Skip-bigrams

**Exemple ROUGE-1 (unigrams) :**
- Référence : "Le chat mange une souris"
- Généré : "Un chat mange la souris"
- Mots communs : chat, mange, souris (3/5)
- ROUGE-1 Recall = 3/5 = 0.60

**Quand utiliser BLEU vs ROUGE ?**
- **BLEU** : Précision importante (traduction, génération précise)
- **ROUGE** : Recall important (résumés, capturer l'essentiel)

#### BERTScore
**Principe :**
Utilise les embeddings contextuels de BERT pour comparer similarité sémantique.

**Avantages :**
- Capture le sens, pas seulement les mots exacts
- "voiture" et "automobile" sont considérés similaires
- Plus robuste aux paraphrases

**Score :**
- Precision, Recall et F1 basés sur similarité cosinus des embeddings

#### Perplexité
**Définition :**
Mesure de la "surprise" du modèle face à une séquence de mots.

**Formule :**
$$ Perplexity = \exp\left(-\frac{1}{N}\sum_{i=1}^{N} \log P(w_i | w_{1:i-1})\right) $$

**Interprétation :**
- **Faible perplexité** : Le modèle prédit bien (comprend le texte)
- **Haute perplexité** : Le modèle est "perplexe" (ne comprend pas)

**Exemple intuitu :**
- Texte : "Le soleil brille"
- Si perplexité = 10 → en moyenne, le modèle hésite entre 10 mots à chaque position
- Si perplexité = 100 → le modèle est beaucoup plus incertain

**Usage :**
- Évaluation de modèles de langage
- Comparaison de modèles (plus bas = meilleur)
- Ne mesure PAS directement la qualité de génération pour l'humain

**Visualisation PowerPoint recommandée :**
- Tableau comparatif : BLEU, ROUGE, BERTScore, Perplexité
- Exemples de générations avec scores
- Graphique d'évolution pendant l'entraînement

**Métriques humaines (gold standard) :**

#### Évaluation humaine
**Critères typiques :**
1. **Fluency** : Le texte est-il naturel, grammaticalement correct ?
2. **Coherence** : Le texte est-il cohérent, logique ?
3. **Relevance** : Le texte répond-il à la consigne ?
4. **Informativeness** : Le texte apporte-t-il des informations utiles ?

**Format :**
- Échelle de Likert (1-5)
- Comparaison A/B (quel texte est meilleur ?)
- Classement de plusieurs générations

**Pourquoi c'est essentiel ?**
Les métriques automatiques ne corrèlent pas toujours avec la qualité perçue par les humains.

**Visualisation PowerPoint recommandée :**
- Graphiques radar avec les différents critères
- Exemples annotés par des humains
- Comparaison modèle vs baseline avec votes humains

**Comment présenter à la direction :**
"Le modèle génère des résumés avec un ROUGE-L de 0.67, comparable à l'état de l'art. L'évaluation humaine confirme que 85% des résumés sont jugés 'bons' ou 'excellents' en termes de qualité et de pertinence."

### 4.4 Visualisations spécifiques NLP

#### Attention Weights
**Pour Transformers/BERT :**
Visualise quels mots le modèle "attend to" (prête attention à).

**Usage :**
- Comprendre les décisions du modèle
- Identifier les mots clés
- Débugger des erreurs

**Format :**
- Heatmap : mots source × mots cible
- Plus la couleur est intense, plus l'attention est forte

**Visualisation PowerPoint recommandée :**
- Heatmap avec annotations sur les mots importants
- Exemples de bonnes et mauvaises attentions
- Interprétation qualitative

#### Word Embeddings (t-SNE / UMAP)
**Principe :**
Visualiser les embeddings de mots en 2D pour comprendre la structure sémantique.

**Usage :**
- Vérifier que les mots similaires sont proches
- Identifier des clusters sémantiques
- Valider la qualité des représentations

**Visualisation PowerPoint recommandée :**
- Scatter plot 2D avec labels sur quelques points
- Colorier par catégorie sémantique
- Zoomer sur des zones intéressantes

**Comment présenter à la direction :**
"Cette visualisation montre comment le modèle organise les mots. On voit que 'chien', 'chat', 'animal' sont proches, confirmant que le modèle capture les relations sémantiques."

#### Exemples de prédictions
**Le plus important pour la communication !**

**Format recommandé :**
Tableau avec :
- Texte d'entrée
- Sortie attendue
- Sortie prédite
- Score de confiance
- Commentaire / Explication

**Stratégie :**
- Montrer des réussites (renforcer la confiance)
- Montrer des échecs typiques (transparence, axes d'amélioration)
- Varier les exemples (courts/longs, faciles/difficiles)

---

## 5. COMPUTER VISION : MÉTRIQUES & VISUALISATIONS SPÉCIFIQUES

### 5.1 Classification d'images

**Métriques identiques à la classification standard :**
- Accuracy, Precision, Recall, F1-Score
- Matrice de confusion
- Top-1 et Top-5 Accuracy (pour multi-classes nombreuses)

**Top-5 Accuracy :**
Le modèle prédit les 5 classes les plus probables. Correct si la vraie classe est dans le top 5.

**Usage :** ImageNet (1000 classes), où prédire exactement la bonne classe est difficile.

**Visualisations spécifiques :**
- Images avec prédictions et probabilités
- Galerie d'erreurs (faux positifs/négatifs)
- Heatmaps d'activation (Grad-CAM, Saliency maps)

### 5.2 Détection d'objets (YOLO, etc.)

#### IoU (Intersection over Union)

**Formule mathématique :**
$$ IoU = \frac{\text{Aire}(A \cap B)}{\text{Aire}(A \cup B)} = \frac{\text{Intersection}}{\text{Union}} $$

Où :
- $$ A $$ = Bounding box prédite
- $$ B $$ = Bounding box réelle (ground truth)

**Interprétation :**
- **IoU = 1** : Parfait chevauchement
- **IoU = 0.5** : Seuil typique pour considérer une détection comme correcte
- **IoU = 0** : Aucun chevauchement

**Visualisation :**

**Pourquoi c'est important ?**
C'est la base de toutes les métriques de détection d'objets.

#### Precision et Recall en détection

**Différence avec la classification :**
Une détection est "correcte" si :
1. La classe prédite = classe réelle
2. IoU > seuil (généralement 0.5)

**Définitions :**
- **TP** : Détection correcte (bonne classe + IoU > seuil)
- **FP** : Détection incorrecte (mauvaise classe ou IoU < seuil)
- **FN** : Objet manqué (non détecté)

$$ Precision = \frac{TP}{TP + FP} = \frac{\text{Détections correctes}}{\text{Toutes les détections}} $$

$$ Recall = \frac{TP}{TP + FN} = \frac{\text{Détections correctes}}{\text{Tous les objets réels}} $$

#### AP (Average Precision)

**Définition :**
Aire sous la courbe Precision-Recall pour une classe donnée.

**Calcul :**
1. Trier les détections par score de confiance décroissant
2. Calculer Precision et Recall à chaque seuil
3. Calculer l'aire sous la courbe

**Interprétation :**
- **AP = 1** : Détection parfaite
- **AP = 0.5** : Performance moyenne
- **AP = 0** : Aucune détection correcte

#### mAP (mean Average Precision)

**Formule :**
$$ mAP = \frac{1}{C} \sum_{c=1}^{C} AP_c $$

Où $$ C $$ = nombre de classes

**Variantes importantes :**
- **mAP@0.5** : IoU seuil = 0.5 (standard PASCAL VOC)
- **mAP@0.75** : IoU seuil = 0.75 (plus strict)
- **mAP@[0.5:0.95]** : Moyenne sur plusieurs seuils IoU de 0.5 à 0.95 par pas de 0.05 (standard COCO)

**Pourquoi plusieurs seuils ?**
- IoU@0.5 est permissif (détection approximative acceptable)
- IoU@0.75 est strict (localisation précise requise)
- mAP@[0.5:0.95] évalue la précision de localisation

**Exemple :**
YOLO sur COCO dataset :
- mAP@0.5 = 0.65 (65% de détections avec IoU > 0.5)
- mAP@0.5:0.95 = 0.45 (45% en moyenne sur tous les seuils)

**Visualisation PowerPoint recommandée :**
- Tableau avec AP par classe
- mAP global en gras
- Comparaison avec modèles de référence (baseline, state-of-the-art)
- Graphiques en barres : AP par classe

**Comment présenter à la direction :**
"Le modèle atteint un mAP de 72%, ce qui signifie qu'il détecte correctement 72% des objets avec une localisation précise. Il performe particulièrement bien sur les véhicules (85%) et moins bien sur les piétons (60%)."

### 5.3 Visualisations spécifiques Computer Vision

#### Détections visuelles
**Le plus important pour communiquer !**

**Format :**
Images avec bounding boxes colorées, labels et scores de confiance.

**Bonnes pratiques :**
- Colorier par classe (code couleur cohérent)
- Afficher le score de confiance
- Montrer à la fois les réussites et les échecs
- Annoter les cas problématiques

**Exemples à montrer :**
1. **True Positives** : Détections correctes (renforce la confiance)
2. **False Positives** : Fausses détections (comprendre les limites)
3. **False Negatives** : Objets manqués (identifier les faiblesses)
4. **Cas limites** : Occlusions, objets petits, etc.

**Visualisation PowerPoint recommandée :**
- Grille d'images 2×2 ou 3×3
- Légendes claires
- Annotations explicatives sur les erreurs

#### Matrice de confusion spatiale
**Pour multi-classes :**
Montre quelles classes sont confondues entre elles.

**Exemple :**

```
          Prédit Voiture  Prédit Camion  Prédit Moto
          Prédit Voiture  Prédit Camion  Prédit Moto
```
