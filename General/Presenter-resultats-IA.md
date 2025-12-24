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
$$Accuracy = \frac{TP + TN}{TP + TN + FP + FN} = \frac{\text{Prédictions correctes}}{\text{Total des prédictions}}$$

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
$$Precision = \frac{TP}{TP + FP} = \frac{\text{Vrais positifs}}{\text{Tous les positifs prédits}}$$

**Interprétation :** "Parmi tous les cas que le modèle a identifiés comme positifs, quelle proportion l'est vraiment ?"

**Quand l'utiliser ?**
- **CoCoût élevé des faux positifs (FP)**
- Exemples métier :
  - Spam : Un vrai email classé comme spam (FP) = client mécontent
  - Recommandations produits : Mauvaise recommandation (FP) = perte de confiance
  - Alertes médicales : Fausse alerte (FP) = tests inutiles coûteux

**Pourquoi optimiser la Precision ?**
Dans certains contextes, il vaut mieux être sûr de ce qu'on détecte, quitte à en rater quelques-uns.

**Analogie** : Tu es un pêcheur et tu veux attraper des thons. La précision mesure : parmi tous les poissons que tu as attrapés, quelle proportion sont vraiment des thons ?
- Si tu attrapes 100 poissons et que 80 sont des thons → Précision = 80%
- Si tu attrapes beaucoup de sardines par erreur → Ta précision baisse

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
$$Recall = \frac{TP}{TP + FN} = \frac{\text{Vrais positifs}}{\text{Tous les positifs réels}}$$

**Interprétation :** "Parmi tous les cas réellement positifs, quelle proportion le modèle a-t-il détectée ?"

**Quand l'utiliser ?**
- **Coût élevé des faux négatifs (FN)**
- Exemples métier :
  - Détection de maladies graves : Rater un cancer (FN) = risque vital
  - Détection de fraudes : Rater une fraude (FN) = perte financière
  - Détection d'intrusions : Rater une cyberattaque (FN) = compromission du système

**Pourquoi optimiser le Recall ?**
Dans certains contextes, il est crucial de ne rien rater, même si cela génère des fausses alertes.

**Analogie** : Il y a 200 thons dans l'océan. Le rappel mesure : combien de ces 200 thons as-tu réussi à attraper ?
- Si tu attrapes 150 thons sur les 200 → Rappel = 75%
- Même si tu n'attrapes que des thons (précision = 100%), si tu en rates beaucoup → Ton rappel est faible


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
$$F1 = 2 \times \frac{Precision \times Recall}{Precision + Recall} = \frac{2 \times TP}{2 \times TP + FP + FN}$$

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
   $$F1_{macro} = \frac{1}{n} \sum_{i=1}^{n} F1_i$$

2. **F1 Micro** : F1 global calculé sur l'ensemble des TP, FP, FN  
   $$F1_{micro} = \frac{2 \times TP_{total}}{2 \times TP_{total} + FP_{total} + FN_{total}}$$

3. **F1 Weighted** : Moyenne pondérée par le nombre d'échantillons par classe  
   $$F1_{weighted} = \sum_{i=1}^{n} \frac{n_i}{N} F1_i$$ 

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
$$MAE = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$$

Où :
- $y_i$ = valeur réelle
- $\hat{y}_i$ = valeur prédite
- $n$ = nombre d'observations

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
$$RMSE = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}$$

**Interprétation :**
Racine carrée de la moyenne des erreurs au carré, dans les mêmes unités que la variable cible.

**Pourquoi l'utiliser ?**
- Pénalise davantage les grandes erreurs (grâce au carré)
- Utile quand les grandes erreurs sont particulièrement coûteuses
- Standard dans la communauté ML

**Différence MAE vs RMSE :**

```python
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
```

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
$$R^2 = 1 - \frac{SS_{res}}{SS_{tot}} = 1 - \frac{\sum_{i=1}^{n} (y_i - \hat{y}_i)^2}{\sum_{i=1}^{n} (y_i - \bar{y})^2}$$

Où :
- $SS_{res}$ = Somme des carrés des résidus (erreur du modèle)
- $SS_{tot}$ = Somme des carrés totale (variance des données)
- $\bar{y}$ = Moyenne des valeurs réelles

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
$$MAPE = \frac{100\%}{n} \sum_{i=1}^{n} \left| \frac{y_i - \hat{y}_i}{y_i} \right|$$

**Interprétation :**
Erreur moyenne en pourcentage.

**Pourquoi l'utiliser ?**
- Facilement interprétable par les non-experts
- Sans unité (comparaison entre différents problèmes)
- Intuitif pour la direction ("erreur de X%")

**Limites importantes :**
1. **Undefined si $y_i = 0$** (division par zéro)
2. **Asymétrique** : pénalise plus les sous-estimations que les sur-estimations
3. **Biaisé vers les petites valeurs**

**Exemple de biais :**
- Prédire 90 au lieu de 100 : erreur = 10%
- Prédire 110 au lieu de 100 : erreur = 9.1%
→ Asymétrie !

**Alternative symétrique : sMAPE**  
$$sMAPE = \frac{100\%}{n} \sum_{i=1}^{n} \frac{|y_i - \hat{y}_i|}{(|y_i| + |\hat{y}_i|)/2}$$

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
Résidu = Erreur = $r_i = y_i - \hat{y}_i$

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
$$BLEU = BP \times \exp\left(\sum_{n=1}^{N} w_n \log p_n\right)$$

Où :
- $p_n$ = précision des n-grams (unigrams, bigrams, trigrams, 4-grams)
- $BP$ = Brevity Penalty (pénalise les textes trop courts)
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
$$Perplexity = \exp\left(-\frac{1}{N}\sum_{i=1}^{N} \log P(w_i | w_{1:i-1})\right)$$

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
$$IoU = \frac{\text{Aire}(A \cap B)}{\text{Aire}(A \cup B)} = \frac{\text{Intersection}}{\text{Union}}$$

Où :
- $A$ = Bounding box prédite
- $B$ = Bounding box réelle (ground truth)

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
$$mAP = \frac{1}{C} \sum_{c=1}^{C} AP_c$$

Où $C$ = nombre de classes

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

---
---
---

# Guide Complet : Présentation des Résultats de Modèles IA

**Guide de référence pour présenter des résultats de modèles IA de manière claire, professionnelle et pertinente**

*Auteur : Guide pour Clément O.*  
*Contexte : Présentations PowerPoint pour direction, utilisateurs et experts IA*  
*Domaines : NLP (Transformers/BERT), Computer Vision (YOLO), Classification, Régression*

---

## Table des matières

1. [Métriques de Classification](#1-metriques-de-classification)
2. [Métriques de Détection d'Objets (YOLO)](#2-métriques-de-détection-dobjets-yolo)
3. [Métriques de Régression](#3-métriques-de-régression)
4. [Visualisations Essentielles pour PowerPoint](#4-visualisations-essentielles-pour-powerpoint)
5. [Métriques Business et ROI](#5-métriques-business-et-roi)
6. [Communication de l'Incertitude](#6-communication-de-lincertitude)
7. [Présentation des Erreurs (Failure Cases)](#7-présentation-des-erreurs-failure-cases)
8. [Comparaison de Modèles](#8-comparaison-de-modèles)
9. [Adaptation selon l'Audience](#9-adaptation-selon-laudience)
10. [Structure Type de Présentation PowerPoint](#10-structure-type-de-présentation-powerpoint)

---

## 1. Métriques de Classification

*Utilisées pour : Classification d'emails, classification d'images, détection binaire/multi-classes*

### 1.1 Matrice de Confusion

**Définition :**  
Tableau croisant les prédictions du modèle avec les vraies valeurs.

**Structure (cas binaire) :**

|                     | **Prédit Positif** | **Prédit Négatif** |
|---------------------|--------------------|--------------------|
| **Réel Positif**    | TP (Vrais Positifs)| FN (Faux Négatifs) |
| **Réel Négatif**    | FP (Faux Positifs) | TN (Vrais Négatifs)|

**Pourquoi l'utiliser ?**
- Vision instantanée des types d'erreurs
- Identification des classes confondues (en multi-classes)
- Essentiel pour expliquer les performances à la direction

**Exemple de code :**

```python
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# y_true : vraies étiquettes, y_pred : prédictions
y_true = [0, 1, 0, 1, 1, 0, 1, 0]
y_pred = [0, 1, 0, 0, 1, 0, 1, 1]

cm = confusion_matrix(y_true, y_pred)

# Visualisation
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Négatif', 'Positif'],
            yticklabels=['Négatif', 'Positif'])
plt.ylabel('Vraie classe')
plt.xlabel('Classe prédite')
plt.title('Matrice de Confusion')
plt.show()
```

**Pour PowerPoint :**
- Utilise une heatmap avec annotations
- Ajoute des pourcentages en plus des valeurs brutes
- Version normalisée pour comparer entre datasets

---

### 1.2 Accuracy (Exactitude)

**Formule mathématique :**  

$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$

**Interprétation :**  
Proportion de prédictions correctes parmi toutes les prédictions.

**Pourquoi l'utiliser ?**
- Métrique la plus simple et intuitive pour la direction
- Bonne vue d'ensemble si les classes sont équilibrées

**⚠️ Piège à éviter :**
- **Classes déséquilibrées** : Si 95% des emails sont "non-urgents", un modèle prédisant toujours "non-urgent" aura 95% d'accuracy mais sera inutile !

**Quand l'utiliser ?**
- ✅ Classes équilibrées (ratio 40-60 minimum)
- ✅ Présentation rapide à la direction (mais toujours accompagner d'autres métriques)
- ❌ Classes très déséquilibrées (fraude, anomalies, emails urgents rares)

```python
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_true, y_pred)
print(f"Accuracy : {accuracy:.2%}")  # 75.00%
```

---

### 1.3 Precision (Précision)

**Formule mathématique :**  

$$\text{Precision} = \frac{TP}{TP + FP}$$

**Interprétation :**  
Parmi les emails classés "urgents" par le modèle, quelle proportion l'est réellement ?

**Analogie** : Tu es un pêcheur et tu veux attraper des thons. La précision mesure : parmi tous les poissons que tu as attrapés, quelle proportion sont vraiment des thons ?
- Si tu attrapes 100 poissons et que 80 sont des thons → Précision = 80%
- Si tu attrapes beaucoup de sardines par erreur → Ta précision baisse

**Pourquoi l'utiliser ?**
- Quand les **Faux Positifs sont coûteux**
- Exemple : Classification "urgent" → si beaucoup de FP, on perd du temps à traiter des emails non-urgents

**Cas d'usage typique :**
- Filtrage spam (on ne veut pas bloquer des emails légitimes)
- Détection de produits défectueux (coût d'inspection inutile)

```python
from sklearn.metrics import precision_score

precision = precision_score(y_true, y_pred)
print(f"Precision : {precision:.2%}")
```

---

### 1.4 Recall (Rappel / Sensibilité)

**Formule mathématique :**  

$$\text{Recall} = \frac{TP}{TP + FN}$$

**Interprétation :**  
Parmi tous les emails réellement urgents, quelle proportion a été détectée par le modèle ?

**Analogie** : Il y a 200 thons dans l'océan. Le rappel mesure : combien de ces 200 thons as-tu réussi à attraper ?
- Si tu attrapes 150 thons sur les 200 → Rappel = 75%
- Même si tu n'attrapes que des thons (précision = 100%), si tu en rates beaucoup → Ton rappel est faible

**Pourquoi l'utiliser ?**
- Quand les **Faux Négatifs sont coûteux**
- Exemple : Classification "urgent" → si beaucoup de FN, on rate des emails urgents !

**Cas d'usage typique :**
- Détection de maladies (on ne veut pas rater un malade)
- Détection d'anomalies critiques
- Emails urgents clients VIP

```python
from sklearn.metrics import recall_score

recall = recall_score(y_true, y_pred)
print(f"Recall : {recall:.2%}")
```

---

### 1.5 F1-Score

**Formule mathématique :**  

$\$F1 = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

**Interprétation :**  
Moyenne harmonique entre Precision et Recall. Équilibre entre les deux.

**Pourquoi l'utiliser ?**
- Quand Precision ET Recall sont importants
- **Métrique unique** pour comparer des modèles
- Gère mieux les classes déséquilibrées que l'Accuracy

**⚠️ Attention :**
- F1 = 0.80 ne veut pas dire 80% de performance globale
- Un F1 élevé n'est bon que si Precision ET Recall sont équilibrés

```python
from sklearn.metrics import f1_score

f1 = f1_score(y_true, y_pred)
print(f"F1-Score : {f1:.2%}")
```

**Trade-off Precision vs Recall :**

Imaginons un modèle de classification d'emails urgents :
- **Seuil bas (0.3)** : Le modèle prédit "urgent" facilement → Recall élevé mais Precision basse (beaucoup de FP)
- **Seuil élevé (0.8)** : Le modèle est prudent → Precision élevée mais Recall bas (on rate des urgents)

```python
from sklearn.metrics import precision_recall_curve

# Obtenir les probabilités prédites
y_proba = [0.1, 0.9, 0.2, 0.4, 0.85, 0.15, 0.95, 0.6]

precision, recall, thresholds = precision_recall_curve(y_true, y_proba)

plt.figure(figsize=(10, 6))
plt.plot(thresholds, precision[:-1], label='Precision')
plt.plot(thresholds, recall[:-1], label='Recall')
plt.xlabel('Seuil de décision')
plt.ylabel('Score')
plt.title('Trade-off Precision vs Recall selon le seuil')
plt.legend()
plt.grid(True)
plt.show()
```

---

### 1.6 Specificité (Specificity)

**Formule mathématique :**  

$$\text{Specificity} = \frac{TN}{TN + FP}$$

**Interprétation :**  
Parmi tous les emails NON-urgents, quelle proportion a été correctement identifiée ?

**Pourquoi l'utiliser ?**
- Complémentaire au Recall
- Important en médical : capacité à identifier les personnes saines

---

### 1.7 Balanced Accuracy

**Formule mathématique :**  

$$\text{Balanced Accuracy} = \frac{\text{Recall}_{\text{classe 1}} + \text{Specificity}}{2}$$

**Pourquoi l'utiliser ?**
- **Classes déséquilibrées** : Meilleure alternative à l'Accuracy classique
- Donne le même poids à chaque classe

**Exemple :**

```python
from sklearn.metrics import balanced_accuracy_score

balanced_acc = balanced_accuracy_score(y_true, y_pred)
print(f"Balanced Accuracy : {balanced_acc:.2%}")
```

---

### 1.8 Courbe ROC et AUC

**ROC (Receiver Operating Characteristic) :**  
Courbe montrant le trade-off entre Recall (Taux de Vrais Positifs) et Taux de Faux Positifs à différents seuils.

**AUC (Area Under Curve) :**  
Aire sous la courbe ROC. Varie entre 0 et 1.

**Interprétation de l'AUC :**
- **AUC = 0.5** : Modèle aléatoire (inutile)
- **AUC = 0.7-0.8** : Acceptable
- **AUC = 0.8-0.9** : Bon modèle
- **AUC > 0.9** : Excellent modèle
- **AUC = 1.0** : Modèle parfait (ou overfitting !)

**Pourquoi l'utiliser ?**
- Évalue la capacité du modèle à discriminer entre classes
- **Indépendant du seuil** de décision choisi
- Très utilisé en recherche et facile à présenter

```python
from sklearn.metrics import roc_curve, auc

fpr, tpr, thresholds = roc_curve(y_true, y_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(10, 8))
plt.plot(fpr, tpr, color='darkorange', lw=2, 
         label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
         label='Aléatoire (AUC = 0.50)')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Taux de Faux Positifs (1 - Specificité)')
plt.ylabel('Taux de Vrais Positifs (Recall)')
plt.title('Courbe ROC')
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.show()
```

**Pour PowerPoint :**
- Ajoute la ligne de référence (diagonale) pour montrer le niveau aléatoire
- Indique clairement la valeur de l'AUC
- Si comparaison de modèles : superpose plusieurs courbes ROC

---

### 1.9 Courbe Précision-Recall

**Pourquoi l'utiliser plutôt que ROC ?**
- **Classes très déséquilibrées** : Plus informative que ROC
- Exemple : détection d'anomalies (1% de positifs), fraude, emails urgents rares

**Différence ROC vs Precision-Recall :**

```python
from sklearn.metrics import precision_recall_curve, average_precision_score

precision, recall, _ = precision_recall_curve(y_true, y_proba)
avg_precision = average_precision_score(y_true, y_proba)

plt.figure(figsize=(10, 8))
plt.plot(recall, precision, color='blue', lw=2,
         label=f'PR curve (AP = {avg_precision:.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Courbe Précision-Recall')
plt.legend(loc="upper right")
plt.grid(True, alpha=0.3)
plt.show()
```

**Cookbook : ROC ou PR ?**
- ✅ **ROC** : Classes équilibrées, présentation à des non-experts
- ✅ **Precision-Recall** : Classes déséquilibrées, focus sur la classe minoritaire

---

### 1.10 Métriques Multi-Classes

**Macro-Average :**  
Moyenne simple des métriques de chaque classe (toutes les classes ont le même poids).

$$\text{Macro-F1} = \frac{1}{n} \sum_{i=1}^{n} F1_i$$

**Micro-Average :**  
Agrège les TP, FP, FN de toutes les classes puis calcule la métrique globale.

**Weighted-Average :**  
Moyenne pondérée par le nombre d'échantillons de chaque classe.

**Quand utiliser quoi ?**
- **Macro** : Toutes les classes ont la même importance (même si déséquilibrées)
- **Micro** : Focus sur la performance globale
- **Weighted** : Importance proportionnelle au nombre d'échantillons

```python
from sklearn.metrics import classification_report

# Exemple multi-classes : urgence email (0=normal, 1=important, 2=urgent)
y_true_multi = [0, 1, 2, 0, 1, 2, 0, 1]
y_pred_multi = [0, 1, 1, 0, 1, 2, 0, 2]

print(classification_report(y_true_multi, y_pred_multi, 
                           target_names=['Normal', 'Important', 'Urgent']))
```

---

## 2. Métriques de Détection d'Objets (YOLO)

*Utilisées pour : Détection automatisée par computer vision, segmentation, localisation*

### 2.1 IoU (Intersection over Union)

**Formule mathématique :**  

$$\text{IoU} = \frac{\text{Aire de l'intersection}}{\text{Aire de l'union}} = \frac{A \cap B}{A \cup B}$$

**Interprétation :**  
Mesure le chevauchement entre la bounding box prédite et la bounding box réelle.

**Valeurs typiques :**
- **IoU ≥ 0.5** : Détection considérée comme correcte (seuil standard)
- **IoU ≥ 0.75** : Détection de haute qualité
- **IoU < 0.5** : Faux Positif

**Pourquoi l'utiliser ?**
- Détermine si une détection est un TP ou un FP
- Essentiel pour calculer la Precision et le Recall en détection d'objets

**Visualisation :**

```python
def calculate_iou(box1, box2):
    """
    box format: [x1, y1, x2, y2] (top-left et bottom-right)
    """
    # Coordonnées de l'intersection
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])
    
    # Aire de l'intersection
    inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)
    
    # Aires des deux boxes
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    # Aire de l'union
    union_area = box1_area + box2_area - inter_area
    
    # IoU
    iou = inter_area / union_area if union_area > 0 else 0
    return iou

# Exemple
box_ground_truth = [50, 50, 150, 150]  # Vraie boîte
box_predicted = [60, 60, 160, 140]     # Prédiction

iou = calculate_iou(box_ground_truth, box_predicted)
print(f"IoU : {iou:.2f}")  # 0.68
```

---

### 2.2 mAP (mean Average Precision)

**Définition :**  
Métrique principale pour évaluer les modèles de détection d'objets.

**Calcul en étapes :**

1. **Pour chaque classe** :
   - Ordonne toutes les détections par score de confiance décroissant
   - Calcule Precision et Recall à chaque seuil
   - Calcule l'Average Precision (AP) = aire sous la courbe Précision-Recall
   
2. **mAP** = Moyenne des AP de toutes les classes

**Variantes :**
- **mAP@0.5** : IoU seuil = 0.5 (standard PASCAL VOC)
- **mAP@0.75** : IoU seuil = 0.75 (plus strict)
- **mAP@[0.5:0.95]** : Moyenne sur plusieurs seuils IoU (standard COCO)

**Pourquoi l'utiliser ?**
- **Métrique standard** pour benchmarker les modèles de détection
- Prend en compte Precision, Recall ET la localisation (via IoU)
- Permet de comparer différentes architectures (YOLOv5 vs YOLOv8 vs Faster R-CNN)

**Interprétation :**
- **mAP@0.5 = 0.60** : Performance acceptable pour des applications réelles
- **mAP@0.5 = 0.80+** : Très bon modèle
- **mAP@0.5:0.95** : Plus exigeant (souvent 20-30% inférieur au mAP@0.5)

```python
# Avec YOLOv8 (ultralytics)
from ultralytics import YOLO

model = YOLO('yolov8n.pt')
results = model.val(data='coco.yaml')  # Validation sur dataset

print(f"mAP@0.5 : {results.box.map50:.3f}")
print(f"mAP@0.5:0.95 : {results.box.map:.3f}")
```

**Pour PowerPoint :**
- Tableau comparatif des mAP entre modèles
- Graphique en barres par classe (montre les classes bien/mal détectées)

---

### 2.3 Precision et Recall en détection

**Adaptation pour détection d'objets :**

- **True Positive (TP)** : Détection avec IoU ≥ seuil et bonne classe
- **False Positive (FP)** : Détection avec IoU < seuil OU mauvaise classe OU détection en doublon
- **False Negative (FN)** : Objet non détecté

**Calcul identique à la classification :**

$$\text{Precision} = \frac{TP}{TP + FP}$$

$$\text{Recall} = \frac{TP}{TP + FN}$$

**Exemple concret (détection de défauts) :**
- 100 défauts réels dans les images
- Modèle détecte 90 défauts (dont 85 corrects et 5 faux)
- Recall = 85/100 = 0.85 (on a trouvé 85% des défauts)
- Precision = 85/90 = 0.94 (94% de nos détections sont correctes)

---

### 2.4 Confusion entre Classes (Détection)

**Pourquoi l'utiliser ?**
- Identifier les paires de classes confondues par le modèle
- Exemple : YOLO confond "voiture" et "camion"

**Visualisation :**

```python
import numpy as np
import seaborn as sns

# Matrice de confusion pour détection multi-classes
# Exemple : détection de 3 types de défauts
confusion_matrix_detection = np.array([
    [120, 10, 5],    # Défaut type A
    [8, 95, 12],     # Défaut type B
    [3, 15, 87]      # Défaut type C
])

plt.figure(figsize=(10, 8))
sns.heatmap(confusion_matrix_detection, annot=True, fmt='d', cmap='YlOrRd',
            xticklabels=['Défaut A', 'Défaut B', 'Défaut C'],
            yticklabels=['Défaut A', 'Défaut B', 'Défaut C'])
plt.ylabel('Vraie classe')
plt.xlabel('Classe prédite')
plt.title('Matrice de Confusion - Détection YOLO')
plt.show()
```

---

### 2.5 Vitesse d'Inférence (FPS, Latence)

**Métriques complémentaires critiques pour la production :**

- **FPS (Frames Per Second)** : Nombre d'images traitées par seconde
- **Latence** : Temps de traitement d'une image (ms)

**Pourquoi l'utiliser ?**
- Contraintes temps réel (vidéo surveillance, ligne de production)
- Trade-off précision vs vitesse

**Exemple :**

| Modèle      | mAP@0.5 | FPS (GPU) | Latence (ms) |
|-------------|---------|-----------|--------------|
| YOLOv8n     | 0.52    | 380       | 2.6          |
| YOLOv8s     | 0.61    | 285       | 3.5          |
| YOLOv8m     | 0.67    | 165       | 6.0          |
| YOLOv8l     | 0.70    | 110       | 9.1          |

**Pour PowerPoint :**
- Graphique scatter : mAP (axe Y) vs Latence (axe X)
- Montre le meilleur compromis précision/vitesse

```python
import matplotlib.pyplot as plt

models = ['YOLOv8n', 'YOLOv8s', 'YOLOv8m', 'YOLOv8l']
map_scores = [0.52, 0.61, 0.67, 0.70]
latency = [2.6, 3.5, 6.0, 9.1]

plt.figure(figsize=(10, 6))
plt.scatter(latency, map_scores, s=200, alpha=0.6)
for i, model in enumerate(models):
    plt.annotate(model, (latency[i], map_scores[i]), 
                 fontsize=12, ha='right')
plt.xlabel('Latence (ms)', fontsize=12)
plt.ylabel('mAP@0.5', fontsize=12)
plt.title('Trade-off Précision vs Vitesse', fontsize=14)
plt.grid(True, alpha=0.3)
plt.show()
```

---

## 3. Métriques de Régression

*Utilisées pour : Prédiction de valeurs continues (prix, délais, quantités)*

### 3.1 MAE (Mean Absolute Error)

**Formule mathématique :**  

$$\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$$

**Interprétation :**  
Erreur moyenne absolue, dans les mêmes unités que la variable cible.

**Pourquoi l'utiliser ?**
- **Interprétation directe** : "En moyenne, le modèle se trompe de X unités"
- Robuste aux outliers (pas de carré)
- Facile à expliquer à la direction

**Exemple :**
- Prédiction de temps de traitement d'emails : MAE = 5 min
- → Le modèle se trompe en moyenne de 5 minutes

```python
from sklearn.metrics import mean_absolute_error
import numpy as np

y_true = np.array([100, 120, 90, 110, 105])
y_pred = np.array([98, 125, 88, 105, 110])

mae = mean_absolute_error(y_true, y_pred)
print(f"MAE : {mae:.2f}")  # 3.80
```

---

### 3.2 RMSE (Root Mean Squared Error)

**Formule mathématique :**  

$$\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}$$

**Interprétation :**  
Racine carrée de la moyenne des erreurs au carré, dans les mêmes unités que la variable cible.

**Pourquoi l'utiliser ?**
- Pénalise davantage les **grandes erreurs** (grâce au carré)
- Utile quand les grandes erreurs sont particulièrement coûteuses
- Standard dans la communauté ML

**Différence MAE vs RMSE :**

```python
y_true = np.array([100, 100, 100, 100, 100])
y_pred1 = np.array([105, 105, 105, 105, 105])  # Erreurs constantes
y_pred2 = np.array([100, 100, 100, 100, 125])  # Une grosse erreur

mae1 = mean_absolute_error(y_true, y_pred1)
rmse1 = np.sqrt(np.mean((y_true - y_pred1)**2))
print(f"Cas 1 - MAE: {mae1:.2f}, RMSE: {rmse1:.2f}")  # MAE: 5.00, RMSE: 5.00

mae2 = mean_absolute_error(y_true, y_pred2)
rmse2 = np.sqrt(np.mean((y_true - y_pred2)**2))
print(f"Cas 2 - MAE: {mae2:.2f}, RMSE: {rmse2:.2f}")  # MAE: 5.00, RMSE: 11.18

# Même MAE mais RMSE plus élevé car une grosse erreur
```

**Quand préférer RMSE à MAE ?**
- Quand les grandes erreurs sont **inacceptables**
- Quand on veut être **conservateur**

**Quand préférer MAE à RMSE ?**
- Quand on veut une métrique **robuste aux outliers**
- Quand toutes les erreurs ont le **même coût**

---

### 3.3 R² (Coefficient de Détermination)

**Formule mathématique :**  

$\$R^2 = 1 - \frac{\sum_{i=1}^{n} (y_i - \hat{y}_i)^2}{\sum_{i=1}^{n} (y_i - \bar{y})^2}$$

Où $$\bar{y}$$ est la moyenne des valeurs réelles.

**Interprétation :**  
Proportion de la variance expliquée par le modèle.

- **R² = 1** : Modèle parfait
- **R² = 0.8** : Le modèle explique 80% de la variance
- **R² = 0** : Modèle aussi bon qu'une prédiction par la moyenne
- **R² < 0** : Modèle pire qu'une prédiction par la moyenne !

**Pourquoi l'utiliser ?**
- **Sans unité** : Permet de comparer des modèles sur différents problèmes
- Très utilisé en régression linéaire et statistiques

**⚠️ Attention :**
- R² peut être trompeur si le modèle est complexe (overfitting)
- Préférer le **R² ajusté** pour comparer des modèles avec différents nombres de variables

```python
from sklearn.metrics import r2_score

r2 = r2_score(y_true, y_pred)
print(f"R² : {r2:.3f}")
```

---

### 3.4 MAPE (Mean Absolute Percentage Error)

**Formule mathématique :**  

$$\text{MAPE} = \frac{100}{n} \sum_{i=1}^{n} \left| \frac{y_i - \hat{y}_i}{y_i} \right|$$

**Interprétation :**  
Erreur moyenne en pourcentage.

**Pourquoi l'utiliser ?**
- **Interprétation business** : "Le modèle se trompe en moyenne de X%"
- Utile pour communiquer avec la direction

**⚠️ Attention :**
- Sensible aux **valeurs proches de zéro** (division par $$y_i$$)
- Asymétrique : sous-estimation pénalisée plus que sur-estimation

```python
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

mape = mean_absolute_percentage_error(y_true, y_pred)
print(f"MAPE : {mape:.2f}%")
```

---

## 4. Visualisations Essentielles pour PowerPoint

### 4.1 Distribution des Scores de Confiance

**Pourquoi l'utiliser ?**
- Montre si le modèle est **confiant** dans ses prédictions
- Détecte les zones d'incertitude

```python
import matplotlib.pyplot as plt

# Scores de confiance pour les classes correctes vs incorrectes
correct_conf = [0.92, 0.88, 0.95, 0.85, 0.90, 0.93]
incorrect_conf = [0.55, 0.62, 0.58, 0.51, 0.68, 0.60]

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.hist(correct_conf, bins=10, alpha=0.7, color='green', label='Correct')
plt.xlabel('Score de confiance')
plt.ylabel('Fréquence')
plt.title('Prédictions Correctes')
plt.legend()

plt.subplot(1, 2, 2)
plt.hist(incorrect_conf, bins=10, alpha=0.7, color='red', label='Incorrect')
plt.xlabel('Score de confiance')
plt.ylabel('Fréquence')
plt.title('Prédictions Incorrectes')
plt.legend()

plt.tight_layout()
plt.show()
```

**Pour PowerPoint :**
- Histogrammes côte à côte (correct vs incorrect)
- Aide à justifier le seuil de décision choisi

---

### 4.2 Learning Curves (Courbes d'Apprentissage)

**Pourquoi l'utiliser ?**
- Diagnostic d'**overfitting** ou **underfitting**
- Montre l'évolution de la performance pendant l'entraînement

```python
# Exemple avec historique d'entraînement
epochs = range(1, 51)
train_loss = [0.8 - 0.01*e + np.random.rand()*0.05 for e in epochs]
val_loss = [0.75 - 0.008*e + np.random.rand()*0.08 for e in epochs]

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(epochs, train_loss, label='Train Loss', linewidth=2)
plt.plot(epochs, val_loss, label='Validation Loss', linewidth=2)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Courbes de Loss')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
train_acc = [0.6 + 0.008*e - np.random.rand()*0.02 for e in epochs]
val_acc = [0.58 + 0.007*e - np.random.rand()*0.03 for e in epochs]
plt.plot(epochs, train_acc, label='Train Accuracy', linewidth=2)
plt.plot(epochs, val_acc, label='Validation Accuracy', linewidth=2)
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Courbes d\'Accuracy')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

**Diagnostic :**
- **Overfitting** : Train loss continue de baisser, Val loss remonte
- **Underfitting** : Train et Val loss élevées et stagnent
- **Good fit** : Train et Val loss convergent vers une valeur basse

---

### 4.3 Exemples de Prédictions (Good/Bad Cases)

**Pourquoi l'utiliser ?**
- **Rend le modèle tangible** pour la direction et les utilisateurs
- Montre les forces et faiblesses du modèle

**Structure pour PowerPoint :**

**Slide "Exemples de Succès" :**
- 3-4 exemples bien classés avec score de confiance élevé
- Légende : Email, vraie classe, prédiction, confiance

**Slide "Cas d'Erreurs" :**
- 3-4 exemples mal classés
- Analyse : Pourquoi le modèle s'est trompé ?

```python
# Exemple pour classification d'emails
examples = [
    {"text": "URGENT : Problème critique client VIP", 
     "true": "Urgent", "pred": "Urgent", "conf": 0.95, "correct": True},
    {"text": "Réunion hebdomadaire lundi 10h", 
     "true": "Normal", "pred": "Normal", "conf": 0.88, "correct": True},
    {"text": "Merci pour votre réponse rapide", 
     "true": "Normal", "pred": "Important", "conf": 0.62, "correct": False},
]

# Visualisation pour PowerPoint (tableau ou cards)
for ex in examples:
    color = 'green' if ex['correct'] else 'red'
    print(f"[{color.upper()}] {ex['text'][:50]}...")
    print(f"  Vraie classe: {ex['true']} | Prédiction: {ex['pred']} | Confiance: {ex['conf']:.0%}\n")
```

---

### 4.4 Feature Importance (Importance des Variables)

**Pourquoi l'utiliser ?**
- Explique **quelles variables** influencent le modèle
- Utile pour les modèles type Random Forest, XGBoost
- Crédibilise le modèle auprès des experts métier

```python
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

# Exemple avec Random Forest
feature_names = ['Longueur email', 'Nb mots-clés urgent', 'Heure réception', 
                 'Expéditeur VIP', 'Pièces jointes']
importances = [0.35, 0.28, 0.18, 0.12, 0.07]

plt.figure(figsize=(10, 6))
plt.barh(feature_names, importances, color='steelblue')
plt.xlabel('Importance')
plt.title('Importance des Variables dans le Modèle')
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.show()
```

**Pour PowerPoint :**
- Top 10 features les plus importantes
- Aide à valider que le modèle utilise des variables pertinentes

---

### 4.5 Calibration Plot (Courbe de Calibration)

**Pourquoi l'utiliser ?**
- Vérifie si les **probabilités prédites** sont fiables
- Exemple : Si le modèle prédit 70% de confiance, est-ce qu'il a raison 70% du temps ?

```python
from sklearn.calibration import calibration_curve

prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=10)

plt.figure(figsize=(10, 8))
plt.plot(prob_pred, prob_true, marker='o', linewidth=2, label='Modèle')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Parfaitement calibré')
plt.xlabel('Probabilité prédite')
plt.ylabel('Fraction de positifs')
plt.title('Courbe de Calibration')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

**Interprétation :**
- Courbe proche de la diagonale = modèle bien calibré
- Courbe au-dessus = modèle sous-confiant
- Courbe en-dessous = modèle sur-confiant

---

## 5. Métriques Business et ROI

### 5.1 Temps Économisé

**Formule :**

$$\text{Temps économisé} = (\text{Temps manuel} - \text{Temps avec IA}) \times \text{Volume traité}$$

**Exemple (Classification d'emails) :**
- Temps manuel : 3 min/email
- Temps avec IA : 30 sec/email (vérification)
- Volume : 1000 emails/mois
- **Temps économisé = (3 - 0.5) × 1000 = 2500 min/mois = 42h/mois**

```python
# Calculateur ROI
def calculate_time_saved(manual_time_per_item, ai_time_per_item, 
                         volume_per_month, hourly_wage):
    time_saved_per_item = manual_time_per_item - ai_time_per_item
    total_time_saved_hours = (time_saved_per_item * volume_per_month) / 60
    cost_saved = total_time_saved_hours * hourly_wage
    
    return {
        'time_saved_hours': total_time_saved_hours,
        'cost_saved_monthly': cost_saved,
        'cost_saved_yearly': cost_saved * 12
    }

results = calculate_time_saved(
    manual_time_per_item=3,      # 3 min
    ai_time_per_item=0.5,        # 30 sec
    volume_per_month=1000,       # 1000 emails/mois
    hourly_wage=40               # 40€/h
)

print(f"Temps économisé : {results['time_saved_hours']:.0f} heures/mois")
print(f"Coût économisé : {results['cost_saved_yearly']:,.0f}€/an")
```

---

### 5.2 Taux d'Automatisation

**Formule :**

$$\text{Taux d'automatisation} = \frac{\text{Nombre de cas traités automatiquement}}{\text{Nombre total de cas}} \times 100$$

**Exemple :**
- 1000 emails traités
- 850 traités automatiquement (confiance > 0.8)
- 150 nécessitent une vérification humaine
- **Taux d'automatisation = 850/1000 = 85%**

**Pour PowerPoint :**
- Graphique en donut montrant le taux d'automatisation
- Objectif : 80%+ pour justifier l'investissement IA

---

### 5.3 Réduction du Taux d'Erreur

**Avant/Après avec IA :**

| Métrique                | Avant IA (Manuel) | Avec IA | Amélioration |
|-------------------------|-------------------|---------|--------------|
| Emails mal classés      | 8%                | 3%      | -62.5%       |
| Temps de traitement     | 3h/jour           | 45min   | -75%         |
| Réclamations clients    | 15/mois           | 4/mois  | -73%         |

**Pour PowerPoint :**
- Tableau comparatif avant/après
- Focus sur les KPIs métier (réclamations, temps, erreurs)

---

## 6. Communication de l'Incertitude

### 6.1 Intervalles de Confiance

**Pourquoi l'utiliser ?**
- Quantifie l'**incertitude** des prédictions
- Critique pour les décisions à fort impact

**Exemple (Régression) :**

```python
from sklearn.ensemble import GradientBoostingRegressor
import numpy as np

# Entraînement avec quantile regression
model_lower = GradientBoostingRegressor(loss='quantile', alpha=0.05)
model_upper = GradientBoostingRegressor(loss='quantile', alpha=0.95)

# X_train, y_train : données d'entraînement
# Prédictions avec intervalles de confiance à 90%
# y_pred_lower = model_lower.predict(X_test)
# y_pred_upper = model_upper.predict(X_test)

# Visualisation
plt.figure(figsize=(12, 6))
x = np.arange(len(y_true))
plt.plot(x, y_true, 'o', label='Vraies valeurs', alpha=0.5)
plt.plot(x, y_pred, '-', label='Prédictions', linewidth=2)
# plt.fill_between(x, y_pred_lower, y_pred_upper, alpha=0.3, 
#                  label='Intervalle de confiance 90%')
plt.xlabel('Échantillon')
plt.ylabel('Valeur')
plt.title('Prédictions avec Intervalles de Confiance')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

---

### 6.2 Scores de Confiance et Seuils de Décision

**Stratégie pour la production :**

| Score de confiance | Action                              |
|--------------------|-------------------------------------|
| > 0.9              | Traitement automatique              |
| 0.7 - 0.9          | Traitement automatique + alerte     |
| 0.5 - 0.7          | Revue humaine obligatoire           |
| < 0.5              | Rejet ou escalade                   |

**Pour PowerPoint :**
- Présente cette stratégie de seuils
- Montre la distribution des scores pour estimer le volume de revue humaine

---

## 7. Présentation des Erreurs (Failure Cases)

### 7.1 Analyse des Types d'Erreurs

**Classification des erreurs :**

1. **Erreurs acceptables** : Cas limites, ambiguïtés
2. **Erreurs critiques** : Mauvaise classification avec forte confiance
3. **Erreurs systématiques** : Pattern récurrent (ex: confusion systématique entre 2 classes)

**Pour PowerPoint :**

**Slide "Analyse des Erreurs" :**
- Camembert : Répartition des types d'erreurs
- Liste des top 3 causes d'erreurs
- Actions correctives envisagées

```python
# Analyse des erreurs par catégorie
error_categories = {
    'Emails ambigus (frontière floue)': 45,
    'Manque de contexte': 25,
    'Nouveaux types non vus en train': 15,
    'Erreurs aléatoires': 15
}

plt.figure(figsize=(10, 8))
plt.pie(error_categories.values(), labels=error_categories.keys(), 
        autopct='%1.1f%%', startangle=90, colors=['#ff9999', '#66b3ff', '#99ff99', '#ffcc99'])
plt.title('Répartition des Types d\'Erreurs', fontsize=14)
plt.show()
```

---

### 7.2 Exemples d'Erreurs Annotés

**Structure :**

Pour chaque erreur présentée :
1. **Texte de l'email** (ou image pour YOLO)
2. **Vraie classe** vs **Prédiction**
3. **Score de confiance**
4. **Explication** : Pourquoi le modèle s'est trompé ?
5. **Action** : Comment corriger (retraining, ajout de features, etc.) ?

**Exemple (Classification email) :**

      Email : "Bonjour, pouvez-vous me rappeler demain ?"
      Vraie classe : Important
      Prédiction : Normal (confiance 0.68)
      Explication : Manque de mots-clés "urgent", phrase courte, pas de deadline précise.
      Action : Ajouter feature "demande d'action" détectée par NER.


---

## 8. Comparaison de Modèles

### 8.1 Tableau Comparatif Multi-Critères

**Pour PowerPoint :**

| Modèle           | Accuracy | F1-Score | Temps inférence | Taille modèle | ROI (€/an) |
|------------------|----------|----------|-----------------|---------------|------------|
| Baseline (règles)| 0.72     | 0.65     | 5 ms            | -             | 0          |
| Random Forest    | 0.84     | 0.81     | 15 ms           | 50 MB         | 25 000     |
| BERT fine-tuned  | 0.91     | 0.89     | 80 ms           | 440 MB        | 38 000     |
| **GPT-4 (API)**  | **0.93** | **0.91** | 1200 ms         | -             | 15 000     |

**Recommandation** : BERT fine-tuned (meilleur compromis perf/coût/latence)

---

### 8.2 Graphiques de Comparaison

**Radar Chart (Comparaison multi-dimensionnelle) :**

{{{python
import matplotlib.pyplot as plt
import numpy as np

categories = ['Accuracy', 'Vitesse', 'Coût', 'Maintenabilité', 'Explicabilité']
model1 = [0.84, 0.9, 0.95, 0.85, 0.9]  # Random Forest
model2 = [0.91, 0.5, 0.7, 0.6, 0.4]    # BERT

angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
model1 += model1[:1]
model2 += model2[:1]
angles += angles[:1]

fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
ax.plot(angles, model1, 'o-', linewidth=2, label='Random Forest', color='blue')
ax.fill(angles, model1, alpha=0.25, color='blue')
ax.plot(angles, model2, 'o-', linewidth=2, label='BERT', color='red')
ax.fill(angles, model2, alpha=0.25, color='red')
ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, size=12)
ax.set_ylim(0, 1)
ax.set_title('Comparaison Multi-Critères des Modèles', size=16, pad=20)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
ax.grid(True)
plt.tight_layout()
plt.show()
}}}

---

### 8.3 Trade-off Précision vs Coût/Vitesse

**Scatter Plot :**

{{{python
models = ['Baseline', 'Random Forest', 'BERT', 'GPT-4 API']
f1_scores = [0.65, 0.81, 0.89, 0.91]
costs = [0, 100, 500, 2000]  # Coût annuel d'inférence (€)

plt.figure(figsize=(12, 8))
colors = ['gray', 'green', 'blue', 'red']
sizes = [100, 200, 200, 200]

for i, model in enumerate(models):
    plt.scatter(costs[i], f1_scores[i], s=sizes[i], alpha=0.6, 
                c=colors[i], label=model)
    plt.annotate(model, (costs[i], f1_scores[i]), 
                 fontsize=12, ha='center', xytext=(0, 15),
                 textcoords='offset points')

plt.xlabel('Coût annuel d\'inférence (€)', fontsize=14)
plt.ylabel('F1-Score', fontsize=14)
plt.title('Trade-off Performance vs Coût', fontsize=16)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=12)
plt.tight_layout()
plt.show()
}}}

---

## 9. Adaptation selon l'Audience

### 9.1 Pour la Direction (Non-Experts Techniques)

**À privilégier :**
- ✅ **Métriques business** : Temps économisé, ROI, taux d'automatisation
- ✅ **Graphiques simples** : Barres, camemberts, avant/après
- ✅ **Exemples concrets** : "Le modèle détecte 9 emails urgents sur 10"
- ✅ **Comparaison avec baseline** : Montrer l'amélioration vs situation actuelle
- ✅ **Visualisations de résultats** : Matrice de confusion avec couleurs

**À éviter :**
- ❌ Formules mathématiques complexes
- ❌ Jargon technique (overfitting, hyperparamètres, etc.)
- ❌ Trop de métriques (s'en tenir à 2-3 max)

**Exemple de slide pour direction :**

**Titre : "Résultats du Modèle de Classification d'Emails"**
- 📊 Accuracy : 91% (vs 72% avec règles manuelles)
- ⏱️ Temps économisé : 42h/mois = 50 000€/an
- 🤖 Taux d'automatisation : 85%
- ⭐ Satisfaction utilisateurs : +35%

---

### 9.2 Pour les Utilisateurs Métier

**À privilégier :**
- ✅ **Exemples réels** : Montrer des emails/images qu'ils reconnaissent
- ✅ **Cas d'usage** : "Comment le modèle m'aide au quotidien ?"
- ✅ **Transparence sur les limites** : "Quand le modèle a besoin d'aide humaine"
- ✅ **Feedback loop** : "Comment signaler une erreur ?"

**À éviter :**
- ❌ Métriques techniques (AUC, F1, mAP)
- ❌ Comparaison de modèles (peu pertinent pour eux)

---

### 9.3 Pour les Experts IA (Chef, Pairs)

**À privilégier :**
- ✅ **Métriques techniques complètes** : Precision, Recall, F1, AUC, mAP
- ✅ **Courbes détaillées** : ROC, Precision-Recall, Learning curves
- ✅ **Détails d'architecture** : Choix du modèle, hyperparamètres, augmentation de données
- ✅ **Analyse approfondie des erreurs** : Failure modes, confusion entre classes
- ✅ **Benchmarks** : Comparaison avec SOTA (state-of-the-art)

**Exemple de slide pour experts :**

**Titre : "Performances du Modèle BERT Fine-tuned"**
- Macro-F1 : 0.89 (vs 0.81 Random Forest)
- AUC-ROC : 0.94
- mAP@0.5 : 0.87 (pour détection YOLO)
- Confusion principale : Classe "Important" vs "Urgent" (15% des erreurs)
- Hyperparamètres : learning_rate=2e-5, batch_size=16, epochs=10

---

## 10. Structure Type de Présentation PowerPoint

### 10.1 Slide 1 : Contexte et Objectif

**Contenu :**
- Problème métier à résoudre
- Objectif du modèle
- Bénéfices attendus

      **Exemple :**
      
      Titre : Classification Automatique des Emails Clients
      Problème :
      
          1000+ emails/jour reçus
          Temps de tri manuel : 3 min/email
          Emails urgents parfois ratés
      
      Objectif :
      Automatiser la classification en 3 catégories (Normal, Important, Urgent)
      Bénéfices :
      
          Réduction du temps de traitement de 75%
          Priorisation automatique des emails urgents
          Amélioration satisfaction client

---

### 10.2 Slide 2 : Données et Méthode

**Contenu :**
- Description du dataset
- Répartition des classes
- Approche choisie (modèle)

**Exemple :**

      Dataset :
      
          15 000 emails étiquetés manuellement
          Période : Jan 2023 - Déc 2024
          Répartition : Normal (60%), Important (30%), Urgent (10%)
      
      Approche :
      
          Modèle : BERT fine-tuned (Transformer)
          Split : 70% train, 15% validation, 15% test
          Augmentation de données pour classe "Urgent" (SMOTE)

---

### 10.3 Slide 3 : Résultats Principaux

**Contenu :**
- Métriques clés (2-3 max pour la direction)
- Comparaison avec baseline
- Graphique de performance

**Exemple (Direction) :**

Jackie
Jackie
17:26
Completed in 14 min 59 sec
Parfait 
@Clément o
 ! Je vais te créer un guide complet et structuré. Vu l'ampleur du sujet, je vais le découper en sections claires pour que tu puisses naviguer facilement.
🚀 Je te prépare ton "Guide ultime de présentation des résultats de modèles IA"
Donne-moi quelques minutes pour structurer tout ça proprement avec :

    Les métriques essentielles (math + cookbook)
    Les visualisations clés pour PowerPoint
    Les adaptations selon ton audience (direction vs experts IA)
    Les cas spécifiques : classification d'emails et détection YOLO
    Métriques business, incertitude, failure cases, comparaisons de modèles

Je reviens avec le guide complet ! ⏳

# Guide Complet : Présentation des Résultats de Modèles IA

**Guide de référence pour présenter des résultats de modèles IA de manière claire, professionnelle et pertinente**

*Auteur : Guide pour Clément O.*  
*Contexte : Présentations PowerPoint pour direction, utilisateurs et experts IA*  
*Domaines : NLP (Transformers/BERT), Computer Vision (YOLO), Classification, Régression*

---

## Table des matières

1. [Métriques de Classification](#1-métriques-de-classification)
2. [Métriques de Détection d'Objets (YOLO)](#2-métriques-de-détection-dobjets-yolo)
3. [Métriques de Régression](#3-métriques-de-régression)
4. [Visualisations Essentielles pour PowerPoint](#4-visualisations-essentielles-pour-powerpoint)
5. [Métriques Business et ROI](#5-métriques-business-et-roi)
6. [Communication de l'Incertitude](#6-communication-de-lincertitude)
7. [Présentation des Erreurs (Failure Cases)](#7-présentation-des-erreurs-failure-cases)
8. [Comparaison de Modèles](#8-comparaison-de-modèles)
9. [Adaptation selon l'Audience](#9-adaptation-selon-laudience)
10. [Structure Type de Présentation PowerPoint](#10-structure-type-de-présentation-powerpoint)

---

## 1. Métriques de Classification

*Utilisées pour : Classification d'emails, classification d'images, détection binaire/multi-classes*

### 1.1 Matrice de Confusion

**Définition :**  
Tableau croisant les prédictions du modèle avec les vraies valeurs.

**Structure (cas binaire) :**

|                     | **Prédit Positif** | **Prédit Négatif** |
|---------------------|--------------------|--------------------|
| **Réel Positif**    | TP (Vrais Positifs)| FN (Faux Négatifs) |
| **Réel Négatif**    | FP (Faux Positifs) | TN (Vrais Négatifs)|

**Pourquoi l'utiliser ?**
- Vision instantanée des types d'erreurs
- Identification des classes confondues (en multi-classes)
- Essentiel pour expliquer les performances à la direction

**Exemple de code :**

{{{python
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# y_true : vraies étiquettes, y_pred : prédictions
y_true = [0, 1, 0, 1, 1, 0, 1, 0]
y_pred = [0, 1, 0, 0, 1, 0, 1, 1]

cm = confusion_matrix(y_true, y_pred)

# Visualisation
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Négatif', 'Positif'],
            yticklabels=['Négatif', 'Positif'])
plt.ylabel('Vraie classe')
plt.xlabel('Classe prédite')
plt.title('Matrice de Confusion')
plt.show()
}}}

**Pour PowerPoint :**
- Utilise une heatmap avec annotations
- Ajoute des pourcentages en plus des valeurs brutes
- Version normalisée pour comparer entre datasets

---

### 1.2 Accuracy (Exactitude)

**Formule mathématique :**  

$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$

**Interprétation :**  
Proportion de prédictions correctes parmi toutes les prédictions.

**Pourquoi l'utiliser ?**
- Métrique la plus simple et intuitive pour la direction
- Bonne vue d'ensemble si les classes sont équilibrées

**⚠️ Piège à éviter :**
- **Classes déséquilibrées** : Si 95% des emails sont "non-urgents", un modèle prédisant toujours "non-urgent" aura 95% d'accuracy mais sera inutile !

**Quand l'utiliser ?**
- ✅ Classes équilibrées (ratio 40-60 minimum)
- ✅ Présentation rapide à la direction (mais toujours accompagner d'autres métriques)
- ❌ Classes très déséquilibrées (fraude, anomalies, emails urgents rares)

{{{python
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_true, y_pred)
print(f"Accuracy : {accuracy:.2%}")  # 75.00%
}}}

---

### 1.3 Precision (Précision)

**Formule mathématique :**  

$$\text{Precision} = \frac{TP}{TP + FP}$$

**Interprétation :**  
Parmi les emails classés "urgents" par le modèle, quelle proportion l'est réellement ?

**Pourquoi l'utiliser ?**
- Quand les **Faux Positifs sont coûteux**
- Exemple : Classification "urgent" → si beaucoup de FP, on perd du temps à traiter des emails non-urgents

**Cas d'usage typique :**
- Filtrage spam (on ne veut pas bloquer des emails légitimes)
- Détection de produits défectueux (coût d'inspection inutile)

{{{python
from sklearn.metrics import precision_score

precision = precision_score(y_true, y_pred)
print(f"Precision : {precision:.2%}")
}}}

---

### 1.4 Recall (Rappel / Sensibilité)

**Formule mathématique :**  

$$\text{Recall} = \frac{TP}{TP + FN}$$

**Interprétation :**  
Parmi tous les emails réellement urgents, quelle proportion a été détectée par le modèle ?

**Pourquoi l'utiliser ?**
- Quand les **Faux Négatifs sont coûteux**
- Exemple : Classification "urgent" → si beaucoup de FN, on rate des emails urgents !

**Cas d'usage typique :**
- Détection de maladies (on ne veut pas rater un malade)
- Détection d'anomalies critiques
- Emails urgents clients VIP

{{{python
from sklearn.metrics import recall_score

recall = recall_score(y_true, y_pred)
print(f"Recall : {recall:.2%}")
}}}

---

### 1.5 F1-Score

**Formule mathématique :**  

$\$F1 = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

**Interprétation :**  
Moyenne harmonique entre Precision et Recall. Équilibre entre les deux.

**Pourquoi l'utiliser ?**
- Quand Precision ET Recall sont importants
- **Métrique unique** pour comparer des modèles
- Gère mieux les classes déséquilibrées que l'Accuracy

**⚠️ Attention :**
- F1 = 0.80 ne veut pas dire 80% de performance globale
- Un F1 élevé n'est bon que si Precision ET Recall sont équilibrés

{{{python
from sklearn.metrics import f1_score

f1 = f1_score(y_true, y_pred)
print(f"F1-Score : {f1:.2%}")
}}}

**Trade-off Precision vs Recall :**

Imaginons un modèle de classification d'emails urgents :
- **Seuil bas (0.3)** : Le modèle prédit "urgent" facilement → Recall élevé mais Precision basse (beaucoup de FP)
- **Seuil élevé (0.8)** : Le modèle est prudent → Precision élevée mais Recall bas (on rate des urgents)

{{{python
from sklearn.metrics import precision_recall_curve

# Obtenir les probabilités prédites
y_proba = [0.1, 0.9, 0.2, 0.4, 0.85, 0.15, 0.95, 0.6]

precision, recall, thresholds = precision_recall_curve(y_true, y_proba)

plt.figure(figsize=(10, 6))
plt.plot(thresholds, precision[:-1], label='Precision')
plt.plot(thresholds, recall[:-1], label='Recall')
plt.xlabel('Seuil de décision')
plt.ylabel('Score')
plt.title('Trade-off Precision vs Recall selon le seuil')
plt.legend()
plt.grid(True)
plt.show()
}}}

---

### 1.6 Specificité (Specificity)

**Formule mathématique :**  

$$\text{Specificity} = \frac{TN}{TN + FP}$$

**Interprétation :**  
Parmi tous les emails NON-urgents, quelle proportion a été correctement identifiée ?

**Pourquoi l'utiliser ?**
- Complémentaire au Recall
- Important en médical : capacité à identifier les personnes saines

---

### 1.7 Balanced Accuracy

**Formule mathématique :**  

$$\text{Balanced Accuracy} = \frac{\text{Recall}_{\text{classe 1}} + \text{Specificity}}{2}$$

**Pourquoi l'utiliser ?**
- **Classes déséquilibrées** : Meilleure alternative à l'Accuracy classique
- Donne le même poids à chaque classe

**Exemple :**

{{{python
from sklearn.metrics import balanced_accuracy_score

balanced_acc = balanced_accuracy_score(y_true, y_pred)
print(f"Balanced Accuracy : {balanced_acc:.2%}")
}}}

---

### 1.8 Courbe ROC et AUC

**ROC (Receiver Operating Characteristic) :**  
Courbe montrant le trade-off entre Recall (Taux de Vrais Positifs) et Taux de Faux Positifs à différents seuils.

**AUC (Area Under Curve) :**  
Aire sous la courbe ROC. Varie entre 0 et 1.

**Interprétation de l'AUC :**
- **AUC = 0.5** : Modèle aléatoire (inutile)
- **AUC = 0.7-0.8** : Acceptable
- **AUC = 0.8-0.9** : Bon modèle
- **AUC > 0.9** : Excellent modèle
- **AUC = 1.0** : Modèle parfait (ou overfitting !)

**Pourquoi l'utiliser ?**
- Évalue la capacité du modèle à discriminer entre classes
- **Indépendant du seuil** de décision choisi
- Très utilisé en recherche et facile à présenter

{{{python
from sklearn.metrics import roc_curve, auc

fpr, tpr, thresholds = roc_curve(y_true, y_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(10, 8))
plt.plot(fpr, tpr, color='darkorange', lw=2, 
         label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
         label='Aléatoire (AUC = 0.50)')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Taux de Faux Positifs (1 - Specificité)')
plt.ylabel('Taux de Vrais Positifs (Recall)')
plt.title('Courbe ROC')
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.show()
}}}

**Pour PowerPoint :**
- Ajoute la ligne de référence (diagonale) pour montrer le niveau aléatoire
- Indique clairement la valeur de l'AUC
- Si comparaison de modèles : superpose plusieurs courbes ROC

---

### 1.9 Courbe Précision-Recall

**Pourquoi l'utiliser plutôt que ROC ?**
- **Classes très déséquilibrées** : Plus informative que ROC
- Exemple : détection d'anomalies (1% de positifs), fraude, emails urgents rares

**Différence ROC vs Precision-Recall :**

{{{python
from sklearn.metrics import precision_recall_curve, average_precision_score

precision, recall, _ = precision_recall_curve(y_true, y_proba)
avg_precision = average_precision_score(y_true, y_proba)

plt.figure(figsize=(10, 8))
plt.plot(recall, precision, color='blue', lw=2,
         label=f'PR curve (AP = {avg_precision:.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Courbe Précision-Recall')
plt.legend(loc="upper right")
plt.grid(True, alpha=0.3)
plt.show()
}}}

**Cookbook : ROC ou PR ?**
- ✅ **ROC** : Classes équilibrées, présentation à des non-experts
- ✅ **Precision-Recall** : Classes déséquilibrées, focus sur la classe minoritaire

---

### 1.10 Métriques Multi-Classes

**Macro-Average :**  
Moyenne simple des métriques de chaque classe (toutes les classes ont le même poids).

$$\text{Macro-F1} = \frac{1}{n} \sum_{i=1}^{n} F1_i$$

**Micro-Average :**  
Agrège les TP, FP, FN de toutes les classes puis calcule la métrique globale.

**Weighted-Average :**  
Moyenne pondérée par le nombre d'échantillons de chaque classe.

**Quand utiliser quoi ?**
- **Macro** : Toutes les classes ont la même importance (même si déséquilibrées)
- **Micro** : Focus sur la performance globale
- **Weighted** : Importance proportionnelle au nombre d'échantillons

{{{python
from sklearn.metrics import classification_report

# Exemple multi-classes : urgence email (0=normal, 1=important, 2=urgent)
y_true_multi = [0, 1, 2, 0, 1, 2, 0, 1]
y_pred_multi = [0, 1, 1, 0, 1, 2, 0, 2]

print(classification_report(y_true_multi, y_pred_multi, 
                           target_names=['Normal', 'Important', 'Urgent']))
}}}

---

## 2. Métriques de Détection d'Objets (YOLO)

*Utilisées pour : Détection automatisée par computer vision, segmentation, localisation*

### 2.1 IoU (Intersection over Union)

**Formule mathématique :**  

$$\text{IoU} = \frac{\text{Aire de l'intersection}}{\text{Aire de l'union}} = \frac{A \cap B}{A \cup B}$$

**Interprétation :**  
Mesure le chevauchement entre la bounding box prédite et la bounding box réelle.

**Valeurs typiques :**
- **IoU ≥ 0.5** : Détection considérée comme correcte (seuil standard)
- **IoU ≥ 0.75** : Détection de haute qualité
- **IoU < 0.5** : Faux Positif

**Pourquoi l'utiliser ?**
- Détermine si une détection est un TP ou un FP
- Essentiel pour calculer la Precision et le Recall en détection d'objets

**Visualisation :**

{{{python
def calculate_iou(box1, box2):
    """
    box format: [x1, y1, x2, y2] (top-left et bottom-right)
    """
    # Coordonnées de l'intersection
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])
    
    # Aire de l'intersection
    inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)
    
    # Aires des deux boxes
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    # Aire de l'union
    union_area = box1_area + box2_area - inter_area
    
    # IoU
    iou = inter_area / union_area if union_area > 0 else 0
    return iou

# Exemple
box_ground_truth = [50, 50, 150, 150]  # Vraie boîte
box_predicted = [60, 60, 160, 140]     # Prédiction

iou = calculate_iou(box_ground_truth, box_predicted)
print(f"IoU : {iou:.2f}")  # 0.68
}}}

---

### 2.2 mAP (mean Average Precision)

**Définition :**  
Métrique principale pour évaluer les modèles de détection d'objets.

**Calcul en étapes :**

1. **Pour chaque classe** :
   - Ordonne toutes les détections par score de confiance décroissant
   - Calcule Precision et Recall à chaque seuil
   - Calcule l'Average Precision (AP) = aire sous la courbe Précision-Recall
   
2. **mAP** = Moyenne des AP de toutes les classes

**Variantes :**
- **mAP@0.5** : IoU seuil = 0.5 (standard PASCAL VOC)
- **mAP@0.75** : IoU seuil = 0.75 (plus strict)
- **mAP@[0.5:0.95]** : Moyenne sur plusieurs seuils IoU (standard COCO)

**Pourquoi l'utiliser ?**
- **Métrique standard** pour benchmarker les modèles de détection
- Prend en compte Precision, Recall ET la localisation (via IoU)
- Permet de comparer différentes architectures (YOLOv5 vs YOLOv8 vs Faster R-CNN)

**Interprétation :**
- **mAP@0.5 = 0.60** : Performance acceptable pour des applications réelles
- **mAP@0.5 = 0.80+** : Très bon modèle
- **mAP@0.5:0.95** : Plus exigeant (souvent 20-30% inférieur au mAP@0.5)

{{{python
# Avec YOLOv8 (ultralytics)
from ultralytics import YOLO

model = YOLO('yolov8n.pt')
results = model.val(data='coco.yaml')  # Validation sur dataset

print(f"mAP@0.5 : {results.box.map50:.3f}")
print(f"mAP@0.5:0.95 : {results.box.map:.3f}")
}}}

**Pour PowerPoint :**
- Tableau comparatif des mAP entre modèles
- Graphique en barres par classe (montre les classes bien/mal détectées)

---

### 2.3 Precision et Recall en détection

**Adaptation pour détection d'objets :**

- **True Positive (TP)** : Détection avec IoU ≥ seuil et bonne classe
- **False Positive (FP)** : Détection avec IoU < seuil OU mauvaise classe OU détection en doublon
- **False Negative (FN)** : Objet non détecté

**Calcul identique à la classification :**

$$\text{Precision} = \frac{TP}{TP + FP}$$

$$\text{Recall} = \frac{TP}{TP + FN}$$

**Exemple concret (détection de défauts) :**
- 100 défauts réels dans les images
- Modèle détecte 90 défauts (dont 85 corrects et 5 faux)
- Recall = 85/100 = 0.85 (on a trouvé 85% des défauts)
- Precision = 85/90 = 0.94 (94% de nos détections sont correctes)

---

### 2.4 Confusion entre Classes (Détection)

**Pourquoi l'utiliser ?**
- Identifier les paires de classes confondues par le modèle
- Exemple : YOLO confond "voiture" et "camion"

**Visualisation :**

{{{python
import numpy as np
import seaborn as sns

# Matrice de confusion pour détection multi-classes
# Exemple : détection de 3 types de défauts
confusion_matrix_detection = np.array([
    [120, 10, 5],    # Défaut type A
    [8, 95, 12],     # Défaut type B
    [3, 15, 87]      # Défaut type C
])

plt.figure(figsize=(10, 8))
sns.heatmap(confusion_matrix_detection, annot=True, fmt='d', cmap='YlOrRd',
            xticklabels=['Défaut A', 'Défaut B', 'Défaut C'],
            yticklabels=['Défaut A', 'Défaut B', 'Défaut C'])
plt.ylabel('Vraie classe')
plt.xlabel('Classe prédite')
plt.title('Matrice de Confusion - Détection YOLO')
plt.show()
}}}

---

### 2.5 Vitesse d'Inférence (FPS, Latence)

**Métriques complémentaires critiques pour la production :**

- **FPS (Frames Per Second)** : Nombre d'images traitées par seconde
- **Latence** : Temps de traitement d'une image (ms)

**Pourquoi l'utiliser ?**
- Contraintes temps réel (vidéo surveillance, ligne de production)
- Trade-off précision vs vitesse

**Exemple :**

| Modèle      | mAP@0.5 | FPS (GPU) | Latence (ms) |
|-------------|---------|-----------|--------------|
| YOLOv8n     | 0.52    | 380       | 2.6          |
| YOLOv8s     | 0.61    | 285       | 3.5          |
| YOLOv8m     | 0.67    | 165       | 6.0          |
| YOLOv8l     | 0.70    | 110       | 9.1          |

**Pour PowerPoint :**
- Graphique scatter : mAP (axe Y) vs Latence (axe X)
- Montre le meilleur compromis précision/vitesse

{{{python
import matplotlib.pyplot as plt

models = ['YOLOv8n', 'YOLOv8s', 'YOLOv8m', 'YOLOv8l']
map_scores = [0.52, 0.61, 0.67, 0.70]
latency = [2.6, 3.5, 6.0, 9.1]

plt.figure(figsize=(10, 6))
plt.scatter(latency, map_scores, s=200, alpha=0.6)
for i, model in enumerate(models):
    plt.annotate(model, (latency[i], map_scores[i]), 
                 fontsize=12, ha='right')
plt.xlabel('Latence (ms)', fontsize=12)
plt.ylabel('mAP@0.5', fontsize=12)
plt.title('Trade-off Précision vs Vitesse', fontsize=14)
plt.grid(True, alpha=0.3)
plt.show()
}}}

---

## 3. Métriques de Régression

*Utilisées pour : Prédiction de valeurs continues (prix, délais, quantités)*

### 3.1 MAE (Mean Absolute Error)

**Formule mathématique :**  

$$\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$$

**Interprétation :**  
Erreur moyenne absolue, dans les mêmes unités que la variable cible.

**Pourquoi l'utiliser ?**
- **Interprétation directe** : "En moyenne, le modèle se trompe de X unités"
- Robuste aux outliers (pas de carré)
- Facile à expliquer à la direction

**Exemple :**
- Prédiction de temps de traitement d'emails : MAE = 5 min
- → Le modèle se trompe en moyenne de 5 minutes

{{{python
from sklearn.metrics import mean_absolute_error
import numpy as np

y_true = np.array([100, 120, 90, 110, 105])
y_pred = np.array([98, 125, 88, 105, 110])

mae = mean_absolute_error(y_true, y_pred)
print(f"MAE : {mae:.2f}")  # 3.80
}}}

---

### 3.2 RMSE (Root Mean Squared Error)

**Formule mathématique :**  

$$\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}$$

**Interprétation :**  
Racine carrée de la moyenne des erreurs au carré, dans les mêmes unités que la variable cible.

**Pourquoi l'utiliser ?**
- Pénalise davantage les **grandes erreurs** (grâce au carré)
- Utile quand les grandes erreurs sont particulièrement coûteuses
- Standard dans la communauté ML

**Différence MAE vs RMSE :**

{{{python
y_true = np.array([100, 100, 100, 100, 100])
y_pred1 = np.array([105, 105, 105, 105, 105])  # Erreurs constantes
y_pred2 = np.array([100, 100, 100, 100, 125])  # Une grosse erreur

mae1 = mean_absolute_error(y_true, y_pred1)
rmse1 = np.sqrt(np.mean((y_true - y_pred1)**2))
print(f"Cas 1 - MAE: {mae1:.2f}, RMSE: {rmse1:.2f}")  # MAE: 5.00, RMSE: 5.00

mae2 = mean_absolute_error(y_true, y_pred2)
rmse2 = np.sqrt(np.mean((y_true - y_pred2)**2))
print(f"Cas 2 - MAE: {mae2:.2f}, RMSE: {rmse2:.2f}")  # MAE: 5.00, RMSE: 11.18

# Même MAE mais RMSE plus élevé car une grosse erreur
}}}

**Quand préférer RMSE à MAE ?**
- Quand les grandes erreurs sont **inacceptables**
- Quand on veut être **conservateur**

**Quand préférer MAE à RMSE ?**
- Quand on veut une métrique **robuste aux outliers**
- Quand toutes les erreurs ont le **même coût**

---

### 3.3 R² (Coefficient de Détermination)

**Formule mathématique :**  

$\$R^2 = 1 - \frac{\sum_{i=1}^{n} (y_i - \hat{y}_i)^2}{\sum_{i=1}^{n} (y_i - \bar{y})^2}$$

Où $$\bar{y}$$ est la moyenne des valeurs réelles.

**Interprétation :**  
Proportion de la variance expliquée par le modèle.

- **R² = 1** : Modèle parfait
- **R² = 0.8** : Le modèle explique 80% de la variance
- **R² = 0** : Modèle aussi bon qu'une prédiction par la moyenne
- **R² < 0** : Modèle pire qu'une prédiction par la moyenne !

**Pourquoi l'utiliser ?**
- **Sans unité** : Permet de comparer des modèles sur différents problèmes
- Très utilisé en régression linéaire et statistiques

**⚠️ Attention :**
- R² peut être trompeur si le modèle est complexe (overfitting)
- Préférer le **R² ajusté** pour comparer des modèles avec différents nombres de variables

{{{python
from sklearn.metrics import r2_score

r2 = r2_score(y_true, y_pred)
print(f"R² : {r2:.3f}")
}}}

---

### 3.4 MAPE (Mean Absolute Percentage Error)

**Formule mathématique :**  

$$\text{MAPE} = \frac{100}{n} \sum_{i=1}^{n} \left| \frac{y_i - \hat{y}_i}{y_i} \right|$$

**Interprétation :**  
Erreur moyenne en pourcentage.

**Pourquoi l'utiliser ?**
- **Interprétation business** : "Le modèle se trompe en moyenne de X%"
- Utile pour communiquer avec la direction

**⚠️ Attention :**
- Sensible aux **valeurs proches de zéro** (division par $$y_i$$)
- Asymétrique : sous-estimation pénalisée plus que sur-estimation

{{{python
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

mape = mean_absolute_percentage_error(y_true, y_pred)
print(f"MAPE : {mape:.2f}%")
}}}

---

## 4. Visualisations Essentielles pour PowerPoint

### 4.1 Distribution des Scores de Confiance

**Pourquoi l'utiliser ?**
- Montre si le modèle est **confiant** dans ses prédictions
- Détecte les zones d'incertitude

{{{python
import matplotlib.pyplot as plt

# Scores de confiance pour les classes correctes vs incorrectes
correct_conf = [0.92, 0.88, 0.95, 0.85, 0.90, 0.93]
incorrect_conf = [0.55, 0.62, 0.58, 0.51, 0.68, 0.60]

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.hist(correct_conf, bins=10, alpha=0.7, color='green', label='Correct')
plt.xlabel('Score de confiance')
plt.ylabel('Fréquence')
plt.title('Prédictions Correctes')
plt.legend()

plt.subplot(1, 2, 2)
plt.hist(incorrect_conf, bins=10, alpha=0.7, color='red', label='Incorrect')
plt.xlabel('Score de confiance')
plt.ylabel('Fréquence')
plt.title('Prédictions Incorrectes')
plt.legend()

plt.tight_layout()
plt.show()
}}}

**Pour PowerPoint :**
- Histogrammes côte à côte (correct vs incorrect)
- Aide à justifier le seuil de décision choisi

---

### 4.2 Learning Curves (Courbes d'Apprentissage)

**Pourquoi l'utiliser ?**
- Diagnostic d'**overfitting** ou **underfitting**
- Montre l'évolution de la performance pendant l'entraînement

{{{python
# Exemple avec historique d'entraînement
epochs = range(1, 51)
train_loss = [0.8 - 0.01*e + np.random.rand()*0.05 for e in epochs]
val_loss = [0.75 - 0.008*e + np.random.rand()*0.08 for e in epochs]

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(epochs, train_loss, label='Train Loss', linewidth=2)
plt.plot(epochs, val_loss, label='Validation Loss', linewidth=2)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Courbes de Loss')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
train_acc = [0.6 + 0.008*e - np.random.rand()*0.02 for e in epochs]
val_acc = [0.58 + 0.007*e - np.random.rand()*0.03 for e in epochs]
plt.plot(epochs, train_acc, label='Train Accuracy', linewidth=2)
plt.plot(epochs, val_acc, label='Validation Accuracy', linewidth=2)
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Courbes d\'Accuracy')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
}}}

**Diagnostic :**
- **Overfitting** : Train loss continue de baisser, Val loss remonte
- **Underfitting** : Train et Val loss élevées et stagnent
- **Good fit** : Train et Val loss convergent vers une valeur basse

---

### 4.3 Exemples de Prédictions (Good/Bad Cases)

**Pourquoi l'utiliser ?**
- **Rend le modèle tangible** pour la direction et les utilisateurs
- Montre les forces et faiblesses du modèle

**Structure pour PowerPoint :**

**Slide "Exemples de Succès" :**
- 3-4 exemples bien classés avec score de confiance élevé
- Légende : Email, vraie classe, prédiction, confiance

**Slide "Cas d'Erreurs" :**
- 3-4 exemples mal classés
- Analyse : Pourquoi le modèle s'est trompé ?

{{{python
# Exemple pour classification d'emails
examples = [
    {"text": "URGENT : Problème critique client VIP", 
     "true": "Urgent", "pred": "Urgent", "conf": 0.95, "correct": True},
    {"text": "Réunion hebdomadaire lundi 10h", 
     "true": "Normal", "pred": "Normal", "conf": 0.88, "correct": True},
    {"text": "Merci pour votre réponse rapide", 
     "true": "Normal", "pred": "Important", "conf": 0.62, "correct": False},
]

# Visualisation pour PowerPoint (tableau ou cards)
for ex in examples:
    color = 'green' if ex['correct'] else 'red'
    print(f"[{color.upper()}] {ex['text'][:50]}...")
    print(f"  Vraie classe: {ex['true']} | Prédiction: {ex['pred']} | Confiance: {ex['conf']:.0%}\n")
}}}

---

### 4.4 Feature Importance (Importance des Variables)

**Pourquoi l'utiliser ?**
- Explique **quelles variables** influencent le modèle
- Utile pour les modèles type Random Forest, XGBoost
- Crédibilise le modèle auprès des experts métier

{{{python
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

# Exemple avec Random Forest
feature_names = ['Longueur email', 'Nb mots-clés urgent', 'Heure réception', 
                 'Expéditeur VIP', 'Pièces jointes']
importances = [0.35, 0.28, 0.18, 0.12, 0.07]

plt.figure(figsize=(10, 6))
plt.barh(feature_names, importances, color='steelblue')
plt.xlabel('Importance')
plt.title('Importance des Variables dans le Modèle')
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.show()
}}}

**Pour PowerPoint :**
- Top 10 features les plus importantes
- Aide à valider que le modèle utilise des variables pertinentes

---

### 4.5 Calibration Plot (Courbe de Calibration)

**Pourquoi l'utiliser ?**
- Vérifie si les **probabilités prédites** sont fiables
- Exemple : Si le modèle prédit 70% de confiance, est-ce qu'il a raison 70% du temps ?

{{{python
from sklearn.calibration import calibration_curve

prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=10)

plt.figure(figsize=(10, 8))
plt.plot(prob_pred, prob_true, marker='o', linewidth=2, label='Modèle')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Parfaitement calibré')
plt.xlabel('Probabilité prédite')
plt.ylabel('Fraction de positifs')
plt.title('Courbe de Calibration')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
}}}

**Interprétation :**
- Courbe proche de la diagonale = modèle bien calibré
- Courbe au-dessus = modèle sous-confiant
- Courbe en-dessous = modèle sur-confiant

---

## 5. Métriques Business et ROI

### 5.1 Temps Économisé

**Formule :**

$$\text{Temps économisé} = (\text{Temps manuel} - \text{Temps avec IA}) \times \text{Volume traité}$$

**Exemple (Classification d'emails) :**
- Temps manuel : 3 min/email
- Temps avec IA : 30 sec/email (vérification)
- Volume : 1000 emails/mois
- **Temps économisé = (3 - 0.5) × 1000 = 2500 min/mois = 42h/mois**

{{{python
# Calculateur ROI
def calculate_time_saved(manual_time_per_item, ai_time_per_item, 
                         volume_per_month, hourly_wage):
    time_saved_per_item = manual_time_per_item - ai_time_per_item
    total_time_saved_hours = (time_saved_per_item * volume_per_month) / 60
    cost_saved = total_time_saved_hours * hourly_wage
    
    return {
        'time_saved_hours': total_time_saved_hours,
        'cost_saved_monthly': cost_saved,
        'cost_saved_yearly': cost_saved * 12
    }

results = calculate_time_saved(
    manual_time_per_item=3,      # 3 min
    ai_time_per_item=0.5,        # 30 sec
    volume_per_month=1000,       # 1000 emails/mois
    hourly_wage=40               # 40€/h
)

print(f"Temps économisé : {results['time_saved_hours']:.0f} heures/mois")
print(f"Coût économisé : {results['cost_saved_yearly']:,.0f}€/an")
}}}

---

### 5.2 Taux d'Automatisation

**Formule :**

$$\text{Taux d'automatisation} = \frac{\text{Nombre de cas traités automatiquement}}{\text{Nombre total de cas}} \times 100$$

**Exemple :**
- 1000 emails traités
- 850 traités automatiquement (confiance > 0.8)
- 150 nécessitent une vérification humaine
- **Taux d'automatisation = 850/1000 = 85%**

**Pour PowerPoint :**
- Graphique en donut montrant le taux d'automatisation
- Objectif : 80%+ pour justifier l'investissement IA

---

### 5.3 Réduction du Taux d'Erreur

**Avant/Après avec IA :**

| Métrique                | Avant IA (Manuel) | Avec IA | Amélioration |
|-------------------------|-------------------|---------|--------------|
| Emails mal classés      | 8%                | 3%      | -62.5%       |
| Temps de traitement     | 3h/jour           | 45min   | -75%         |
| Réclamations clients    | 15/mois           | 4/mois  | -73%         |

**Pour PowerPoint :**
- Tableau comparatif avant/après
- Focus sur les KPIs métier (réclamations, temps, erreurs)

---

## 6. Communication de l'Incertitude

### 6.1 Intervalles de Confiance

**Pourquoi l'utiliser ?**
- Quantifie l'**incertitude** des prédictions
- Critique pour les décisions à fort impact

**Exemple (Régression) :**

{{{python
from sklearn.ensemble import GradientBoostingRegressor
import numpy as np

# Entraînement avec quantile regression
model_lower = GradientBoostingRegressor(loss='quantile', alpha=0.05)
model_upper = GradientBoostingRegressor(loss='quantile', alpha=0.95)

# X_train, y_train : données d'entraînement
# Prédictions avec intervalles de confiance à 90%
# y_pred_lower = model_lower.predict(X_test)
# y_pred_upper = model_upper.predict(X_test)

# Visualisation
plt.figure(figsize=(12, 6))
x = np.arange(len(y_true))
plt.plot(x, y_true, 'o', label='Vraies valeurs', alpha=0.5)
plt.plot(x, y_pred, '-', label='Prédictions', linewidth=2)
# plt.fill_between(x, y_pred_lower, y_pred_upper, alpha=0.3, 
#                  label='Intervalle de confiance 90%')
plt.xlabel('Échantillon')
plt.ylabel('Valeur')
plt.title('Prédictions avec Intervalles de Confiance')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
}}}

---

### 6.2 Scores de Confiance et Seuils de Décision

**Stratégie pour la production :**

| Score de confiance | Action                              |
|--------------------|-------------------------------------|
| > 0.9              | Traitement automatique              |
| 0.7 - 0.9          | Traitement automatique + alerte     |
| 0.5 - 0.7          | Revue humaine obligatoire           |
| < 0.5              | Rejet ou escalade                   |

**Pour PowerPoint :**
- Présente cette stratégie de seuils
- Montre la distribution des scores pour estimer le volume de revue humaine

---

## 7. Présentation des Erreurs (Failure Cases)

### 7.1 Analyse des Types d'Erreurs

**Classification des erreurs :**

1. **Erreurs acceptables** : Cas limites, ambiguïtés
2. **Erreurs critiques** : Mauvaise classification avec forte confiance
3. **Erreurs systématiques** : Pattern récurrent (ex: confusion systématique entre 2 classes)

**Pour PowerPoint :**

**Slide "Analyse des Erreurs" :**
- Camembert : Répartition des types d'erreurs
- Liste des top 3 causes d'erreurs
- Actions correctives envisagées

{{{python
# Analyse des erreurs par catégorie
error_categories = {
    'Emails ambigus (frontière floue)': 45,
    'Manque de contexte': 25,
    'Nouveaux types non vus en train': 15,
    'Erreurs aléatoires': 15
}

plt.figure(figsize=(10, 8))
plt.pie(error_categories.values(), labels=error_categories.keys(), 
        autopct='%1.1f%%', startangle=90, colors=['#ff9999', '#66b3ff', '#99ff99', '#ffcc99'])
plt.title('Répartition des Types d\'Erreurs', fontsize=14)
plt.show()
}}}

---

### 7.2 Exemples d'Erreurs Annotés

**Structure :**

Pour chaque erreur présentée :
1. **Texte de l'email** (ou image pour YOLO)
2. **Vraie classe** vs **Prédiction**
3. **Score de confiance**
4. **Explication** : Pourquoi le modèle s'est trompé ?
5. **Action** : Comment corriger (retraining, ajout de features, etc.) ?

**Exemple (Classification email) :**

Email : "Bonjour, pouvez-vous me rappeler demain ?"
Vraie classe : Important
Prédiction : Normal (confiance 0.68)
Explication : Manque de mots-clés "urgent", phrase courte, pas de deadline précise.
Action : Ajouter feature "demande d'action" détectée par NER.


---

## 8. Comparaison de Modèles

### 8.1 Tableau Comparatif Multi-Critères

**Pour PowerPoint :**

| Modèle           | Accuracy | F1-Score | Temps inférence | Taille modèle | ROI (€/an) |
|------------------|----------|----------|-----------------|---------------|------------|
| Baseline (règles)| 0.72     | 0.65     | 5 ms            | -             | 0          |
| Random Forest    | 0.84     | 0.81     | 15 ms           | 50 MB         | 25 000     |
| BERT fine-tuned  | 0.91     | 0.89     | 80 ms           | 440 MB        | 38 000     |
| **GPT-4 (API)**  | **0.93** | **0.91** | 1200 ms         | -             | 15 000     |

**Recommandation** : BERT fine-tuned (meilleur compromis perf/coût/latence)

---

### 8.2 Graphiques de Comparaison

**Radar Chart (Comparaison multi-dimensionnelle) :**

{{{python
import matplotlib.pyplot as plt
import numpy as np

categories = ['Accuracy', 'Vitesse', 'Coût', 'Maintenabilité', 'Explicabilité']
model1 = [0.84, 0.9, 0.95, 0.85, 0.9]  # Random Forest
model2 = [0.91, 0.5, 0.7, 0.6, 0.4]    # BERT

angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
model1 += model1[:1]
model2 += model2[:1]
angles += angles[:1]

fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
ax.plot(angles, model1, 'o-', linewidth=2, label='Random Forest', color='blue')
ax.fill(angles, model1, alpha=0.25, color='blue')
ax.plot(angles, model2, 'o-', linewidth=2, label='BERT', color='red')
ax.fill(angles, model2, alpha=0.25, color='red')
ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, size=12)
ax.set_ylim(0, 1)
ax.set_title('Comparaison Multi-Critères des Modèles', size=16, pad=20)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
ax.grid(True)
plt.tight_layout()
plt.show()
}}}

---

### 8.3 Trade-off Précision vs Coût/Vitesse

**Scatter Plot :**

{{{python
models = ['Baseline', 'Random Forest', 'BERT', 'GPT-4 API']
f1_scores = [0.65, 0.81, 0.89, 0.91]
costs = [0, 100, 500, 2000]  # Coût annuel d'inférence (€)

plt.figure(figsize=(12, 8))
colors = ['gray', 'green', 'blue', 'red']
sizes = [100, 200, 200, 200]

for i, model in enumerate(models):
    plt.scatter(costs[i], f1_scores[i], s=sizes[i], alpha=0.6, 
                c=colors[i], label=model)
    plt.annotate(model, (costs[i], f1_scores[i]), 
                 fontsize=12, ha='center', xytext=(0, 15),
                 textcoords='offset points')

plt.xlabel('Coût annuel d\'inférence (€)', fontsize=14)
plt.ylabel('F1-Score', fontsize=14)
plt.title('Trade-off Performance vs Coût', fontsize=16)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=12)
plt.tight_layout()
plt.show()
}}}

---

## 9. Adaptation selon l'Audience

### 9.1 Pour la Direction (Non-Experts Techniques)

**À privilégier :**
- ✅ **Métriques business** : Temps économisé, ROI, taux d'automatisation
- ✅ **Graphiques simples** : Barres, camemberts, avant/après
- ✅ **Exemples concrets** : "Le modèle détecte 9 emails urgents sur 10"
- ✅ **Comparaison avec baseline** : Montrer l'amélioration vs situation actuelle
- ✅ **Visualisations de résultats** : Matrice de confusion avec couleurs

**À éviter :**
- ❌ Formules mathématiques complexes
- ❌ Jargon technique (overfitting, hyperparamètres, etc.)
- ❌ Trop de métriques (s'en tenir à 2-3 max)

**Exemple de slide pour direction :**

**Titre : "Résultats du Modèle de Classification d'Emails"**
- 📊 Accuracy : 91% (vs 72% avec règles manuelles)
- ⏱️ Temps économisé : 42h/mois = 50 000€/an
- 🤖 Taux d'automatisation : 85%
- ⭐ Satisfaction utilisateurs : +35%

---

### 9.2 Pour les Utilisateurs Métier

**À privilégier :**
- ✅ **Exemples réels** : Montrer des emails/images qu'ils reconnaissent
- ✅ **Cas d'usage** : "Comment le modèle m'aide au quotidien ?"
- ✅ **Transparence sur les limites** : "Quand le modèle a besoin d'aide humaine"
- ✅ **Feedback loop** : "Comment signaler une erreur ?"

**À éviter :**
- ❌ Métriques techniques (AUC, F1, mAP)
- ❌ Comparaison de modèles (peu pertinent pour eux)

---

### 9.3 Pour les Experts IA (Chef, Pairs)

**À privilégier :**
- ✅ **Métriques techniques complètes** : Precision, Recall, F1, AUC, mAP
- ✅ **Courbes détaillées** : ROC, Precision-Recall, Learning curves
- ✅ **Détails d'architecture** : Choix du modèle, hyperparamètres, augmentation de données
- ✅ **Analyse approfondie des erreurs** : Failure modes, confusion entre classes
- ✅ **Benchmarks** : Comparaison avec SOTA (state-of-the-art)

**Exemple de slide pour experts :**

**Titre : "Performances du Modèle BERT Fine-tuned"**
- Macro-F1 : 0.89 (vs 0.81 Random Forest)
- AUC-ROC : 0.94
- mAP@0.5 : 0.87 (pour détection YOLO)
- Confusion principale : Classe "Important" vs "Urgent" (15% des erreurs)
- Hyperparamètres : learning_rate=2e-5, batch_size=16, epochs=10

---

## 10. Structure Type de Présentation PowerPoint

### 10.1 Slide 1 : Contexte et Objectif

**Contenu :**
- Problème métier à résoudre
- Objectif du modèle
- Bénéfices attendus

**Exemple :**

Titre : Classification Automatique des Emails Clients
Problème :

    1000+ emails/jour reçus
    Temps de tri manuel : 3 min/email
    Emails urgents parfois ratés

Objectif :
Automatiser la classification en 3 catégories (Normal, Important, Urgent)
Bénéfices :

    Réduction du temps de traitement de 75%
    Priorisation automatique des emails urgents
    Amélioration satisfaction client


---

### 10.2 Slide 2 : Données et Méthode

**Contenu :**
- Description du dataset
- Répartition des classes
- Approche choisie (modèle)

**Exemple :**

Dataset :

    15 000 emails étiquetés manuellement
    Période : Jan 2023 - Déc 2024
    Répartition : Normal (60%), Important (30%), Urgent (10%)

Approche :

    Modèle : BERT fine-tuned (Transformer)
    Split : 70% train, 15% validation, 15% test
    Augmentation de données pour classe "Urgent" (SMOTE)


---

### 10.3 Slide 3 : Résultats Principaux

**Contenu :**
- Métriques clés (2-3 max pour la direction)
- Comparaison avec baseline
- Graphique de performance

**Exemple (Direction) :**

Résultats :
✅ Accuracy : 91% (vs 72% règles manuelles)
✅ F1-Score : 0.89
✅ Emails urgents détectés : 93% (vs 68% avant)
[Graphique : Barres comparatives Baseline vs Modèle IA]

**Exemple (Experts IA) :**

      Performances :
      
          Macro-F1 : 0.89
          AUC-ROC : 0.94
          Precision/Recall par classe :
              Normal : 0.95 / 0.93
              Important : 0.87 / 0.88
              Urgent : 0.82 / 0.85
      
      [Graphique : Matrice de confusion + Courbe ROC]
---

### 10.4 Slide 4 : Analyse Détaillée (Optionnel pour Direction)

**Contenu :**
- Matrice de confusion
- Exemples de prédictions
- Distribution des scores de confiance

**Pour Direction :**

      Où le modèle excelle :
      ✅ Détection d'emails avec mots-clés urgents ("URGENT", "ASAP")
      ✅ Emails de clients VIP automatiquement priorisés
      Points d'attention :
      ⚠️ Emails ambigus nécessitent validation humaine (15%)
      ⚠️ Nouveaux types d'urgence non vus en entraînement
      [Graphique : Exemples d'emails bien/mal classés]

---

### 10.5 Slide 5 : Impact Business et ROI

**Contenu :**
- Temps économisé
- Coûts évités
- Amélioration des KPIs métier

**Exemple :**

      Impact Business :
      💰 Temps économisé : 42h/mois = 50 000€/an
      📈 Taux d'automatisation : 85%
      📉 Réclamations clients : -65%
      ⭐ Satisfaction utilisateurs : +35%
      ROI :
      
          Coût développement + infra : 30 000€
          Économies annuelles : 50 000€
          Retour sur investissement : 6 mois
      
      [Graphique : Évolution du ROI sur 2 ans]

Jackie
Jackie
17:26
Completed in 14 min 59 sec
Parfait 
@Clément o
 ! Je vais te créer un guide complet et structuré. Vu l'ampleur du sujet, je vais le découper en sections claires pour que tu puisses naviguer facilement.
🚀 Je te prépare ton "Guide ultime de présentation des résultats de modèles IA"
Donne-moi quelques minutes pour structurer tout ça proprement avec :

    Les métriques essentielles (math + cookbook)
    Les visualisations clés pour PowerPoint
    Les adaptations selon ton audience (direction vs experts IA)
    Les cas spécifiques : classification d'emails et détection YOLO
    Métriques business, incertitude, failure cases, comparaisons de modèles

Je reviens avec le guide complet ! ⏳

# Guide Complet : Présentation des Résultats de Modèles IA

**Guide de référence pour présenter des résultats de modèles IA de manière claire, professionnelle et pertinente**

*Auteur : Guide pour Clément O.*  
*Contexte : Présentations PowerPoint pour direction, utilisateurs et experts IA*  
*Domaines : NLP (Transformers/BERT), Computer Vision (YOLO), Classification, Régression*

---

## Table des matières

1. [Métriques de Classification](#1-métriques-de-classification)
2. [Métriques de Détection d'Objets (YOLO)](#2-métriques-de-détection-dobjets-yolo)
3. [Métriques de Régression](#3-métriques-de-régression)
4. [Visualisations Essentielles pour PowerPoint](#4-visualisations-essentielles-pour-powerpoint)
5. [Métriques Business et ROI](#5-métriques-business-et-roi)
6. [Communication de l'Incertitude](#6-communication-de-lincertitude)
7. [Présentation des Erreurs (Failure Cases)](#7-présentation-des-erreurs-failure-cases)
8. [Comparaison de Modèles](#8-comparaison-de-modèles)
9. [Adaptation selon l'Audience](#9-adaptation-selon-laudience)
10. [Structure Type de Présentation PowerPoint](#10-structure-type-de-présentation-powerpoint)

---

## 1. Métriques de Classification

*Utilisées pour : Classification d'emails, classification d'images, détection binaire/multi-classes*

### 1.1 Matrice de Confusion

**Définition :**  
Tableau croisant les prédictions du modèle avec les vraies valeurs.

**Structure (cas binaire) :**

|                     | **Prédit Positif** | **Prédit Négatif** |
|---------------------|--------------------|--------------------|
| **Réel Positif**    | TP (Vrais Positifs)| FN (Faux Négatifs) |
| **Réel Négatif**    | FP (Faux Positifs) | TN (Vrais Négatifs)|

**Pourquoi l'utiliser ?**
- Vision instantanée des types d'erreurs
- Identification des classes confondues (en multi-classes)
- Essentiel pour expliquer les performances à la direction

**Exemple de code :**

{{{python
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# y_true : vraies étiquettes, y_pred : prédictions
y_true = [0, 1, 0, 1, 1, 0, 1, 0]
y_pred = [0, 1, 0, 0, 1, 0, 1, 1]

cm = confusion_matrix(y_true, y_pred)

# Visualisation
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Négatif', 'Positif'],
            yticklabels=['Négatif', 'Positif'])
plt.ylabel('Vraie classe')
plt.xlabel('Classe prédite')
plt.title('Matrice de Confusion')
plt.show()
}}}

**Pour PowerPoint :**
- Utilise une heatmap avec annotations
- Ajoute des pourcentages en plus des valeurs brutes
- Version normalisée pour comparer entre datasets

---

### 1.2 Accuracy (Exactitude)

**Formule mathématique :**  

$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$

**Interprétation :**  
Proportion de prédictions correctes parmi toutes les prédictions.

**Pourquoi l'utiliser ?**
- Métrique la plus simple et intuitive pour la direction
- Bonne vue d'ensemble si les classes sont équilibrées

**⚠️ Piège à éviter :**
- **Classes déséquilibrées** : Si 95% des emails sont "non-urgents", un modèle prédisant toujours "non-urgent" aura 95% d'accuracy mais sera inutile !

**Quand l'utiliser ?**
- ✅ Classes équilibrées (ratio 40-60 minimum)
- ✅ Présentation rapide à la direction (mais toujours accompagner d'autres métriques)
- ❌ Classes très déséquilibrées (fraude, anomalies, emails urgents rares)

{{{python
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_true, y_pred)
print(f"Accuracy : {accuracy:.2%}")  # 75.00%
}}}

---

### 1.3 Precision (Précision)

**Formule mathématique :**  

$$\text{Precision} = \frac{TP}{TP + FP}$$

**Interprétation :**  
Parmi les emails classés "urgents" par le modèle, quelle proportion l'est réellement ?

**Pourquoi l'utiliser ?**
- Quand les **Faux Positifs sont coûteux**
- Exemple : Classification "urgent" → si beaucoup de FP, on perd du temps à traiter des emails non-urgents

**Cas d'usage typique :**
- Filtrage spam (on ne veut pas bloquer des emails légitimes)
- Détection de produits défectueux (coût d'inspection inutile)

{{{python
from sklearn.metrics import precision_score

precision = precision_score(y_true, y_pred)
print(f"Precision : {precision:.2%}")
}}}

---

### 1.4 Recall (Rappel / Sensibilité)

**Formule mathématique :**  

$$\text{Recall} = \frac{TP}{TP + FN}$$

**Interprétation :**  
Parmi tous les emails réellement urgents, quelle proportion a été détectée par le modèle ?

**Pourquoi l'utiliser ?**
- Quand les **Faux Négatifs sont coûteux**
- Exemple : Classification "urgent" → si beaucoup de FN, on rate des emails urgents !

**Cas d'usage typique :**
- Détection de maladies (on ne veut pas rater un malade)
- Détection d'anomalies critiques
- Emails urgents clients VIP

{{{python
from sklearn.metrics import recall_score

recall = recall_score(y_true, y_pred)
print(f"Recall : {recall:.2%}")
}}}

---

### 1.5 F1-Score

**Formule mathématique :**  

$\$F1 = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

**Interprétation :**  
Moyenne harmonique entre Precision et Recall. Équilibre entre les deux.

**Pourquoi l'utiliser ?**
- Quand Precision ET Recall sont importants
- **Métrique unique** pour comparer des modèles
- Gère mieux les classes déséquilibrées que l'Accuracy

**⚠️ Attention :**
- F1 = 0.80 ne veut pas dire 80% de performance globale
- Un F1 élevé n'est bon que si Precision ET Recall sont équilibrés

{{{python
from sklearn.metrics import f1_score

f1 = f1_score(y_true, y_pred)
print(f"F1-Score : {f1:.2%}")
}}}

**Trade-off Precision vs Recall :**

Imaginons un modèle de classification d'emails urgents :
- **Seuil bas (0.3)** : Le modèle prédit "urgent" facilement → Recall élevé mais Precision basse (beaucoup de FP)
- **Seuil élevé (0.8)** : Le modèle est prudent → Precision élevée mais Recall bas (on rate des urgents)

{{{python
from sklearn.metrics import precision_recall_curve

# Obtenir les probabilités prédites
y_proba = [0.1, 0.9, 0.2, 0.4, 0.85, 0.15, 0.95, 0.6]

precision, recall, thresholds = precision_recall_curve(y_true, y_proba)

plt.figure(figsize=(10, 6))
plt.plot(thresholds, precision[:-1], label='Precision')
plt.plot(thresholds, recall[:-1], label='Recall')
plt.xlabel('Seuil de décision')
plt.ylabel('Score')
plt.title('Trade-off Precision vs Recall selon le seuil')
plt.legend()
plt.grid(True)
plt.show()
}}}

---

### 1.6 Specificité (Specificity)

**Formule mathématique :**  

$$\text{Specificity} = \frac{TN}{TN + FP}$$

**Interprétation :**  
Parmi tous les emails NON-urgents, quelle proportion a été correctement identifiée ?

**Pourquoi l'utiliser ?**
- Complémentaire au Recall
- Important en médical : capacité à identifier les personnes saines

---

### 1.7 Balanced Accuracy

**Formule mathématique :**  

$$\text{Balanced Accuracy} = \frac{\text{Recall}_{\text{classe 1}} + \text{Specificity}}{2}$$

**Pourquoi l'utiliser ?**
- **Classes déséquilibrées** : Meilleure alternative à l'Accuracy classique
- Donne le même poids à chaque classe

**Exemple :**

{{{python
from sklearn.metrics import balanced_accuracy_score

balanced_acc = balanced_accuracy_score(y_true, y_pred)
print(f"Balanced Accuracy : {balanced_acc:.2%}")
}}}

---

### 1.8 Courbe ROC et AUC

**ROC (Receiver Operating Characteristic) :**  
Courbe montrant le trade-off entre Recall (Taux de Vrais Positifs) et Taux de Faux Positifs à différents seuils.

**AUC (Area Under Curve) :**  
Aire sous la courbe ROC. Varie entre 0 et 1.

**Interprétation de l'AUC :**
- **AUC = 0.5** : Modèle aléatoire (inutile)
- **AUC = 0.7-0.8** : Acceptable
- **AUC = 0.8-0.9** : Bon modèle
- **AUC > 0.9** : Excellent modèle
- **AUC = 1.0** : Modèle parfait (ou overfitting !)

**Pourquoi l'utiliser ?**
- Évalue la capacité du modèle à discriminer entre classes
- **Indépendant du seuil** de décision choisi
- Très utilisé en recherche et facile à présenter

{{{python
from sklearn.metrics import roc_curve, auc

fpr, tpr, thresholds = roc_curve(y_true, y_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(10, 8))
plt.plot(fpr, tpr, color='darkorange', lw=2, 
         label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
         label='Aléatoire (AUC = 0.50)')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Taux de Faux Positifs (1 - Specificité)')
plt.ylabel('Taux de Vrais Positifs (Recall)')
plt.title('Courbe ROC')
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.show()
}}}

**Pour PowerPoint :**
- Ajoute la ligne de référence (diagonale) pour montrer le niveau aléatoire
- Indique clairement la valeur de l'AUC
- Si comparaison de modèles : superpose plusieurs courbes ROC

---

### 1.9 Courbe Précision-Recall

**Pourquoi l'utiliser plutôt que ROC ?**
- **Classes très déséquilibrées** : Plus informative que ROC
- Exemple : détection d'anomalies (1% de positifs), fraude, emails urgents rares

**Différence ROC vs Precision-Recall :**

{{{python
from sklearn.metrics import precision_recall_curve, average_precision_score

precision, recall, _ = precision_recall_curve(y_true, y_proba)
avg_precision = average_precision_score(y_true, y_proba)

plt.figure(figsize=(10, 8))
plt.plot(recall, precision, color='blue', lw=2,
         label=f'PR curve (AP = {avg_precision:.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Courbe Précision-Recall')
plt.legend(loc="upper right")
plt.grid(True, alpha=0.3)
plt.show()
}}}

**Cookbook : ROC ou PR ?**
- ✅ **ROC** : Classes équilibrées, présentation à des non-experts
- ✅ **Precision-Recall** : Classes déséquilibrées, focus sur la classe minoritaire

---

### 1.10 Métriques Multi-Classes

**Macro-Average :**  
Moyenne simple des métriques de chaque classe (toutes les classes ont le même poids).

$$\text{Macro-F1} = \frac{1}{n} \sum_{i=1}^{n} F1_i$$

**Micro-Average :**  
Agrège les TP, FP, FN de toutes les classes puis calcule la métrique globale.

**Weighted-Average :**  
Moyenne pondérée par le nombre d'échantillons de chaque classe.

**Quand utiliser quoi ?**
- **Macro** : Toutes les classes ont la même importance (même si déséquilibrées)
- **Micro** : Focus sur la performance globale
- **Weighted** : Importance proportionnelle au nombre d'échantillons

{{{python
from sklearn.metrics import classification_report

# Exemple multi-classes : urgence email (0=normal, 1=important, 2=urgent)
y_true_multi = [0, 1, 2, 0, 1, 2, 0, 1]
y_pred_multi = [0, 1, 1, 0, 1, 2, 0, 2]

print(classification_report(y_true_multi, y_pred_multi, 
                           target_names=['Normal', 'Important', 'Urgent']))
}}}

---

## 2. Métriques de Détection d'Objets (YOLO)

*Utilisées pour : Détection automatisée par computer vision, segmentation, localisation*

### 2.1 IoU (Intersection over Union)

**Formule mathématique :**  

$$\text{IoU} = \frac{\text{Aire de l'intersection}}{\text{Aire de l'union}} = \frac{A \cap B}{A \cup B}$$

**Interprétation :**  
Mesure le chevauchement entre la bounding box prédite et la bounding box réelle.

**Valeurs typiques :**
- **IoU ≥ 0.5** : Détection considérée comme correcte (seuil standard)
- **IoU ≥ 0.75** : Détection de haute qualité
- **IoU < 0.5** : Faux Positif

**Pourquoi l'utiliser ?**
- Détermine si une détection est un TP ou un FP
- Essentiel pour calculer la Precision et le Recall en détection d'objets

**Visualisation :**

{{{python
def calculate_iou(box1, box2):
    """
    box format: [x1, y1, x2, y2] (top-left et bottom-right)
    """
    # Coordonnées de l'intersection
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])
    
    # Aire de l'intersection
    inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)
    
    # Aires des deux boxes
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    # Aire de l'union
    union_area = box1_area + box2_area - inter_area
    
    # IoU
    iou = inter_area / union_area if union_area > 0 else 0
    return iou

# Exemple
box_ground_truth = [50, 50, 150, 150]  # Vraie boîte
box_predicted = [60, 60, 160, 140]     # Prédiction

iou = calculate_iou(box_ground_truth, box_predicted)
print(f"IoU : {iou:.2f}")  # 0.68
}}}

---

### 2.2 mAP (mean Average Precision)

**Définition :**  
Métrique principale pour évaluer les modèles de détection d'objets.

**Calcul en étapes :**

1. **Pour chaque classe** :
   - Ordonne toutes les détections par score de confiance décroissant
   - Calcule Precision et Recall à chaque seuil
   - Calcule l'Average Precision (AP) = aire sous la courbe Précision-Recall
   
2. **mAP** = Moyenne des AP de toutes les classes

**Variantes :**
- **mAP@0.5** : IoU seuil = 0.5 (standard PASCAL VOC)
- **mAP@0.75** : IoU seuil = 0.75 (plus strict)
- **mAP@[0.5:0.95]** : Moyenne sur plusieurs seuils IoU (standard COCO)

**Pourquoi l'utiliser ?**
- **Métrique standard** pour benchmarker les modèles de détection
- Prend en compte Precision, Recall ET la localisation (via IoU)
- Permet de comparer différentes architectures (YOLOv5 vs YOLOv8 vs Faster R-CNN)

**Interprétation :**
- **mAP@0.5 = 0.60** : Performance acceptable pour des applications réelles
- **mAP@0.5 = 0.80+** : Très bon modèle
- **mAP@0.5:0.95** : Plus exigeant (souvent 20-30% inférieur au mAP@0.5)

{{{python
# Avec YOLOv8 (ultralytics)
from ultralytics import YOLO

model = YOLO('yolov8n.pt')
results = model.val(data='coco.yaml')  # Validation sur dataset

print(f"mAP@0.5 : {results.box.map50:.3f}")
print(f"mAP@0.5:0.95 : {results.box.map:.3f}")
}}}

**Pour PowerPoint :**
- Tableau comparatif des mAP entre modèles
- Graphique en barres par classe (montre les classes bien/mal détectées)

---

### 2.3 Precision et Recall en détection

**Adaptation pour détection d'objets :**

- **True Positive (TP)** : Détection avec IoU ≥ seuil et bonne classe
- **False Positive (FP)** : Détection avec IoU < seuil OU mauvaise classe OU détection en doublon
- **False Negative (FN)** : Objet non détecté

**Calcul identique à la classification :**

$$\text{Precision} = \frac{TP}{TP + FP}$$

$$\text{Recall} = \frac{TP}{TP + FN}$$

**Exemple concret (détection de défauts) :**
- 100 défauts réels dans les images
- Modèle détecte 90 défauts (dont 85 corrects et 5 faux)
- Recall = 85/100 = 0.85 (on a trouvé 85% des défauts)
- Precision = 85/90 = 0.94 (94% de nos détections sont correctes)

---

### 2.4 Confusion entre Classes (Détection)

**Pourquoi l'utiliser ?**
- Identifier les paires de classes confondues par le modèle
- Exemple : YOLO confond "voiture" et "camion"

**Visualisation :**

{{{python
import numpy as np
import seaborn as sns

# Matrice de confusion pour détection multi-classes
# Exemple : détection de 3 types de défauts
confusion_matrix_detection = np.array([
    [120, 10, 5],    # Défaut type A
    [8, 95, 12],     # Défaut type B
    [3, 15, 87]      # Défaut type C
])

plt.figure(figsize=(10, 8))
sns.heatmap(confusion_matrix_detection, annot=True, fmt='d', cmap='YlOrRd',
            xticklabels=['Défaut A', 'Défaut B', 'Défaut C'],
            yticklabels=['Défaut A', 'Défaut B', 'Défaut C'])
plt.ylabel('Vraie classe')
plt.xlabel('Classe prédite')
plt.title('Matrice de Confusion - Détection YOLO')
plt.show()
}}}

---

### 2.5 Vitesse d'Inférence (FPS, Latence)

**Métriques complémentaires critiques pour la production :**

- **FPS (Frames Per Second)** : Nombre d'images traitées par seconde
- **Latence** : Temps de traitement d'une image (ms)

**Pourquoi l'utiliser ?**
- Contraintes temps réel (vidéo surveillance, ligne de production)
- Trade-off précision vs vitesse

**Exemple :**

| Modèle      | mAP@0.5 | FPS (GPU) | Latence (ms) |
|-------------|---------|-----------|--------------|
| YOLOv8n     | 0.52    | 380       | 2.6          |
| YOLOv8s     | 0.61    | 285       | 3.5          |
| YOLOv8m     | 0.67    | 165       | 6.0          |
| YOLOv8l     | 0.70    | 110       | 9.1          |

**Pour PowerPoint :**
- Graphique scatter : mAP (axe Y) vs Latence (axe X)
- Montre le meilleur compromis précision/vitesse

{{{python
import matplotlib.pyplot as plt

models = ['YOLOv8n', 'YOLOv8s', 'YOLOv8m', 'YOLOv8l']
map_scores = [0.52, 0.61, 0.67, 0.70]
latency = [2.6, 3.5, 6.0, 9.1]

plt.figure(figsize=(10, 6))
plt.scatter(latency, map_scores, s=200, alpha=0.6)
for i, model in enumerate(models):
    plt.annotate(model, (latency[i], map_scores[i]), 
                 fontsize=12, ha='right')
plt.xlabel('Latence (ms)', fontsize=12)
plt.ylabel('mAP@0.5', fontsize=12)
plt.title('Trade-off Précision vs Vitesse', fontsize=14)
plt.grid(True, alpha=0.3)
plt.show()
}}}

---

## 3. Métriques de Régression

*Utilisées pour : Prédiction de valeurs continues (prix, délais, quantités)*

### 3.1 MAE (Mean Absolute Error)

**Formule mathématique :**  

$$\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$$

**Interprétation :**  
Erreur moyenne absolue, dans les mêmes unités que la variable cible.

**Pourquoi l'utiliser ?**
- **Interprétation directe** : "En moyenne, le modèle se trompe de X unités"
- Robuste aux outliers (pas de carré)
- Facile à expliquer à la direction

**Exemple :**
- Prédiction de temps de traitement d'emails : MAE = 5 min
- → Le modèle se trompe en moyenne de 5 minutes

{{{python
from sklearn.metrics import mean_absolute_error
import numpy as np

y_true = np.array([100, 120, 90, 110, 105])
y_pred = np.array([98, 125, 88, 105, 110])

mae = mean_absolute_error(y_true, y_pred)
print(f"MAE : {mae:.2f}")  # 3.80
}}}

---

### 3.2 RMSE (Root Mean Squared Error)

**Formule mathématique :**  

$$\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}$$

**Interprétation :**  
Racine carrée de la moyenne des erreurs au carré, dans les mêmes unités que la variable cible.

**Pourquoi l'utiliser ?**
- Pénalise davantage les **grandes erreurs** (grâce au carré)
- Utile quand les grandes erreurs sont particulièrement coûteuses
- Standard dans la communauté ML

**Différence MAE vs RMSE :**

{{{python
y_true = np.array([100, 100, 100, 100, 100])
y_pred1 = np.array([105, 105, 105, 105, 105])  # Erreurs constantes
y_pred2 = np.array([100, 100, 100, 100, 125])  # Une grosse erreur

mae1 = mean_absolute_error(y_true, y_pred1)
rmse1 = np.sqrt(np.mean((y_true - y_pred1)**2))
print(f"Cas 1 - MAE: {mae1:.2f}, RMSE: {rmse1:.2f}")  # MAE: 5.00, RMSE: 5.00

mae2 = mean_absolute_error(y_true, y_pred2)
rmse2 = np.sqrt(np.mean((y_true - y_pred2)**2))
print(f"Cas 2 - MAE: {mae2:.2f}, RMSE: {rmse2:.2f}")  # MAE: 5.00, RMSE: 11.18

# Même MAE mais RMSE plus élevé car une grosse erreur
}}}

**Quand préférer RMSE à MAE ?**
- Quand les grandes erreurs sont **inacceptables**
- Quand on veut être **conservateur**

**Quand préférer MAE à RMSE ?**
- Quand on veut une métrique **robuste aux outliers**
- Quand toutes les erreurs ont le **même coût**

---

### 3.3 R² (Coefficient de Détermination)

**Formule mathématique :**  

$\$R^2 = 1 - \frac{\sum_{i=1}^{n} (y_i - \hat{y}_i)^2}{\sum_{i=1}^{n} (y_i - \bar{y})^2}$$

Où $$\bar{y}$$ est la moyenne des valeurs réelles.

**Interprétation :**  
Proportion de la variance expliquée par le modèle.

- **R² = 1** : Modèle parfait
- **R² = 0.8** : Le modèle explique 80% de la variance
- **R² = 0** : Modèle aussi bon qu'une prédiction par la moyenne
- **R² < 0** : Modèle pire qu'une prédiction par la moyenne !

**Pourquoi l'utiliser ?**
- **Sans unité** : Permet de comparer des modèles sur différents problèmes
- Très utilisé en régression linéaire et statistiques

**⚠️ Attention :**
- R² peut être trompeur si le modèle est complexe (overfitting)
- Préférer le **R² ajusté** pour comparer des modèles avec différents nombres de variables

{{{python
from sklearn.metrics import r2_score

r2 = r2_score(y_true, y_pred)
print(f"R² : {r2:.3f}")
}}}

---

### 3.4 MAPE (Mean Absolute Percentage Error)

**Formule mathématique :**  

$$\text{MAPE} = \frac{100}{n} \sum_{i=1}^{n} \left| \frac{y_i - \hat{y}_i}{y_i} \right|$$

**Interprétation :**  
Erreur moyenne en pourcentage.

**Pourquoi l'utiliser ?**
- **Interprétation business** : "Le modèle se trompe en moyenne de X%"
- Utile pour communiquer avec la direction

**⚠️ Attention :**
- Sensible aux **valeurs proches de zéro** (division par $$y_i$$)
- Asymétrique : sous-estimation pénalisée plus que sur-estimation

{{{python
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

mape = mean_absolute_percentage_error(y_true, y_pred)
print(f"MAPE : {mape:.2f}%")
}}}

---

## 4. Visualisations Essentielles pour PowerPoint

### 4.1 Distribution des Scores de Confiance

**Pourquoi l'utiliser ?**
- Montre si le modèle est **confiant** dans ses prédictions
- Détecte les zones d'incertitude

{{{python
import matplotlib.pyplot as plt

# Scores de confiance pour les classes correctes vs incorrectes
correct_conf = [0.92, 0.88, 0.95, 0.85, 0.90, 0.93]
incorrect_conf = [0.55, 0.62, 0.58, 0.51, 0.68, 0.60]

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.hist(correct_conf, bins=10, alpha=0.7, color='green', label='Correct')
plt.xlabel('Score de confiance')
plt.ylabel('Fréquence')
plt.title('Prédictions Correctes')
plt.legend()

plt.subplot(1, 2, 2)
plt.hist(incorrect_conf, bins=10, alpha=0.7, color='red', label='Incorrect')
plt.xlabel('Score de confiance')
plt.ylabel('Fréquence')
plt.title('Prédictions Incorrectes')
plt.legend()

plt.tight_layout()
plt.show()
}}}

**Pour PowerPoint :**
- Histogrammes côte à côte (correct vs incorrect)
- Aide à justifier le seuil de décision choisi

---

### 4.2 Learning Curves (Courbes d'Apprentissage)

**Pourquoi l'utiliser ?**
- Diagnostic d'**overfitting** ou **underfitting**
- Montre l'évolution de la performance pendant l'entraînement

{{{python
# Exemple avec historique d'entraînement
epochs = range(1, 51)
train_loss = [0.8 - 0.01*e + np.random.rand()*0.05 for e in epochs]
val_loss = [0.75 - 0.008*e + np.random.rand()*0.08 for e in epochs]

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(epochs, train_loss, label='Train Loss', linewidth=2)
plt.plot(epochs, val_loss, label='Validation Loss', linewidth=2)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Courbes de Loss')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
train_acc = [0.6 + 0.008*e - np.random.rand()*0.02 for e in epochs]
val_acc = [0.58 + 0.007*e - np.random.rand()*0.03 for e in epochs]
plt.plot(epochs, train_acc, label='Train Accuracy', linewidth=2)
plt.plot(epochs, val_acc, label='Validation Accuracy', linewidth=2)
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Courbes d\'Accuracy')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
}}}

**Diagnostic :**
- **Overfitting** : Train loss continue de baisser, Val loss remonte
- **Underfitting** : Train et Val loss élevées et stagnent
- **Good fit** : Train et Val loss convergent vers une valeur basse

---

### 4.3 Exemples de Prédictions (Good/Bad Cases)

**Pourquoi l'utiliser ?**
- **Rend le modèle tangible** pour la direction et les utilisateurs
- Montre les forces et faiblesses du modèle

**Structure pour PowerPoint :**

**Slide "Exemples de Succès" :**
- 3-4 exemples bien classés avec score de confiance élevé
- Légende : Email, vraie classe, prédiction, confiance

**Slide "Cas d'Erreurs" :**
- 3-4 exemples mal classés
- Analyse : Pourquoi le modèle s'est trompé ?

{{{python
# Exemple pour classification d'emails
examples = [
    {"text": "URGENT : Problème critique client VIP", 
     "true": "Urgent", "pred": "Urgent", "conf": 0.95, "correct": True},
    {"text": "Réunion hebdomadaire lundi 10h", 
     "true": "Normal", "pred": "Normal", "conf": 0.88, "correct": True},
    {"text": "Merci pour votre réponse rapide", 
     "true": "Normal", "pred": "Important", "conf": 0.62, "correct": False},
]

# Visualisation pour PowerPoint (tableau ou cards)
for ex in examples:
    color = 'green' if ex['correct'] else 'red'
    print(f"[{color.upper()}] {ex['text'][:50]}...")
    print(f"  Vraie classe: {ex['true']} | Prédiction: {ex['pred']} | Confiance: {ex['conf']:.0%}\n")
}}}

---

### 4.4 Feature Importance (Importance des Variables)

**Pourquoi l'utiliser ?**
- Explique **quelles variables** influencent le modèle
- Utile pour les modèles type Random Forest, XGBoost
- Crédibilise le modèle auprès des experts métier

{{{python
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

# Exemple avec Random Forest
feature_names = ['Longueur email', 'Nb mots-clés urgent', 'Heure réception', 
                 'Expéditeur VIP', 'Pièces jointes']
importances = [0.35, 0.28, 0.18, 0.12, 0.07]

plt.figure(figsize=(10, 6))
plt.barh(feature_names, importances, color='steelblue')
plt.xlabel('Importance')
plt.title('Importance des Variables dans le Modèle')
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.show()
}}}

**Pour PowerPoint :**
- Top 10 features les plus importantes
- Aide à valider que le modèle utilise des variables pertinentes

---

### 4.5 Calibration Plot (Courbe de Calibration)

**Pourquoi l'utiliser ?**
- Vérifie si les **probabilités prédites** sont fiables
- Exemple : Si le modèle prédit 70% de confiance, est-ce qu'il a raison 70% du temps ?

{{{python
from sklearn.calibration import calibration_curve

prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=10)

plt.figure(figsize=(10, 8))
plt.plot(prob_pred, prob_true, marker='o', linewidth=2, label='Modèle')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Parfaitement calibré')
plt.xlabel('Probabilité prédite')
plt.ylabel('Fraction de positifs')
plt.title('Courbe de Calibration')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
}}}

**Interprétation :**
- Courbe proche de la diagonale = modèle bien calibré
- Courbe au-dessus = modèle sous-confiant
- Courbe en-dessous = modèle sur-confiant

---

## 5. Métriques Business et ROI

### 5.1 Temps Économisé

**Formule :**

$$\text{Temps économisé} = (\text{Temps manuel} - \text{Temps avec IA}) \times \text{Volume traité}$$

**Exemple (Classification d'emails) :**
- Temps manuel : 3 min/email
- Temps avec IA : 30 sec/email (vérification)
- Volume : 1000 emails/mois
- **Temps économisé = (3 - 0.5) × 1000 = 2500 min/mois = 42h/mois**

{{{python
# Calculateur ROI
def calculate_time_saved(manual_time_per_item, ai_time_per_item, 
                         volume_per_month, hourly_wage):
    time_saved_per_item = manual_time_per_item - ai_time_per_item
    total_time_saved_hours = (time_saved_per_item * volume_per_month) / 60
    cost_saved = total_time_saved_hours * hourly_wage
    
    return {
        'time_saved_hours': total_time_saved_hours,
        'cost_saved_monthly': cost_saved,
        'cost_saved_yearly': cost_saved * 12
    }

results = calculate_time_saved(
    manual_time_per_item=3,      # 3 min
    ai_time_per_item=0.5,        # 30 sec
    volume_per_month=1000,       # 1000 emails/mois
    hourly_wage=40               # 40€/h
)

print(f"Temps économisé : {results['time_saved_hours']:.0f} heures/mois")
print(f"Coût économisé : {results['cost_saved_yearly']:,.0f}€/an")
}}}

---

### 5.2 Taux d'Automatisation

**Formule :**

$$\text{Taux d'automatisation} = \frac{\text{Nombre de cas traités automatiquement}}{\text{Nombre total de cas}} \times 100$$

**Exemple :**
- 1000 emails traités
- 850 traités automatiquement (confiance > 0.8)
- 150 nécessitent une vérification humaine
- **Taux d'automatisation = 850/1000 = 85%**

**Pour PowerPoint :**
- Graphique en donut montrant le taux d'automatisation
- Objectif : 80%+ pour justifier l'investissement IA

---

### 5.3 Réduction du Taux d'Erreur

**Avant/Après avec IA :**

| Métrique                | Avant IA (Manuel) | Avec IA | Amélioration |
|-------------------------|-------------------|---------|--------------|
| Emails mal classés      | 8%                | 3%      | -62.5%       |
| Temps de traitement     | 3h/jour           | 45min   | -75%         |
| Réclamations clients    | 15/mois           | 4/mois  | -73%         |

**Pour PowerPoint :**
- Tableau comparatif avant/après
- Focus sur les KPIs métier (réclamations, temps, erreurs)

---

## 6. Communication de l'Incertitude

### 6.1 Intervalles de Confiance

**Pourquoi l'utiliser ?**
- Quantifie l'**incertitude** des prédictions
- Critique pour les décisions à fort impact

**Exemple (Régression) :**

{{{python
from sklearn.ensemble import GradientBoostingRegressor
import numpy as np

# Entraînement avec quantile regression
model_lower = GradientBoostingRegressor(loss='quantile', alpha=0.05)
model_upper = GradientBoostingRegressor(loss='quantile', alpha=0.95)

# X_train, y_train : données d'entraînement
# Prédictions avec intervalles de confiance à 90%
# y_pred_lower = model_lower.predict(X_test)
# y_pred_upper = model_upper.predict(X_test)

# Visualisation
plt.figure(figsize=(12, 6))
x = np.arange(len(y_true))
plt.plot(x, y_true, 'o', label='Vraies valeurs', alpha=0.5)
plt.plot(x, y_pred, '-', label='Prédictions', linewidth=2)
# plt.fill_between(x, y_pred_lower, y_pred_upper, alpha=0.3, 
#                  label='Intervalle de confiance 90%')
plt.xlabel('Échantillon')
plt.ylabel('Valeur')
plt.title('Prédictions avec Intervalles de Confiance')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
}}}

---

### 6.2 Scores de Confiance et Seuils de Décision

**Stratégie pour la production :**

| Score de confiance | Action                              |
|--------------------|-------------------------------------|
| > 0.9              | Traitement automatique              |
| 0.7 - 0.9          | Traitement automatique + alerte     |
| 0.5 - 0.7          | Revue humaine obligatoire           |
| < 0.5              | Rejet ou escalade                   |

**Pour PowerPoint :**
- Présente cette stratégie de seuils
- Montre la distribution des scores pour estimer le volume de revue humaine

---

## 7. Présentation des Erreurs (Failure Cases)

### 7.1 Analyse des Types d'Erreurs

**Classification des erreurs :**

1. **Erreurs acceptables** : Cas limites, ambiguïtés
2. **Erreurs critiques** : Mauvaise classification avec forte confiance
3. **Erreurs systématiques** : Pattern récurrent (ex: confusion systématique entre 2 classes)

**Pour PowerPoint :**

**Slide "Analyse des Erreurs" :**
- Camembert : Répartition des types d'erreurs
- Liste des top 3 causes d'erreurs
- Actions correctives envisagées

{{{python
# Analyse des erreurs par catégorie
error_categories = {
    'Emails ambigus (frontière floue)': 45,
    'Manque de contexte': 25,
    'Nouveaux types non vus en train': 15,
    'Erreurs aléatoires': 15
}

plt.figure(figsize=(10, 8))
plt.pie(error_categories.values(), labels=error_categories.keys(), 
        autopct='%1.1f%%', startangle=90, colors=['#ff9999', '#66b3ff', '#99ff99', '#ffcc99'])
plt.title('Répartition des Types d\'Erreurs', fontsize=14)
plt.show()
}}}

---

### 7.2 Exemples d'Erreurs Annotés

**Structure :**

Pour chaque erreur présentée :
1. **Texte de l'email** (ou image pour YOLO)
2. **Vraie classe** vs **Prédiction**
3. **Score de confiance**
4. **Explication** : Pourquoi le modèle s'est trompé ?
5. **Action** : Comment corriger (retraining, ajout de features, etc.) ?

**Exemple (Classification email) :**

Email : "Bonjour, pouvez-vous me rappeler demain ?"
Vraie classe : Important
Prédiction : Normal (confiance 0.68)
Explication : Manque de mots-clés "urgent", phrase courte, pas de deadline précise.
Action : Ajouter feature "demande d'action" détectée par NER.


---

## 8. Comparaison de Modèles

### 8.1 Tableau Comparatif Multi-Critères

**Pour PowerPoint :**

| Modèle           | Accuracy | F1-Score | Temps inférence | Taille modèle | ROI (€/an) |
|------------------|----------|----------|-----------------|---------------|------------|
| Baseline (règles)| 0.72     | 0.65     | 5 ms            | -             | 0          |
| Random Forest    | 0.84     | 0.81     | 15 ms           | 50 MB         | 25 000     |
| BERT fine-tuned  | 0.91     | 0.89     | 80 ms           | 440 MB        | 38 000     |
| **GPT-4 (API)**  | **0.93** | **0.91** | 1200 ms         | -             | 15 000     |

**Recommandation** : BERT fine-tuned (meilleur compromis perf/coût/latence)

---

### 8.2 Graphiques de Comparaison

**Radar Chart (Comparaison multi-dimensionnelle) :**

{{{python
import matplotlib.pyplot as plt
import numpy as np

categories = ['Accuracy', 'Vitesse', 'Coût', 'Maintenabilité', 'Explicabilité']
model1 = [0.84, 0.9, 0.95, 0.85, 0.9]  # Random Forest
model2 = [0.91, 0.5, 0.7, 0.6, 0.4]    # BERT

angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
model1 += model1[:1]
model2 += model2[:1]
angles += angles[:1]

fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
ax.plot(angles, model1, 'o-', linewidth=2, label='Random Forest', color='blue')
ax.fill(angles, model1, alpha=0.25, color='blue')
ax.plot(angles, model2, 'o-', linewidth=2, label='BERT', color='red')
ax.fill(angles, model2, alpha=0.25, color='red')
ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, size=12)
ax.set_ylim(0, 1)
ax.set_title('Comparaison Multi-Critères des Modèles', size=16, pad=20)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
ax.grid(True)
plt.tight_layout()
plt.show()
}}}

---

### 8.3 Trade-off Précision vs Coût/Vitesse

**Scatter Plot :**

{{{python
models = ['Baseline', 'Random Forest', 'BERT', 'GPT-4 API']
f1_scores = [0.65, 0.81, 0.89, 0.91]
costs = [0, 100, 500, 2000]  # Coût annuel d'inférence (€)

plt.figure(figsize=(12, 8))
colors = ['gray', 'green', 'blue', 'red']
sizes = [100, 200, 200, 200]

for i, model in enumerate(models):
    plt.scatter(costs[i], f1_scores[i], s=sizes[i], alpha=0.6, 
                c=colors[i], label=model)
    plt.annotate(model, (costs[i], f1_scores[i]), 
                 fontsize=12, ha='center', xytext=(0, 15),
                 textcoords='offset points')

plt.xlabel('Coût annuel d\'inférence (€)', fontsize=14)
plt.ylabel('F1-Score', fontsize=14)
plt.title('Trade-off Performance vs Coût', fontsize=16)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=12)
plt.tight_layout()
plt.show()
}}}

---

## 9. Adaptation selon l'Audience

### 9.1 Pour la Direction (Non-Experts Techniques)

**À privilégier :**
- ✅ **Métriques business** : Temps économisé, ROI, taux d'automatisation
- ✅ **Graphiques simples** : Barres, camemberts, avant/après
- ✅ **Exemples concrets** : "Le modèle détecte 9 emails urgents sur 10"
- ✅ **Comparaison avec baseline** : Montrer l'amélioration vs situation actuelle
- ✅ **Visualisations de résultats** : Matrice de confusion avec couleurs

**À éviter :**
- ❌ Formules mathématiques complexes
- ❌ Jargon technique (overfitting, hyperparamètres, etc.)
- ❌ Trop de métriques (s'en tenir à 2-3 max)

**Exemple de slide pour direction :**

**Titre : "Résultats du Modèle de Classification d'Emails"**
- 📊 Accuracy : 91% (vs 72% avec règles manuelles)
- ⏱️ Temps économisé : 42h/mois = 50 000€/an
- 🤖 Taux d'automatisation : 85%
- ⭐ Satisfaction utilisateurs : +35%

---

### 9.2 Pour les Utilisateurs Métier

**À privilégier :**
- ✅ **Exemples réels** : Montrer des emails/images qu'ils reconnaissent
- ✅ **Cas d'usage** : "Comment le modèle m'aide au quotidien ?"
- ✅ **Transparence sur les limites** : "Quand le modèle a besoin d'aide humaine"
- ✅ **Feedback loop** : "Comment signaler une erreur ?"

**À éviter :**
- ❌ Métriques techniques (AUC, F1, mAP)
- ❌ Comparaison de modèles (peu pertinent pour eux)

---

### 9.3 Pour les Experts IA (Chef, Pairs)

**À privilégier :**
- ✅ **Métriques techniques complètes** : Precision, Recall, F1, AUC, mAP
- ✅ **Courbes détaillées** : ROC, Precision-Recall, Learning curves
- ✅ **Détails d'architecture** : Choix du modèle, hyperparamètres, augmentation de données
- ✅ **Analyse approfondie des erreurs** : Failure modes, confusion entre classes
- ✅ **Benchmarks** : Comparaison avec SOTA (state-of-the-art)

**Exemple de slide pour experts :**

**Titre : "Performances du Modèle BERT Fine-tuned"**
- Macro-F1 : 0.89 (vs 0.81 Random Forest)
- AUC-ROC : 0.94
- mAP@0.5 : 0.87 (pour détection YOLO)
- Confusion principale : Classe "Important" vs "Urgent" (15% des erreurs)
- Hyperparamètres : learning_rate=2e-5, batch_size=16, epochs=10

---

## 10. Structure Type de Présentation PowerPoint

### 10.1 Slide 1 : Contexte et Objectif

**Contenu :**
- Problème métier à résoudre
- Objectif du modèle
- Bénéfices attendus

**Exemple :**

Titre : Classification Automatique des Emails Clients
Problème :

    1000+ emails/jour reçus
    Temps de tri manuel : 3 min/email
    Emails urgents parfois ratés

Objectif :
Automatiser la classification en 3 catégories (Normal, Important, Urgent)
Bénéfices :

    Réduction du temps de traitement de 75%
    Priorisation automatique des emails urgents
    Amélioration satisfaction client


---

### 10.2 Slide 2 : Données et Méthode

**Contenu :**
- Description du dataset
- Répartition des classes
- Approche choisie (modèle)

**Exemple :**

Dataset :

    15 000 emails étiquetés manuellement
    Période : Jan 2023 - Déc 2024
    Répartition : Normal (60%), Important (30%), Urgent (10%)

Approche :

    Modèle : BERT fine-tuned (Transformer)
    Split : 70% train, 15% validation, 15% test
    Augmentation de données pour classe "Urgent" (SMOTE)


---

### 10.3 Slide 3 : Résultats Principaux

**Contenu :**
- Métriques clés (2-3 max pour la direction)
- Comparaison avec baseline
- Graphique de performance

**Exemple (Direction) :**

Résultats :
✅ Accuracy : 91% (vs 72% règles manuelles)
✅ F1-Score : 0.89
✅ Emails urgents détectés : 93% (vs 68% avant)
[Graphique : Barres comparatives Baseline vs Modèle IA]


**Exemple (Experts IA) :**

Performances :

    Macro-F1 : 0.89
    AUC-ROC : 0.94
    Precision/Recall par classe :
        Normal : 0.95 / 0.93
        Important : 0.87 / 0.88
        Urgent : 0.82 / 0.85

[Graphique : Matrice de confusion + Courbe ROC]


---

### 10.4 Slide 4 : Analyse Détaillée (Optionnel pour Direction)

**Contenu :**
- Matrice de confusion
- Exemples de prédictions
- Distribution des scores de confiance

**Pour Direction :**

Où le modèle excelle :
✅ Détection d'emails avec mots-clés urgents ("URGENT", "ASAP")
✅ Emails de clients VIP automatiquement priorisés
Points d'attention :
⚠️ Emails ambigus nécessitent validation humaine (15%)
⚠️ Nouveaux types d'urgence non vus en entraînement
[Graphique : Exemples d'emails bien/mal classés]


---

### 10.5 Slide 5 : Impact Business et ROI

**Contenu :**
- Temps économisé
- Coûts évités
- Amélioration des KPIs métier

**Exemple :**

Impact Business :
💰 Temps économisé : 42h/mois = 50 000€/an
📈 Taux d'automatisation : 85%
📉 Réclamations clients : -65%
⭐ Satisfaction utilisateurs : +35%
ROI :

    Coût développement + infra : 30 000€
    Économies annuelles : 50 000€
    Retour sur investissement : 6 mois

[Graphique : Évolution du ROI sur 2 ans]


---

### 10.6 Slide 6 : Prochaines Étapes et Améliorations

**Contenu :**
- Plan de déploiement
- Améliorations futures
- Monitoring et maintenance

**Exemple :**

      Déploiement :
      ✅ Phase pilote : 100 emails/jour (2 semaines)
      ✅ Feedback utilisateurs + ajustements
      ✅ Déploiement complet : T1 2025
      Améliorations prévues :
      
          Intégration de l'historique client (contexte)
          Détection de sentiment (positif/négatif)
          Support multilingue (anglais + espagnol)
      
      Monitoring :
      
          Dashboard temps réel des performances
          Alertes si accuracy < 85%
          Retraining mensuel avec nouveaux labels

---

## 11. Checklist Avant Présentation

### ✅ Préparation Contenu

- [ ] **Audience identifiée** : Direction / Utilisateurs / Experts ?
- [ ] **Métriques adaptées** : Business pour direction, techniques pour experts
- [ ] **Graphiques clairs** : Titres, légendes, unités, couleurs cohérentes
- [ ] **Exemples concrets** : Au moins 2-3 cas réels parlants
- [ ] **Comparaison baseline** : Toujours montrer l'amélioration
- [ ] **Limitations mentionnées** : Transparence sur les cas d'échec

### ✅ Aspects Visuels PowerPoint

- [ ] **Palette de couleurs cohérente** : 3-4 couleurs max
- [ ] **Taille de police lisible** : Titres ≥ 24pt, texte ≥ 18pt
- [ ] **Graphiques haute résolution** : Export PNG 300 DPI minimum
- [ ] **Animations limitées** : Éviter les effets distrayants
- [ ] **Slides numérotées** : Facilite les questions

### ✅ Storytelling

- [ ] **Fil rouge clair** : Problème → Solution → Résultats → Impact
- [ ] **Chiffres marquants** : Temps économisé, ROI, amélioration %
- [ ] **Call to action** : Prochaines étapes, décisions attendues

---

## 12. Ressources et Outils Pratiques

### 12.1 Bibliothèques Python Essentielles

{{{python
# Installation des outils principaux
pip install scikit-learn matplotlib seaborn pandas numpy
pip install plotly  # Graphiques interactifs
pip install yellowbrick  # Visualisations ML
pip install shap  # Explainabilité des modèles
}}}

### 12.2 Templates de Code Réutilisables

**Fonction tout-en-un pour métriques de classification :**

{{{python
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, confusion_matrix, 
                             classification_report)
import seaborn as sns
import matplotlib.pyplot as plt

def evaluate_classification(y_true, y_pred, y_proba=None, class_names=None):
    """
    Évaluation complète d'un modèle de classification.
    
    Args:
        y_true: Vraies étiquettes
        y_pred: Prédictions
        y_proba: Probabilités prédites (optionnel, pour AUC)
        class_names: Noms des classes (optionnel)
    """
    print("="*60)
    print("MÉTRIQUES DE CLASSIFICATION")
    print("="*60)
    
    # Métriques de base
    print(f"\nAccuracy       : {accuracy_score(y_true, y_pred):.3f}")
    print(f"Precision      : {precision_score(y_true, y_pred, average='weighted'):.3f}")
    print(f"Recall         : {recall_score(y_true, y_pred, average='weighted'):.3f}")
    print(f"F1-Score       : {f1_score(y_true, y_pred, average='weighted'):.3f}")
    
    if y_proba is not None:
        print(f"AUC-ROC        : {roc_auc_score(y_true, y_proba, multi_class='ovr'):.3f}")
    
    # Rapport détaillé
    print("\n" + "="*60)
    print("RAPPORT PAR CLASSE")
    print("="*60)
    print(classification_report(y_true, y_pred, target_names=class_names))
    
    # Matrice de confusion
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('Vraie classe')
    plt.xlabel('Classe prédite')
    plt.title('Matrice de Confusion')
    plt.tight_layout()
    plt.show()
    
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted'),
        'recall': recall_score(y_true, y_pred, average='weighted'),
        'f1': f1_score(y_true, y_pred, average='weighted')
    }

# Exemple d'utilisation
# metrics = evaluate_classification(y_test, y_pred, y_proba, 
#                                   class_names=['Normal', 'Important', 'Urgent'])
}}}

---

## 13. Erreurs Courantes à Éviter

### ❌ Erreur 1 : Utiliser uniquement l'Accuracy avec classes déséquilibrées

**Problème :**  
Avec 95% de classe majoritaire, un modèle prédisant toujours cette classe aura 95% d'accuracy mais sera inutile.

**Solution :**  
Utiliser F1-Score, Balanced Accuracy, ou AUC-ROC.

---

### ❌ Erreur 2 : Ignorer le trade-off Precision vs Recall

**Problème :**  
Optimiser seulement la Precision ou seulement le Recall sans considérer le contexte métier.

**Solution :**  
Définir la priorité avec les métiers : préfère-t-on éviter les FP ou les FN ?

---

### ❌ Erreur 3 : Présenter trop de métriques à la direction

**Problème :**  
Noyer la direction sous 10 métriques techniques (AUC, F1, mAP, Recall, etc.).

**Solution :**  
S'en tenir à 2-3 métriques simples + impact business (ROI, temps économisé).

---

### ❌ Erreur 4 : Ne pas montrer les limitations du modèle

**Problème :**  
Présenter uniquement les succès, cacher les échecs.

**Solution :**  
Transparence sur les failure cases et plan d'amélioration.

---

### ❌ Erreur 5 : Oublier la comparaison avec la baseline

**Problème :**  
Présenter "Accuracy = 85%" sans contexte → difficile d'évaluer si c'est bon.

**Solution :**  
Toujours comparer avec situation actuelle (règles manuelles, ancien modèle, aléatoire).

---

## 14. Glossaire des Termes Clés

| Terme               | Définition Simple                                                                 |
|---------------------|-----------------------------------------------------------------------------------|
| **Accuracy**        | Proportion de prédictions correctes                                               |
| **Precision**       | Proportion de prédictions positives qui sont correctes                            |
| **Recall**          | Proportion de cas positifs détectés par le modèle                                 |
| **F1-Score**        | Moyenne harmonique de Precision et Recall                                         |
| **AUC-ROC**         | Capacité du modèle à discriminer entre classes (0.5=aléatoire, 1=parfait)         |
| **IoU**             | Chevauchement entre boîte prédite et réelle (détection d'objets)                  |
| **mAP**             | Métrique principale pour évaluer la détection d'objets                            |
| **Overfitting**     | Modèle trop spécialisé sur les données d'entraînement (mauvaise généralisation)   |
| **Baseline**        | Modèle de référence simple (règles, aléatoire, ancien système)                    |
| **Confusion Matrix**| Tableau croisant prédictions et vraies valeurs                                    |
| **False Positive**  | Prédiction positive incorrecte (alarme injustifiée)                               |
| **False Negative**  | Prédiction négative incorrecte (cas raté)                                         |
| **ROI**             | Retour sur investissement (gain financier de l'IA vs coût)                        |

---

## 15. Conclusion et Best Practices

### 🎯 Règles d'Or pour Présenter des Résultats IA

1. **Connaître son audience** : Adapter le niveau de détail technique
2. **Raconter une histoire** : Problème → Solution → Impact
3. **Montrer, ne pas juste dire** : Exemples concrets > chiffres abstraits
4. **Être transparent** : Montrer les limites et les failure cases
5. **Quantifier l'impact business** : Toujours traduire en temps/coût/ROI
6. **Comparer avec la baseline** : Montrer l'amélioration
7. **Simplifier sans déformer** : Vulgariser sans perdre l'essence
8. **Préparer les questions** : Anticiper "Et si...?" de la direction
9. **Itérer** : Demander du feedback et améliorer les présentations
10. **Documenter** : Garder une trace écrite pour référence future

### 📚 Pour Aller Plus Loin

**Papers & Ressources :**
- [Scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html) : Documentation des métriques
- [Papers With Code](https://paperswithcode.com/) : Benchmarks et SOTA par tâche
- [Distill.pub](https://distill.pub/) : Visualisations pédagogiques de concepts ML
- [Google PAIR Explorables](https://pair.withgoogle.com/explorables/) : Interactifs pour comprendre les concepts ML

**Outils de Visualisation :**
- [Plotly](https://plotly.com/python/) : Graphiques interactifs pour dashboards
- [Weights & Biases](https://wandb.ai/) : Tracking d'expériences et visualisations
- [TensorBoard](https://www.tensorflow.org/tensorboard) : Visualisation training deep learning
