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

### 2.2 AP (Average Precision)

**Définition :**  

L'Average Precision (AP) est l'**aire sous la courbe Precision-Recall** pour une classe donnée. Elle synthétise la performance du modèle sur tous les seuils de confiance possibles.

**Calcul en étapes :**

1. **Trier** toutes les détections par score de confiance décroissant
2. **Pour chaque détection**, calculer Precision et Recall cumulés :
   - Si IoU ≥ seuil → TP (Vrai Positif)
   - Sinon → FP (Faux Positif)
3. **Tracer** la courbe Precision (y) vs Recall (x)
4. **Calculer** l'aire sous cette courbe = AP

**Formule mathématique (interpolation):**

$$\text{AP} = \sum_{k=1}^{N} P(k) \times \Delta R(k)$$

Où :
- $$P(k)$$ = Precision au k-ième seuil
- $$\Delta R(k)$$ = Variation de Recall entre k et k-1

**Pourquoi l'utiliser ?**

- Résume la performance en **une seule valeur** (entre 0 et 1)
- Indépendant du choix d'un seuil de confiance spécifique
- Prend en compte à la fois Precision ET Recall
- Permet de comparer différents modèles objectivement

**Interprétation visuelle :**

La courbe Precision-Recall typique a une forme **décroissante** :
- Au début (Recall faible) : Precision élevée (seulement les détections très confiantes)
- À la fin (Recall élevé) : Precision baisse (on inclut des détections moins confiantes)

**Valeurs typiques :**

- **AP = 1.0** : Modèle parfait (jamais atteint en pratique)
- **AP = 0.8-0.9** : Excellent modèle
- **AP = 0.6-0.8** : Bon modèle
- **AP < 0.5** : Modèle faible pour cette classe

**Exemple de calcul manuel :**

```python
import numpy as np

# Données de détection pour une classe (ex: voiture)
# Format: [score_confiance, is_correct (1=TP, 0=FP)]
detections = [
    (0.95, 1),  # TP
    (0.90, 1),  # TP
    (0.85, 0),  # FP
    (0.80, 1),  # TP
    (0.75, 0),  # FP
    (0.70, 1),  # TP
    (0.65, 0),  # FP
]

# Nombre d'objets réels (ground truth)
n_ground_truth = 5

# Tri par confiance décroissante (déjà fait ici)
detections_sorted = sorted(detections, key=lambda x: x[0], reverse=True)

# Calcul de Precision et Recall à chaque seuil
precisions = []
recalls = []
tp_cumsum = 0
fp_cumsum = 0

for i, (score, is_correct) in enumerate(detections_sorted):
    if is_correct == 1:
        tp_cumsum += 1
    else:
        fp_cumsum += 1
    
    precision = tp_cumsum / (tp_cumsum + fp_cumsum)
    recall = tp_cumsum / n_ground_truth
    
    precisions.append(precision)
    recalls.append(recall)
    
    print(f"Seuil {i+1}: Score={score:.2f}, P={precision:.3f}, R={recall:.3f}")

# Calcul de l'AP (méthode des rectangles)
ap = 0
for i in range(1, len(recalls)):
    delta_recall = recalls[i] - recalls[i-1]
    ap += precisions[i] * delta_recall

print(f"\nAverage Precision (AP) = {ap:.3f}")

# Sortie exemple :
# Seuil 1: Score=0.95, P=1.000, R=0.200
# Seuil 2: Score=0.90, P=1.000, R=0.400
# Seuil 3: Score=0.85, P=0.667, R=0.400
# Seuil 4: Score=0.80, P=0.750, R=0.600
# Seuil 5: Score=0.75, P=0.600, R=0.600
# Seuil 6: Score=0.70, P=0.667, R=0.800
# Seuil 7: Score=0.65, P=0.571, R=0.800
# 
# Average Precision (AP) = 0.627
```

**Visualisation de la courbe Precision-Recall :**

```python
import matplotlib.pyplot as plt

# En utilisant les données de l'exemple précédent
plt.figure(figsize=(8, 6))
plt.plot(recalls, precisions, marker='o', linewidth=2, markersize=8)
plt.fill_between(recalls, precisions, alpha=0.2)
plt.xlabel('Recall', fontsize=12)
plt.ylabel('Precision', fontsize=12)
plt.title(f'Courbe Precision-Recall (AP = {ap:.3f})', fontsize=14)
plt.grid(True, alpha=0.3)
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.show()

# L'aire sous cette courbe (zone colorée) = AP
```

**Différence entre AP et mAP :**

- **AP** : Performance pour **une seule classe** (ex: AP_voiture = 0.85)
- **mAP** : **Moyenne** des AP de toutes les classes (ex: mAP = moyenne(AP_voiture, AP_piéton, AP_vélo))

**Cas particulier - AP interpolée (méthode PASCAL VOC) :**

Pour lisser la courbe, on utilise souvent une interpolation à 11 points :

$$\text{AP} = \frac{1}{11} \sum_{r \in \{0, 0.1, 0.2, ..., 1.0\}} P_{\text{interp}}(r)$$

Où $$P_{\text{interp}}(r) = \max_{r' \geq r} P(r')$$ (maximum de Precision pour Recall ≥ r)

```python
# Méthode d'interpolation à 11 points (PASCAL VOC)
def calculate_ap_11_point(precisions, recalls):
    ap = 0
    for r in np.arange(0, 1.1, 0.1):
        # Trouver la precision max pour recall >= r
        precisions_above_r = [p for p, rec in zip(precisions, recalls) if rec >= r]
        if len(precisions_above_r) > 0:
            ap += max(precisions_above_r)
    return ap / 11

ap_11pt = calculate_ap_11_point(precisions, recalls)
print(f"AP (11-point interpolation) = {ap_11pt:.3f}")
```

**Points clés à retenir :**

✓ AP mesure la qualité globale des détections pour une classe
✓ Plus l'AP est élevée, meilleur est le modèle pour cette classe
✓ AP combine naturellement Precision et Recall
✓ La forme de la courbe P-R indique le comportement du modèle

---

### 2.4 mAP (mean Average Precision)

**Définition :**  

Le **mAP (mean Average Precision)** est la métrique standard pour évaluer les modèles de détection d'objets. Il s'agit de la **moyenne arithmétique des AP de toutes les classes**.

$$\text{mAP} = \frac{1}{N} \sum_{i=1}^{N} \text{AP}_i$$

Où :
- $$N$$ = nombre total de classes
- $$\text{AP}_i$$ = Average Precision pour la classe $$i$$

---

#### **Calcul détaillé du mAP - Exemple complet**

**Contexte :** Modèle de détection avec 3 classes (voiture, piéton, vélo) sur un dataset de validation.

**Étape 1 : Calculer l'AP pour chaque classe individuellement**

```python
import numpy as np

# Fonction pour calculer l'AP d'une classe
def calculate_ap_for_class(detections, ground_truths, iou_threshold=0.5):
    """
    detections: liste de (score_confiance, bbox, image_id)
    ground_truths: liste de (bbox, image_id) pour cette classe
    iou_threshold: seuil IoU pour considérer un TP
    """
    # 1. Trier détections par confiance décroissante
    detections = sorted(detections, key=lambda x: x[0], reverse=True)
    
    # 2. Marquer les ground truths utilisés (1 GT = 1 détection max)
    gt_matched = {img_id: [False] * len(bboxes) 
                  for img_id, bboxes in ground_truths.items()}
    
    tp = np.zeros(len(detections))
    fp = np.zeros(len(detections))
    
    # 3. Pour chaque détection, vérifier si c'est un TP ou FP
    for det_idx, (score, det_bbox, img_id) in enumerate(detections):
        if img_id not in ground_truths:
            fp[det_idx] = 1  # Pas de GT dans cette image
            continue
        
        # Trouver le meilleur match avec un GT
        best_iou = 0
        best_gt_idx = -1
        
        for gt_idx, gt_bbox in enumerate(ground_truths[img_id]):
            if gt_matched[img_id][gt_idx]:
                continue  # GT déjà matché
            
            iou = calculate_iou(det_bbox, gt_bbox)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx
        
        # Déterminer TP ou FP
        if best_iou >= iou_threshold and best_gt_idx >= 0:
            if not gt_matched[img_id][best_gt_idx]:
                tp[det_idx] = 1
                gt_matched[img_id][best_gt_idx] = True
            else:
                fp[det_idx] = 1  # GT déjà pris
        else:
            fp[det_idx] = 1  # IoU trop faible
    
    # 4. Calculer Precision et Recall cumulés
    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)
    
    total_gt = sum(len(bboxes) for bboxes in ground_truths.values())
    
    recalls = tp_cumsum / total_gt
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum)
    
    # 5. Calculer l'AP (méthode COCO - interpolation à tous les points)
    # Ajouter des points sentinelles
    recalls = np.concatenate(([0], recalls, [1]))
    precisions = np.concatenate(([0], precisions, [0]))
    
    # Interpolation monotone décroissante
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])
    
    # Calcul de l'aire sous la courbe
    indices = np.where(recalls[1:] != recalls[:-1])[0] + 1
    ap = np.sum((recalls[indices] - recalls[indices - 1]) * precisions[indices])
    
    return ap, precisions, recalls

# Fonction IoU (rappel)
def calculate_iou(box1, box2):
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])
    
    inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0
```

**Exemple avec données concrètes :**

```python
# Données simulées pour 3 classes

# CLASSE 1: VOITURE
detections_voiture = [
    (0.95, [100, 100, 200, 200], 'img1'),  # TP
    (0.92, [150, 150, 250, 250], 'img1'),  # TP
    (0.88, [50, 50, 120, 120], 'img2'),    # FP (mauvais IoU)
    (0.85, [300, 300, 400, 400], 'img2'),  # TP
    (0.70, [500, 500, 600, 600], 'img3'),  # FP (pas de GT)
]

ground_truths_voiture = {
    'img1': [[105, 105, 205, 205], [155, 155, 255, 255]],
    'img2': [[305, 305, 405, 405]],
    # img3 n'a pas de voiture
}

ap_voiture, _, _ = calculate_ap_for_class(
    detections_voiture, 
    ground_truths_voiture, 
    iou_threshold=0.5
)
print(f"AP Voiture: {ap_voiture:.3f}")  # Exemple: 0.833

# CLASSE 2: PIÉTON
detections_pieton = [
    (0.90, [200, 200, 250, 300], 'img1'),  # TP
    (0.85, [300, 300, 340, 400], 'img1'),  # TP
    (0.75, [100, 100, 130, 180], 'img2'),  # TP
    (0.65, [400, 400, 440, 500], 'img3'),  # FP
]

ground_truths_pieton = {
    'img1': [[205, 205, 255, 305], [305, 305, 345, 405]],
    'img2': [[105, 105, 135, 185]],
}

ap_pieton, _, _ = calculate_ap_for_class(
    detections_pieton, 
    ground_truths_pieton, 
    iou_threshold=0.5
)
print(f"AP Piéton: {ap_pieton:.3f}")  # Exemple: 0.750

# CLASSE 3: VÉLO
detections_velo = [
    (0.88, [100, 100, 180, 200], 'img1'),  # TP
    (0.80, [200, 200, 280, 300], 'img2'),  # FP
    (0.72, [300, 300, 380, 400], 'img3'),  # TP
]

ground_truths_velo = {
    'img1': [[105, 105, 185, 205]],
    'img3': [[305, 305, 385, 405]],
}

ap_velo, _, _ = calculate_ap_for_class(
    detections_velo, 
    ground_truths_velo, 
    iou_threshold=0.5
)
print(f"AP Vélo: {ap_velo:.3f}")  # Exemple: 0.667
```

**Étape 2 : Calculer le mAP**

```python
# Calcul du mAP = moyenne des AP
ap_scores = {
    'voiture': 0.833,
    'piéton': 0.750,
    'vélo': 0.667
}

mAP = np.mean(list(ap_scores.values()))
print(f"\nmAP@0.5 = {mAP:.3f}")  # 0.750

# Détail par classe
print("\nDétail par classe:")
for classe, ap in ap_scores.items():
    print(f"  {classe}: AP = {ap:.3f}")

# Sortie:
# mAP@0.5 = 0.750
# 
# Détail par classe:
#   voiture: AP = 0.833
#   piéton: AP = 0.750
#   vélo: AP = 0.667
```

---

#### **Variantes du mAP : Comprendre les différents seuils IoU**

Le mAP peut être calculé avec **différents seuils IoU**, ce qui change la définition de ce qu'est une "bonne détection".

##### **1. mAP@0.5 (PASCAL VOC)**

**Définition :** Une détection est considérée comme correcte si **IoU ≥ 0.5** avec un ground truth.

**Caractéristiques :**
- Seuil **permissif** (accepte des localisations approximatives)
- Standard pour PASCAL VOC Challenge
- Valeurs typiquement **plus élevées**

**Quand l'utiliser :**
- Applications où la localisation précise n'est pas critique
- Détection d'objets généraux (surveillance, comptage)

```python
# Calcul mAP@0.5
mAP_50 = calculate_map(all_detections, all_ground_truths, iou_threshold=0.5)
print(f"mAP@0.5 = {mAP_50:.3f}")  # Ex: 0.750
```

##### **2. mAP@0.75 (Strict)**

**Définition :** Une détection est considérée comme correcte si **IoU ≥ 0.75** avec un ground truth.

**Caractéristiques :**
- Seuil **exigeant** (nécessite une localisation précise)
- Valeurs typiquement **30-40% inférieures** au mAP@0.5
- Pénalise les détections mal alignées

**Quand l'utiliser :**
- Applications nécessitant une localisation précise (robotique, chirurgie assistée)
- Segmentation d'instance

```python
# Calcul mAP@0.75
mAP_75 = calculate_map(all_detections, all_ground_truths, iou_threshold=0.75)
print(f"mAP@0.75 = {mAP_75:.3f}")  # Ex: 0.520 (plus bas que mAP@0.5)
```

##### **3. mAP@[0.5:0.95] (COCO - Standard actuel)**

**Définition :** Moyenne des mAP calculés pour **10 seuils IoU** : 0.5, 0.55, 0.60, ..., 0.90, 0.95.

$$\text{mAP@[0.5:0.95]} = \frac{1}{10} \sum_{t=0.5}^{0.95} \text{mAP@}t$$

Avec $$t$$ variant par pas de 0.05.

**Caractéristiques :**
- Métrique **la plus complète** et robuste
- Standard pour **MS COCO dataset** (référence actuelle)
- Évalue la qualité de localisation sur un large spectre
- Valeurs typiquement **20-30% inférieures** au mAP@0.5

**Pourquoi c'est mieux :**
- Ne favorise pas les modèles qui "trichent" avec des boîtes imprécises
- Récompense la précision de localisation
- Plus discriminant entre bons et excellents modèles

```python
# Calcul mAP@[0.5:0.95] (méthode COCO)
def calculate_map_coco(detections, ground_truths):
    iou_thresholds = np.arange(0.5, 1.0, 0.05)  # [0.5, 0.55, ..., 0.95]
    map_scores = []
    
    for iou_thresh in iou_thresholds:
        mAP_at_thresh = calculate_map(detections, ground_truths, iou_threshold=iou_thresh)
        map_scores.append(mAP_at_thresh)
        print(f"mAP@{iou_thresh:.2f} = {mAP_at_thresh:.3f}")
    
    map_coco = np.mean(map_scores)
    return map_coco

# Exemple de sortie
# mAP@0.50 = 0.750
# mAP@0.55 = 0.735
# mAP@0.60 = 0.710
# mAP@0.65 = 0.680
# mAP@0.70 = 0.640
# mAP@0.75 = 0.590
# mAP@0.80 = 0.530
# mAP@0.85 = 0.460
# mAP@0.90 = 0.380
# mAP@0.95 = 0.280

map_coco = calculate_map_coco(all_detections, all_ground_truths)
print(f"\nmAP@[0.5:0.95] = {map_coco:.3f}")  # Ex: 0.576
```

**Comparaison visuelle :**

```python
import matplotlib.pyplot as plt

# Données de comparaison
iou_thresholds = np.arange(0.5, 1.0, 0.05)
map_values = [0.750, 0.735, 0.710, 0.680, 0.640, 0.590, 0.530, 0.460, 0.380, 0.280]

plt.figure(figsize=(10, 6))
plt.plot(iou_thresholds, map_values, marker='o', linewidth=2, markersize=8)
plt.axhline(y=np.mean(map_values), color='r', linestyle='--', 
            label=f'mAP@[0.5:0.95] = {np.mean(map_values):.3f}')
plt.xlabel('Seuil IoU', fontsize=12)
plt.ylabel('mAP', fontsize=12)
plt.title('Évolution du mAP selon le seuil IoU', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend()
plt.xlim([0.45, 1.0])
plt.ylim([0, 1])
plt.show()
```

---

#### **Tableau comparatif : Impact du seuil IoU**

| Seuil IoU | Nom | Tolérance | Valeur typique | Usage |
|-----------|-----|-----------|----------------|-------|
| **0.5** | mAP@0.5 | Permissif | 70-85% | PASCAL VOC, détection générale |
| **0.75** | mAP@0.75 | Strict | 40-60% | Localisation précise |
| **0.5:0.95** | mAP COCO | Très strict | 50-70% | Benchmark moderne (COCO) |

**Relation approximative :**
$$\text{mAP@[0.5:0.95]} \approx 0.7 \times \text{mAP@0.5}$$

---

#### **Implémentation pratique avec YOLOv8**

```python
from ultralytics import YOLO
import json

# Charger un modèle pré-entraîné
model = YOLO('yolov8n.pt')  # nano (plus rapide)
# model = YOLO('yolov8s.pt')  # small
# model = YOLO('yolov8m.pt')  # medium
# model = YOLO('yolov8l.pt')  # large
# model = YOLO('yolov8x.pt')  # extra-large

# Validation sur dataset COCO
results = model.val(
    data='coco.yaml',           # Configuration du dataset
    split='val',                # Split de validation
    batch=16,                   # Batch size
    imgsz=640,                  # Taille d'image
    device=0,                   # GPU 0 (ou 'cpu')
    verbose=True,               # Affichage détaillé
    plots=True,                 # Générer des plots
)

# Accès aux métriques globales
print("\n=== MÉTRIQUES GLOBALES ===")
print(f"mAP@0.5       : {results.box.map50:.4f}")      # mAP à IoU=0.5
print(f"mAP@0.75      : {results.box.map75:.4f}")      # mAP à IoU=0.75
print(f"mAP@[0.5:0.95]: {results.box.map:.4f}")        # mAP COCO (principal)

print(f"\nPrecision: {results.box.mp:.4f}")            # Precision moyenne
print(f"Recall   : {results.box.mr:.4f}")              # Recall moyen

# Accès aux métriques par classe
print("\n=== MÉTRIQUES PAR CLASSE ===")
class_names = model.names  # Dictionnaire {id: nom_classe}

for class_id, class_name in class_names.items():
    if class_id < len(results.box.maps):
        ap50 = results.box.maps[class_id]  # AP@0.5 pour cette classe
        print(f"{class_name:15s}: AP@0.5 = {ap50:.4f}")

# Exporter les résultats en JSON
results_dict = {
    'mAP@0.5': float(results.box.map50),
    'mAP@0.75': float(results.box.map75),
    'mAP@[0.5:0.95]': float(results.box.map),
    'precision': float(results.box.mp),
    'recall': float(results.box.mr),
    'per_class_ap': {
        class_names[i]: float(results.box.maps[i]) 
        for i in range(len(results.box.maps))
    }
}

with open('validation_results.json', 'w') as f:
    json.dump(results_dict, f, indent=2)

print("\nRésultats sauvegardés dans 'validation_results.json'")

# Exemple de sortie :
# === MÉTRIQUES GLOBALES ===
# mAP@0.5       : 0.6234
# mAP@0.75      : 0.4521
# mAP@[0.5:0.95]: 0.4789
# 
# Precision: 0.7123
# Recall   : 0.6834
# 
# === MÉTRIQUES PAR CLASSE ===
# person         : AP@0.5 = 0.7234
# bicycle        : AP@0.5 = 0.5821
# car            : AP@0.5 = 0.6945
# ...
```

---

#### **Pourquoi utiliser le mAP ?**

**1. Métrique unifiée**
- Combine Precision, Recall et qualité de localisation (IoU)
- Une seule valeur pour évaluer un modèle complet

**2. Indépendant du seuil de confiance**
- Évalue la performance sur **tous** les seuils possibles
- Pas besoin de choisir arbitrairement un seuil

**3. Permet la comparaison objective**

```python
# Comparaison de modèles
models_comparison = {
    'YOLOv8n': {'mAP@0.5': 0.623, 'mAP@[0.5:0.95]': 0.479},
    'YOLOv8s': {'mAP@0.5': 0.677, 'mAP@[0.5:0.95]': 0.531},
    'YOLOv8m': {'mAP@0.5': 0.721, 'mAP@[0.5:0.95]': 0.582},
    'YOLOv8l': {'mAP@0.5': 0.749, 'mAP@[0.5:0.95]': 0.611},
    'YOLOv8x': {'mAP@0.5': 0.761, 'mAP@[0.5:0.95]': 0.628},
}

import pandas as pd
df = pd.DataFrame(models_comparison).T
print(df)

#          mAP@0.5  mAP@[0.5:0.95]
# YOLOv8n    0.623           0.479
# YOLOv8s    0.677           0.531
# YOLOv8m    0.721           0.582
# YOLOv8l    0.749           0.611
# YOLOv8x    0.761           0.628
```

**4. Détecte les faiblesses par classe**

```python
# Analyser les classes problématiques
class_aps = {
    'person': 0.82,
    'car': 0.78,
    'bicycle': 0.65,
    'dog': 0.58,      # ← Classe faible
    'cat': 0.55,      # ← Classe faible
    'bird': 0.48,     # ← Très faible
}

# Identifier les classes < 0.6
weak_classes = {k: v for k, v in class_aps.items() if v < 0.6}
print("Classes nécessitant plus de données d'entraînement:")
for classe, ap in sorted(weak_classes.items(), key=lambda x: x[1]):
    print(f"  {classe}: AP = {ap:.2f}")

# Sortie:
# Classes nécessitant plus de données d'entraînement:
#   bird: AP = 0.48
#   cat: AP = 0.55
#   dog: AP = 0.58
```

---

#### **Interprétation des valeurs de mAP**

| mAP@0.5 | mAP@[0.5:0.95] | Qualité | Usage recommandé |
|---------|----------------|---------|------------------|
| < 0.3 | < 0.2 | Très faible | Modèle non fonctionnel |
| 0.3 - 0.5 | 0.2 - 0.35 | Faible | Prototype initial |
| 0.5 - 0.7 | 0.35 - 0.50 | Moyen | Développement |
| 0.7 - 0.85 | 0.50 - 0.65 | Bon | Production (applications générales) |
| > 0.85 | > 0.65 | Excellent | Production (applications critiques) |

**Règles empiriques :**
- **mAP@0.5 ≥ 0.7** : Acceptable pour la plupart des applications réelles
- **mAP@[0.5:0.95] ≥ 0.5** : Bon modèle selon standards COCO
- **Écart mAP@0.5 vs mAP@[0.5:0.95]** : Si > 30%, le modèle a des problèmes de localisation précise

---

#### **Visualisation des résultats**

```python
import matplotlib.pyplot as plt
import numpy as np

# Données de comparaison de modèles
models = ['YOLOv5s', 'YOLOv8s', 'Faster R-CNN', 'RetinaNet']
map50 = [0.662, 0.677, 0.689, 0.671]
map50_95 = [0.485, 0.531, 0.542, 0.523]

x = np.arange(len(models))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
bars1 = ax.bar(x - width/2, map50, width, label='mAP@0.5', color='#3b82f6')
bars2 = ax.bar(x + width/2, map50_95, width, label='mAP@[0.5:0.95]', color='#8b5cf6')

ax.set_xlabel('Modèle', fontsize=12)
ax.set_ylabel('mAP', fontsize=12)
ax.set_title('Comparaison des performances de détection', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# Ajouter les valeurs sur les barres
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.show()
```

---

#### **Points clés à retenir**

✓ **mAP = moyenne des AP** de toutes les classes  
✓ **mAP@0.5** = standard permissif (PASCAL VOC)  
✓ **mAP@[0.5:0.95]** = standard moderne et strict (COCO)  
✓ Plus le mAP est **proche de 1.0**, meilleur est le modèle  
✓ Le mAP **seul ne suffit pas** : analyser les AP par classe pour identifier les faiblesses  
✓ **Trade-off vitesse/précision** : YOLOv8n (rapide, mAP faible) vs YOLOv8x (lent, mAP élevé)  

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

```python
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
```

---

### 8.3 Trade-off Précision vs Coût/Vitesse

**Scatter Plot :**

```python
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
```

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

```python
# Installation des outils principaux
pip install scikit-learn matplotlib seaborn pandas numpy
pip install plotly  # Graphiques interactifs
pip install yellowbrick  # Visualisations ML
pip install shap  # Explainabilité des modèles
```

### 12.2 Templates de Code Réutilisables

**Fonction tout-en-un pour métriques de classification :**

```python
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
```

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
