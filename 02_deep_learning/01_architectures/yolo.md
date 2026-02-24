# YOLO (You Only Look Once) - Cours Complet

## 1. Introduction : Pourquoi YOLO ?

**Contexte** : La détection d'objets est une tâche fondamentale en computer vision qui consiste à :
- **Localiser** des objets dans une image (bounding boxes)
- **Classifier** ces objets (chat, chien, humain...)

**Révolution YOLO** : Avant YOLO (2015), les détecteurs (R-CNN, Fast R-CNN) utilisaient une approche en **2 étapes** :
1. Proposer des régions candidates (Region Proposals)
2. Classifier chaque région

→ **Problème** : Lent (plusieurs secondes par image), complexe à entraîner

**Innovation YOLO** : Reformule la détection comme un **problème de régression unique** (one-shot)
- Une seule passe forward dans le réseau
- Prédit directement bounding boxes + classes simultanément
- **Résultat** : 45+ FPS (temps réel) vs < 1 FPS pour R-CNN

---

## 2. Rappel Minimal : CNN & Feature Extraction

### Concepts clés pour comprendre YOLO

**CNN (Convolutional Neural Network)** : Réseau qui extrait des features hiérarchiques
- **Premières couches** : Détectent des patterns simples (bords, textures)
- **Couches intermédiaires** : Patterns complexes (formes, parties d'objets)
- **Couches finales** : Représentations sémantiques de haut niveau

**Opérations essentielles** :
- **Convolution** : Extraction de features avec des filtres
- **Pooling** : Réduction de dimension (downsampling)
- **Feature maps** : Représentations spatiales à différentes échelles

**Pourquoi c'est important pour YOLO ?**
Le backbone CNN de YOLO extrait des feature maps qui contiennent l'information spatiale ET sémantique nécessaire pour localiser et classifier les objets.

---

## 3. Architecture YOLOv1 : Les Fondations

### 3.1 Principe de Base

**Idée centrale** : Diviser l'image en une grille $\$S \times S$$ (ex: 7×7)

Chaque cellule de la grille est responsable de :
1. Prédire $\$B$$ bounding boxes (généralement B=2)
2. Calculer un **confidence score** par box
3. Prédire $\$C$$ probabilités de classes (conditionnelles)

**Pipeline** :
Image (448×448)
→ Backbone CNN (24 conv layers)
→ Feature map
→ Fully Connected Layers
→ Tensor de sortie (S × S × (B×5 + C))

### 3.2 Format de Sortie

Pour chaque cellule de la grille, le réseau prédit un vecteur de dimension $\$B \times 5 + C$$ :

**Par bounding box (5 valeurs)** :
- $$x, y$$ : Coordonnées du centre (relatives à la cellule)
- $$w, h$$ : Largeur et hauteur (relatives à l'image entière)
- $$\text{confidence}$$ : Score de confiance

**Confidence score** :
$$\text{Confidence} = P(\text{Object}) \times \text{IoU}_{\text{pred}}^{\text{truth}}$$

- $\$P(\text{Object})$$ : Probabilité qu'un objet existe dans cette box
- $$\text{IoU}$$ (Intersection over Union) : Qualité de la localisation

**Probabilités de classes** (C valeurs, partagées entre les B boxes) :
$\$P(C_i | \text{Object})$$ pour chaque classe $$i$$

**Prédiction finale** :
$\$P(C_i) \times \text{Confidence} = P(C_i | \text{Object}) \times P(\text{Object}) \times \text{IoU}$$

### 3.3 Exemple Concret

Configuration YOLOv1 : $\$S=7, B=2, C=20$$ (PASCAL VOC)
- Grille : 7×7 = 49 cellules
- Par cellule : 2×5 + 20 = **30 valeurs**
- **Sortie totale** : Tensor 7×7×30

Pour une image avec un chien au centre :
- La cellule (3,3) aurait $\$P(\text{Object}) \approx 1$$
- $$x, y$$ proche de (0.5, 0.5) (centre de la cellule)
- $$w, h$$ proportionnel à la taille du chien
- $\$P(\text{Chien}|\text{Object}) \approx 1$$, autres classes ≈ 0

---

## 4. Mathématiques : Loss Function

### 4.1 Pourquoi cette Loss ?

La loss de YOLO doit **pénaliser simultanément** :
1. Erreurs de localisation (coordonnées $$x, y, w, h$$)
2. Erreurs de confidence (présence/absence d'objet)
3. Erreurs de classification

**Problème** : Ces erreurs ne sont pas de même magnitude
- Localisation : Valeurs continues entre 0 et 1
- Classification : Probabilités
- Boxes vides vs boxes pleines : Déséquilibre massif

**Solution** : Loss pondérée avec coefficients $$\lambda$$

### 4.2 Loss Complète (YOLOv1)

$$
\begin{align}
\mathcal{L} = & \lambda_{\text{coord}} \sum_{i=0}^{S^2} \sum_{j=0}^{B} \mathbb{1}_{ij}^{\text{obj}} \left[ (x_i - \hat{x}_i)^2 + (y_i - \hat{y}_i)^2 \right] \\
& + \lambda_{\text{coord}} \sum_{i=0}^{S^2} \sum_{j=0}^{B} \mathbb{1}_{ij}^{\text{obj}} \left[ (\sqrt{w_i} - \sqrt{\hat{w}_i})^2 + (\sqrt{h_i} - \sqrt{\hat{h}_i})^2 \right] \\
& + \sum_{i=0}^{S^2} \sum_{j=0}^{B} \mathbb{1}_{ij}^{\text{obj}} (C_i - \hat{C}_i)^2 \\
& + \lambda_{\text{noobj}} \sum_{i=0}^{S^2} \sum_{j=0}^{B} \mathbb{1}_{ij}^{\text{noobj}} (C_i - \hat{C}_i)^2 \\
& + \sum_{i=0}^{S^2} \mathbb{1}_{i}^{\text{obj}} \sum_{c \in \text{classes}} (p_i(c) - \hat{p}_i(c))^2
\end{align}
$$

### 4.3 Décryptage Terme par Terme

**Notation** :
- $$\mathbb{1}_{ij}^{\text{obj}}$$ : Indicatrice = 1 si la box $$j$$ de la cellule $$i$$ est responsable d'un objet
- $$\mathbb{1}_{ij}^{\text{noobj}}$$ : Indicatrice = 1 si la box ne contient pas d'objet
- $$\hat{x}, \hat{y}, \hat{w}, \hat{h}$$ : Prédictions du réseau
- $$x, y, w, h$$ : Ground truth
- $$\lambda_{\text{coord}} = 5$$ : Amplifier l'importance de la localisation
- $$\lambda_{\text{noobj}} = 0.5$$ : Réduire l'impact des boxes vides (nombreuses)

**Ligne 1** : Loss de localisation (centre)
- MSE sur $$x, y$$
- Uniquement pour les boxes "responsables"

**Ligne 2** : Loss de dimensions
- MSE sur $$\sqrt{w}, \sqrt{h}$$
- **Pourquoi la racine carrée ?** Problème d'échelle !
  - Erreur de 2px sur une petite box (10px) : Très grave
  - Erreur de 2px sur une grande box (100px) : Négligeable
  - La racine carrée **réduit la pénalité** pour les grandes boxes

**Ligne 3** : Loss de confidence (avec objet)
- MSE entre confidence prédite et IoU réelle
- $$\hat{C}_i$$ devrait tendre vers $$\text{IoU}_{\text{pred}}^{\text{truth}}$$

**Ligne 4** : Loss de confidence (sans objet)
- MSE avec coefficient réduit ($$\lambda_{\text{noobj}} = 0.5$$)
- **Pourquoi réduire ?** La majorité des boxes sont vides → équilibrer la loss

**Ligne 5** : Loss de classification
- MSE sur les probabilités de classes
- Uniquement si un objet existe dans la cellule

### 4.4 Assignation de Responsabilité

**Question cruciale** : Quelle box est "responsable" d'un objet ?

**Règle** : Parmi les $\$B$$ boxes d'une cellule, celle avec l'**IoU maximale** avec la ground truth

**Conséquence** : 
- Une seule box par cellule reçoit le signal de gradient pour la localisation
- Les autres boxes apprennent seulement à prédire "pas d'objet"

---

## 5. Métriques : IoU (Intersection over Union)

### 5.1 Définition

$$\text{IoU} = \frac{\text{Aire}(B_{\text{pred}} \cap B_{\text{gt}})}{\text{Aire}(B_{\text{pred}} \cup B_{\text{gt}})}$$

- $\$B_{\text{pred}}$$ : Bounding box prédite
- $\$B_{\text{gt}}$$ : Ground truth box
- $$\cap$$ : Intersection (zone de chevauchement)
- $$\cup$$ : Union (zone totale couverte)

**Interprétation** :
- IoU = 0 : Aucun chevauchement
- IoU = 1 : Boxes parfaitement alignées
- IoU ≥ 0.5 : Généralement considéré comme "bonne" détection

### 5.2 Pourquoi IoU plutôt que distance euclidienne ?

**Avantage** : Invariant à l'échelle
- Deux boxes de 10×10 avec 5px de décalage : IoU ≈ 0.4
- Deux boxes de 100×100 avec 5px de décalage : IoU ≈ 0.9
→ Pénalité proportionnelle à la taille

---

## 6. Post-Processing : Non-Maximum Suppression (NMS)

### 6.1 Problème

YOLO prédit $\$S^2 \times B$$ bounding boxes (ex: 7×7×2 = 98 boxes)
→ Plusieurs boxes peuvent détecter le **même objet**

### 6.2 Algorithme NMS

**Objectif** : Garder uniquement la "meilleure" box par objet

**Étapes** :
1. Filtrer les boxes avec $$\text{confidence} < \text{seuil}$$ (ex: 0.3)
2. Pour chaque classe :
   - Trier les boxes par confidence décroissant
   - Garder la box avec le score le plus élevé
   - Supprimer toutes les boxes avec $$\text{IoU} > \text{seuil}_{\text{NMS}}$$ (ex: 0.5) par rapport à cette box
   - Répéter jusqu'à épuisement

**Pourquoi ça marche ?**
- Les boxes qui se chevauchent fortement (IoU élevé) détectent probablement le même objet
- On garde celle avec le plus haut confidence

**Hyperparamètres clés** :
- $$\text{seuil}_{\text{conf}}$$ : Trade-off précision/rappel
- $$\text{seuil}_{\text{NMS}}$$ : Agressivité de la suppression (0.5 typique)

---

## 7. Évolution : YOLOv2 (YOLO9000)

### 7.1 Limitations de YOLOv1

1. **Précision médiocre** sur petits objets
2. **Localisation imprécise** (grille grossière 7×7)
3. **Une seule échelle** de détection
4. **Limite d'objets par cellule** (max B=2)

### 7.2 Améliorations YOLOv2

#### **Batch Normalization**
- Ajouté après chaque couche conv
- **Impact** : +2% mAP, convergence plus rapide

#### **Résolution plus élevée**
- Entraînement à 448×448 (vs 224×224)
- **Pourquoi ?** Plus de détails spatiaux

#### **Anchor Boxes** (révolution majeure)

**Concept** : Prédéfinis des "boîtes de référence" avec ratios/tailles variés

**Différence avec YOLOv1** :
- v1 : Prédit $$w, h$$ directement (valeurs arbitraires)
- v2 : Prédit des **offsets** par rapport aux ancres

**Mathématiques** :
$$b_x = \sigma(t_x) + c_x$$
$$b_y = \sigma(t_y) + c_y$$
$$b_w = p_w \cdot e^{t_w}$$
$$b_h = p_h \cdot e^{t_h}$$

Où :
- $$t_x, t_y, t_w, t_h$$ : Offsets prédits par le réseau
- $$c_x, c_y$$ : Coordonnées de la cellule
- $$p_w, p_h$$ : Dimensions de l'anchor box
- $$\sigma$$ : Fonction sigmoïde (contraindre $$x, y$$ dans la cellule)

**Pourquoi des ancres ?**
- Facilite l'apprentissage (prédire des ajustements plutôt que des valeurs absolues)
- Meilleures performances sur objets de tailles variées
- **K-means clustering** sur le dataset pour définir les ancres optimales

#### **Multi-Scale Feature Maps** (passthrough layer)
- Concat features haute résolution (26×26) avec basse résolution (13×13)
- **Impact** : +1% sur petits objets

#### **Grille plus fine**
- 13×13 au lieu de 7×7
- **Impact** : Plus de cellules = meilleure localisation

**Résultats** : 76.8 mAP (PASCAL VOC 2007) à 67 FPS

---

## 8. Évolution : YOLOv3

### 8.1 Innovation Majeure : Feature Pyramid Network (FPN)

**Problème multi-échelle** : Détecter simultanément :
- Petits objets (chat lointain)
- Gros objets (humain au premier plan)

**Solution** : Prédictions à **3 échelles différentes**

**Architecture** :
1. **Backbone** : Darknet-53 (53 couches conv)
2. **Détections à 3 niveaux** :
   - **Fine** (52×52) : Petits objets, features haute résolution
   - **Medium** (26×26) : Objets moyens
   - **Coarse** (13×13) : Gros objets, réceptive field large

**Pourquoi 3 échelles ?**
- **52×52** : Préserve détails spatiaux (petits objets nécessitent précision)
- **13×13** : Contexte global (gros objets bénéficient du contexte)

**Ancres par échelle** : 3 ancres × 3 échelles = **9 ancres totales**
- Définies par K-means sur le dataset
- Assignées aux échelles par taille

### 8.2 Changements dans la Loss

**Objectness vs Confidence** :
- Nouvelle formulation avec **Binary Cross-Entropy** au lieu de MSE
- Prédit séparément :
  - $$\text{objectness}$$ : Probabilité qu'un objet existe
  - $$\text{class probabilities}$$ : Probabilités de classes

$$\mathcal{L}_{\text{obj}} = -\sum \left[ y_i \log(\hat{y}_i) + (1-y_i) \log(1-\hat{y}_i) \right]$$

**Pourquoi BCE ?** Plus adapté pour des probabilités (optimisation plus stable)

### 8.3 Multi-Label Classification

- Passage de **Softmax à Sigmoïdes indépendants**
- **Pourquoi ?** Permet de détecter plusieurs labels non-mutuellement exclusifs
  - Ex: "Femme" + "Personne" simultanément

**Résultats** : 57.9 mAP (COCO) à 30 FPS

---

## 9. Versions Modernes : YOLOv5 & YOLOv8

### 9.1 YOLOv5 (Ultralytics, 2020)

**Améliorations pratiques** :
- **Architecture modulaire** : Focus Layer, CSP (Cross Stage Partial)
- **Auto-anchor** : Calcul automatique des ancres optimales
- **Mosaic augmentation** : Combine 4 images en une seule (diversité++)
- **Label smoothing** : Régularisation (évite overconfidence)

**Variantes** : YOLOv5n/s/m/l/x (Nano à Extra-large)
- Trade-off vitesse/précision

### 9.2 YOLOv8 (2023)

**Architecture décomplexifiée** :
- **Anchor-free** : Plus besoin d'ancres prédéfinies !
  - Prédit directement centre + dimensions
  - Simplifie l'entraînement, réduit hyperparamètres
  
- **Decoupled head** : Têtes séparées pour classification et régression
  - **Pourquoi ?** Tâches différentes → représentations optimisées indépendamment
  
**Nouvelles loss functions** :
- **CIoU loss** (Complete IoU) pour la localisation :
$$\mathcal{L}_{\text{CIoU}} = 1 - \text{IoU} + \frac{\rho^2(b, b^{gt})}{c^2} + \alpha v$$

Où :
- $$\rho$$ : Distance euclidienne entre centres
- $$c$$ : Diagonale du plus petit rectangle englobant
- $$v$$ : Mesure de similarité de ratio d'aspect
- **Impact** : Pénalise non seulement l'IoU mais aussi distance et forme

**Résultats** : State-of-the-art mAP avec latence réduite

---

## 10. Comparaison : YOLO vs Autres Architectures

| Architecture | Approche | Vitesse | Précision | Complexité |
|--------------|----------|---------|-----------|------------|
| **R-CNN** | 2-stage (proposals + classification) | Très lente (~2s/img) | Bonne | Haute |
| **Fast R-CNN** | Shared CNN + RoI pooling | Lente (~0.5s/img) | Bonne | Moyenne |
| **Faster R-CNN** | RPN (Region Proposal Network) | Moyenne (~0.2s/img) | Excellente | Haute |
| **SSD** | Multi-scale predictions | Rapide (~30 FPS) | Moyenne | Moyenne |
| **YOLO** | Single-shot regression | **Très rapide (45+ FPS)** | Bonne à Excellente | Basse |

### Pourquoi YOLO est plus rapide ?

**R-CNN family** :
- Génère ~2000 région proposals
- Classifie **chaque région indépendamment**
- $\$O(n_{proposals} \times \text{coût}_{\text{CNN}})$$

**YOLO** :
- **Une seule passe** forward
- Prédit toutes les boxes simultanément
- $\$O(1 \times \text{coût}_{\text{CNN}})$$

**Trade-off historique** :
- YOLO sacrifiait précision pour vitesse (v1-v3)
- YOLOv5+ atteint des performances comparables à Faster R-CNN **tout en restant temps réel**

---

## 11. Aspects Pratiques : Entraîner YOLO

### 11.1 Stratégies d'Entraînement

#### **Transfer Learning** (recommandé)
- Partir d'un modèle pré-entraîné (ImageNet ou COCO)
- **Fine-tuner** sur ton dataset (chiens/chats/humains)
- **Pourquoi ?** Features bas-niveau déjà apprises (bords, textures)

**Stratégie de fine-tuning** :
1. **Freeze** les premières couches (backbone)
2. Entraîner uniquement la tête (détection head)
3. **Unfreeze** progressivement les couches

#### **From Scratch** (si dataset massif > 100k images)
- Nécessite beaucoup de données
- Entraînement plus long (jours/semaines)

### 11.2 Data Augmentation

**Augmentations spatiales** :
- Random scaling, rotation, translation
- Flips horizontaux
- **Mosaic** : Combine 4 images (force à apprendre contexte varié)

**Augmentations photométriques** :
- Ajustement brightness/contrast/saturation
- Blur, noise

**Pourquoi c'est crucial ?**
Détection nécessite invariance à :
- Échelle (chat proche vs lointain)
- Position (centre vs bord)
- Conditions lumineuses

### 11.3 Hyperparamètres Clés

| Paramètre | Valeur typique | Impact |
|-----------|----------------|--------|
| **Learning rate** | 0.01 (début) → 0.0001 (fin) | Convergence |
| **Batch size** | 16-64 | Stabilité gradient (compromis GPU) |
| **Epochs** | 100-300 | Trade-off temps/performance |
| **IoU threshold (NMS)** | 0.45-0.5 | Agressivité suppression |
| **Confidence threshold** | 0.25-0.5 | Précision vs Rappel |
| **Image size** | 640×640 (v8) | Trade-off vitesse/précision |

### 11.4 Métriques d'Évaluation

**mAP (mean Average Precision)** :
$$\text{mAP} = \frac{1}{C} \sum_{i=1}^{C} \text{AP}_i$$

Où :
- $\$C$$ : Nombre de classes
- $$\text{AP}_i$$ : Average Precision pour la classe $$i$$

**Précision & Rappel** :
$$\text{Precision} = \frac{TP}{TP + FP}$$
$$\text{Recall} = \frac{TP}{TP + FN}$$

**Courbe P-R** : Trace Precision vs Recall pour différents seuils
- **AP** : Aire sous la courbe P-R

**mAP@0.5** vs **mAP@0.5:0.95** :
- @0.5 : IoU ≥ 0.5 considéré correct
- @0.5:0.95 : Moyenne sur IoU de 0.5 à 0.95 (par pas de 0.05)
- **Plus strict** : Pénalise localisations imprécises

---

## 12. Pour Ton Projet : Chiens/Chats/Humains

### 12.1 Choix d'Architecture

**Recommandation** : YOLOv8 (medium ou large)

**Justification** :
- **Anchor-free** : Simplifie le tuning
- **État de l'art** : Précision/vitesse optimale
- **Excellente documentation** (Ultralytics)
- **Transfer learning facile** : Modèles pré-entraînés sur COCO (contient déjà chiens/chats/humains !)

### 12.2 Stratégie Recommandée

1. **Commencer avec modèle pré-entraîné COCO**
   - Ces 3 classes existent déjà (mAP déjà ~50-60%)
   
2. **Fine-tuner sur ton dataset spécifique** si :
   - Conditions particulières (éclairage, angles...)
   - Races de chiens/chats spécifiques
   - Contexte métier (caméras de surveillance...)

3. **Augmentation agressive**
   - Mosaic + MixUp
   - Variations d'échelle importantes

### 12.3 Défis Potentiels

**Occlusions** :
- Chien partiellement caché derrière humain
- Solution : Augmentations avec cutout/erasing

**Chevauchements** :
- Plusieurs chiens proches
- Solution : Ajuster IoU threshold NMS

**Petits objets** :
- Chat lointain
- Solution : Augmenter résolution input (640→1280)

---

## 13. Ressources & Sources

### Papers Fondamentaux

1. **YOLOv1** : Redmon et al. (2016)
   *"You Only Look Once: Unified, Real-Time Object Detection"*
   https://arxiv.org/abs/1506.02640

2. **YOLOv2/YOLO9000** : Redmon & Farhadi (2017)
   *"YOLO9000: Better, Faster, Stronger"*
   https://arxiv.org/abs/1612.08242

3. **YOLOv3** : Redmon & Farhadi (2018)
   *"YOLOv3: An Incremental Improvement"*
   https://arxiv.org/abs/1804.02767

4. **CIoU Loss** : Zheng et al. (2020)
   *"Distance-IoU Loss: Faster and Better Learning for Bounding Box Regression"*
   https://arxiv.org/abs/1911.08287

### Comparaisons & Contexte

5. **Faster R-CNN** : Ren et al. (2015)
   *"Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks"*
   https://arxiv.org/abs/1506.01497

6. **Feature Pyramid Networks** : Lin et al. (2017)
   *"Feature Pyramid Networks for Object Detection"*
   https://arxiv.org/abs/1612.03144

### Documentation Pratique

- **Ultralytics YOLOv8** : https://docs.ultralytics.com/
- **Papers with Code** (leaderboards) : https://paperswithcode.com/task/object-detection

---

## 14. Récapitulatif : Points Clés à Retenir

### Concepts Fondamentaux
✅ **YOLO = Single-shot detector** : Une passe forward, prédictions simultanées  
✅ **Grille + Ancres** : Divise l'image, chaque cellule prédit des boxes  
✅ **IoU** : Métrique centrale pour localisation et NMS  
✅ **Multi-échelle** : Détections à différentes résolutions (v3+)  

### Mathématiques
✅ **Loss pondérée** : Équilibre localisation / confidence / classification  
✅ **Racine carrée sur w,h** : Correction du problème d'échelle  
✅ **CIoU** : Version moderne intégrant distance et forme  

### Évolution
✅ **v1** : Proof of concept (vitesse révolutionnaire)  
✅ **v2-v3** : Ancres + multi-échelle (précision++)  
✅ **v5-v8** : Optimisations modernes, anchor-free, décomplexification  

### Pratique
✅ **Transfer learning > From scratch** (presque toujours)  
✅ **Data augmentation** = critique pour robustesse  
✅ **mAP@0.5:0.95** = métrique de référence  
✅ **Trade-off vitesse/précision** via taille modèle (nano → xlarge)  

### Pour Ton Projet
✅ **YOLOv8m** pré-entraîné COCO = point de départ optimal  
✅ Fine-tuning si conditions spécifiques  
✅ Attention aux petits objets (résolution++) et occlusions (augmentations++)  

---

**Fin du cours** - Tu as maintenant les fondations pour comprendre, implémenter et optimiser YOLO pour ton projet de détection ! 🎯
