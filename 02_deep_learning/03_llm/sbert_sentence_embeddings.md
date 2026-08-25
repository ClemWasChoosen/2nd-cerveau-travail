# Sentence-BERT (SBERT) et les Sentence Embeddings — Comprendre, choisir et utiliser les modèles de représentation sémantique

> **Résumé en une phrase** : Ce cours explique comment transformer une phrase ou un document en vecteur sémantique comparable, pourquoi SBERT a été proposé pour rendre cette comparaison efficace, comment les objectifs d'entraînement structurent l'espace vectoriel, et comment choisir entre les principales familles de modèles d'embeddings.

---

## 📋 Métadonnées

| Attribut | Valeur |
|----------|---------|
| **Créé le** | 2026-08-25 |
| **Dernière mise à jour** | 2026-08-25 |
| **Domaine** | Deep Learning / NLP / Recherche sémantique |
| **Niveau** | Intermédiaire → Avancé |
| **Durée de lecture** | ~75 minutes |
| **Fichier proposé** | `sbert_sentence_embeddings.md` |
| **Emplacement proposé** | `/02_deep_learning/03_llm/` |
| **Tags** | `#sbert` `#sentence-embeddings` `#embeddings` `#nlp` `#transformers` `#semantic-similarity` `#contrastive-learning` `#retrieval` |

### Prérequis

- [x] Vecteurs, produit scalaire et norme euclidienne
- [x] Similarité cosinus et embeddings — [RAG](./rag.md)
- [x] Fonctionnement général d'un Transformer — [BERT et variantes](../01_architectures/bert_architecture_variantes.md)
- [x] Notions de base de descente de gradient et de fonction de perte
- [ ] Algèbre linéaire avancée — utile, mais non indispensable

### Cours connexes

- **Prérequis** : [BERT et variantes françaises](../01_architectures/bert_architecture_variantes.md) — architecture de l'encodeur Transformer
- **Prérequis** : [RAG](./rag.md) — embeddings, recherche vectorielle et similarité cosinus
- **Complémentaire** : [Extraction d'intention et routage](./nlp_intent_detection_email_routing.md) — application de la similarité sémantique à une classification
- **Complémentaire** : [Distillation des LLM](./distillation_llm.md) — compression et transfert de connaissances
- **À approfondir** : [SimCSE](https://arxiv.org/abs/2104.08821) — apprentissage contrastif des sentence embeddings

---

## 🎯 Objectifs d'apprentissage

À la fin de ce cours, vous serez capable de :

1. **Distinguer** BERT, SBERT, Sentence Transformers, bi-encoder et cross-encoder
2. **Expliquer** pourquoi un BERT standard n'est pas automatiquement un bon modèle de similarité entre phrases
3. **Décrire** l'architecture siamoise de SBERT et le rôle des poids partagés
4. **Calculer** une similarité cosinus entre deux embeddings normalisés
5. **Comprendre** le rôle du pooling, de la normalisation et de la fonction de perte
6. **Comparer** des familles de modèles : MiniLM, MPNet, modèles multilingues, E5, BGE et CrossEncoder
7. **Traiter** le problème des textes longs, de la troncature et de la perte d'information
8. **Évaluer** un modèle d'embedding avec des métriques adaptées à la tâche
9. **Choisir** un modèle en fonction de la langue, de la longueur des textes, de la latence et du type de recherche
10. **Identifier** les limites d'une décision fondée uniquement sur un score de similarité

---

## 1. Le problème : comparer le sens de deux textes

### 1.1 Même sens, formulations différentes

Deux phrases peuvent exprimer la même idée sans partager exactement les mêmes mots :

> « Je souhaite arrêter mon abonnement. »
>
> « Je voudrais mettre fin à mon contrat. »

Une méthode fondée uniquement sur les mots communs risque de sous-estimer leur proximité. Le problème général est donc de construire une fonction qui transforme un texte en représentation numérique dans laquelle la distance entre deux vecteurs reflète autant que possible la proximité de leur sens.

On note cette fonction :

$$
E : \mathcal{T} \rightarrow \mathbb{R}^{d}
$$

où :

- $$\mathcal{T}$$ est l'ensemble des textes possibles ;
- $$E$$ est le modèle d'encodage ;
- $$d$$ est la dimension de l'embedding ;
- $$E(x)$$ est le vecteur représentant le texte $$x$$.

L'objectif n'est pas que deux textes similaires obtiennent nécessairement le même vecteur. Il est que leur géométrie soit exploitable :

$$
\text{similarité sémantique}(x,y) \text{ élevée}
\Rightarrow
\text{similarité}(E(x), E(y)) \text{ élevée}
$$

Cette implication est une propriété apprise statistiquement, et non une loi mathématique garantie pour tous les textes.

### 1.2 Pourquoi un simple mot-clé ne suffit pas ?

Les représentations lexicales classiques souffrent de plusieurs limites :

| Limite | Exemple | Conséquence |
|---|---|---|
| Synonymie | « résilier » / « annuler » | Deux concepts proches occupent des dimensions différentes |
| Polysémie | « banque » financière / « banque » de données | Un même mot peut avoir plusieurs sens |
| Morphologie | « paiement » / « payer » | Les formes dérivées ne sont pas toujours rapprochées |
| Ordre des mots | « ne pas accepter » / « accepter » | La négation et la composition sont difficiles à représenter |
| Contexte | « contrat annuel » / « contrat d'assurance annuel » | Le sens dépend des mots environnants |

Les embeddings neuronaux cherchent à représenter non seulement la présence d'un mot, mais aussi son contexte et ses relations avec les autres tokens.

---

## 2. BERT produit-il directement de bons vecteurs de phrases ?

### 2.1 BERT est un encodeur contextualisé

BERT reçoit une séquence de tokens et produit un vecteur contextualisé pour chaque token. Si une séquence contient $$L$$ tokens et que la dimension cachée vaut $$d$$, la sortie peut être représentée par une matrice :

$$
H =
\begin{bmatrix}
h_1 \\
h_2 \\
\vdots \\
h_L
\end{bmatrix}
\in \mathbb{R}^{L \times d}
$$

Le vecteur $$h_i$$ dépend du token situé en position $$i$$ et du contexte de la séquence entière.

BERT a principalement été pré-entraîné avec des objectifs de langage, notamment la prédiction de tokens masqués. Cette tâche apprend des représentations riches, mais elle ne garantit pas que la moyenne ou le vecteur `[CLS]` de deux phrases aura une bonne géométrie pour la comparaison sémantique.

### 2.2 Le problème du cross-encoder

Une manière efficace d'utiliser BERT pour comparer deux phrases consiste à les envoyer ensemble dans le modèle :

```text
[CLS] phrase A [SEP] phrase B [SEP]
```

Le Transformer peut alors faire interagir directement chaque token de la phrase A avec chaque token de la phrase B. Un classifieur ou une tête de régression produit ensuite un score de similarité.

Cette architecture est appelée **cross-encoder** :

```text
phrase A ─┐
          ├──► BERT avec interactions croisées ───► score
phrase B ─┘
```

Elle est souvent très précise pour une paire donnée, mais elle ne fournit pas naturellement un vecteur indépendant réutilisable. Si un texte doit être comparé à $$N$$ candidats, il faut recalculer le réseau pour chaque paire :

$$
\text{coût approximatif} = O(N \times C_{\text{encodeur}})
$$

Pour rechercher parmi une grande collection, ce coût devient rapidement important.

### 2.3 Pourquoi le vecteur `[CLS]` ne suffit pas

Le token `[CLS]` est une représentation utile pour certaines tâches de classification après fine-tuning. Cependant, utiliser directement ce vecteur comme embedding universel suppose que l'espace de BERT possède déjà les bonnes propriétés géométriques pour la tâche visée.

Ce n'est pas automatique pour trois raisons :

1. **L'objectif de pré-entraînement est différent** : prédire des tokens n'est pas la même chose qu'ordonner des phrases par similarité.
2. **La géométrie n'est pas explicitement contrainte** : rien n'impose que deux paraphrases soient proches en cosinus.
3. **Les représentations peuvent être anisotropes** : les vecteurs peuvent occuper une région étroite de l'espace, ce qui réduit la qualité de la comparaison angulaire.

Il faut donc entraîner le modèle avec une tâche qui correspond directement à l'usage final : produire des embeddings comparables.

---

## 3. L'idée centrale de SBERT

### 3.1 Définition

**Sentence-BERT (SBERT)** est une modification de BERT qui utilise une architecture siamoise ou triplet afin de produire des embeddings de phrases comparables efficacement avec une similarité cosinus.

Le terme **SBERT** désigne historiquement la méthode introduite par Reimers et Gurevych en 2019. Le terme **Sentence Transformers** désigne aujourd'hui à la fois une bibliothèque et un écosystème plus large de modèles d'embeddings, de CrossEncoders et d'outils d'entraînement.

Il est donc important de distinguer :

| Terme | Signification |
|---|---|
| **BERT** | Architecture pré-entraînée d'encodeur Transformer |
| **SBERT** | Architecture et stratégie d'entraînement pour obtenir des embeddings de phrases comparables |
| **Sentence Transformers** | Bibliothèque et famille de modèles modernes inspirés de SBERT |
| **Bi-encoder** | Deux textes encodés séparément par le même encodeur |
| **Cross-encoder** | Deux textes encodés ensemble pour produire directement un score |

### 3.2 Architecture siamoise : un seul modèle, deux passages

À l'entraînement, les deux branches ont exactement les mêmes paramètres :

```text
                 poids partagés θ
phrase A ───► fθ ───► u = Eθ(A) ──┐
                                  ├──► loss
phrase B ───► fθ ───► v = Eθ(B) ──┘
```

Il ne s'agit pas de deux modèles indépendants. Il s'agit d'un seul modèle $$f_\theta$$ appliqué deux fois, avec les mêmes poids $$\theta$$.

On obtient :

$$
 u = f_\theta(A),
\qquad
 v = f_\theta(B)
$$

La fonction de perte compare ensuite $$u$$ et $$v$$ en fonction de la relation connue entre $$A$$ et $$B$$.

La rétropropagation calcule un gradient par rapport au même paramètre $$\theta$$ :

$$
\theta \leftarrow \theta - \eta \nabla_\theta \mathcal{L}(u,v)
$$

Les deux passages contribuent donc à mettre à jour un unique modèle partagé.

### 3.3 À l'inférence : les textes sont encodés séparément

Une fois le modèle entraîné, on n'a plus besoin d'envoyer les deux textes ensemble :

```text
phrase A ───► encodeur ───► vecteur u
phrase B ───► encodeur ───► vecteur v

                         u, v ───► similarité cosinus
```

Les embeddings des éléments stables d'une collection peuvent être calculés une seule fois, stockés, puis comparés rapidement à un nouvel embedding.

Le gain essentiel de SBERT est donc double :

1. **Apprentissage** : l'espace vectoriel est explicitement structuré pour la comparaison sémantique.
2. **Inférence** : les vecteurs sont calculés séparément et réutilisables.

---

## 4. Du texte au vecteur : tokenisation, pooling et normalisation

### 4.1 Tokenisation

Le texte est d'abord découpé en tokens ou sous-tokens. Pour un Transformer, une phrase devient une séquence :

```text
« Les modèles comprennent le contexte. »
        ↓
[Les, modèles, comprennent, le, contexte, .]
```

En pratique, certains mots sont découpés en sous-unités. Cette étape est importante pour gérer les mots rares, les variantes morphologiques et le vocabulaire fixe du modèle.

### 4.2 Sortie du Transformer

Le Transformer produit un vecteur par token :

$$
H = (h_1, h_2, \ldots, h_L),
\qquad h_i \in \mathbb{R}^{d}
$$

Il faut ensuite réduire la séquence de vecteurs à un seul vecteur de phrase.

### 4.3 Mean pooling avec masque d'attention

Le **mean pooling** calcule la moyenne des vecteurs des tokens réels, sans inclure les tokens de padding.

Soit $$m_i \in \{0,1\}$$ le masque d'attention :

- $$m_i=1$$ si le token appartient au texte réel ;
- $$m_i=0$$ si le token est du padding.

Le vecteur de phrase est :

$$
 s = \frac{\sum_{i=1}^{L} m_i h_i}{\sum_{i=1}^{L} m_i}
$$

Le masque est indispensable. Sans lui, les vecteurs de padding influenceraient la moyenne et introduiraient un signal artificiel dépendant de la longueur du batch.

### 4.4 Autres stratégies de pooling

| Pooling | Principe | Avantages | Limites |
|---|---|---|---|
| `mean` | Moyenne des tokens | Stable, simple, souvent performant | Peut lisser des informations importantes |
| `CLS` | Utilise le token `[CLS]` | Rapide, directement disponible | Dépend fortement de l'entraînement du modèle |
| `max` | Maximum par dimension | Retient les activations fortes | Peut être sensible à un token isolé |
| `mean_sqrt_len` | Somme divisée par $$\sqrt{L}$$ | Compromis entre somme et moyenne | Moins intuitif |
| Attention pooling | Poids appris par token | Peut sélectionner les tokens utiles | Ajoute des paramètres et de la complexité |

Il n'existe pas un pooling universellement optimal. Le pooling est une composante du modèle entraîné : changer le pooling sans réévaluer le modèle peut modifier la géométrie des embeddings.

### 4.5 Normalisation L2

Pour comparer les vecteurs avec le cosinus, on peut normaliser chaque vecteur :

$$
\hat{s} = \frac{s}{\lVert s \rVert_2}
$$

avec :

$$
\lVert s \rVert_2 = \sqrt{\sum_{j=1}^{d} s_j^2}
$$

Après normalisation :

$$
\lVert \hat{s} \rVert_2 = 1
$$

La similarité cosinus devient alors un simple produit scalaire :

$$
\cos(\hat{u},\hat{v}) = \hat{u}^{T}\hat{v}
$$

C'est à la fois mathématiquement exact et efficace pour calculer une matrice de similarités.

---

## 5. La similarité cosinus : géométrie et interprétation

### 5.1 Formule

Pour deux vecteurs non nuls $$u$$ et $$v$$ :

$$
\cos(u,v) = \frac{u \cdot v}{\lVert u \rVert_2 \lVert v \rVert_2}
$$

Géométriquement, cette quantité mesure l'angle entre les deux vecteurs :

- $$1$$ : même direction ;
- $$0$$ : directions orthogonales ;
- $$-1$$ : directions opposées.

Dans un modèle d'embeddings de phrases, une proximité élevée est généralement souhaitable pour deux textes sémantiquement proches. Cependant, il faut éviter d'interpréter mécaniquement une valeur comme une probabilité.

### 5.2 Pourquoi le cosinus plutôt que la distance euclidienne ?

La norme d'un embedding peut varier pour des raisons qui ne correspondent pas directement au sens. Le cosinus se concentre sur l'orientation :

$$
\cos(u,v) = \cos(\alpha u, \beta v)
\quad \text{pour } \alpha>0,\ \beta>0
$$

La longueur des vecteurs est donc ignorée, ce qui est souvent adapté lorsque la direction encode le contenu sémantique.

Une fois les vecteurs normalisés, la distance euclidienne et le cosinus sont monotoniquement liés :

$$
\lVert \hat{u}-\hat{v}\rVert_2^2
= \lVert \hat{u}\rVert_2^2 + \lVert \hat{v}\rVert_2^2 - 2\hat{u}^{T}\hat{v}
= 2 - 2\cos(\hat{u},\hat{v})
$$

Ainsi, maximiser le cosinus revient à minimiser la distance euclidienne au carré sur la sphère unité.

### 5.3 Le score n'est pas une probabilité

Un score de cosinus de `0.82` ne signifie pas « 82 % de chances que les deux textes aient la même intention ». Il signifie uniquement que, dans la géométrie apprise par le modèle, les deux vecteurs ont une forte proximité angulaire relative.

Pour transformer un score en décision, il faut calibrer un seuil sur des données représentatives :

$$
\text{décision}(x) =
\begin{cases}
\text{classe candidate} & \text{si } s(x) \geq \tau \\
\text{abstention ou revue} & \text{sinon}
\end{cases}
$$

Le seuil $$\tau$$ dépend du modèle, de la langue, des classes, du type de texte et du coût relatif des faux positifs et des faux négatifs.

---

## 6. Les fonctions de perte : comment structurer l'espace vectoriel

La fonction de perte est le mécanisme qui transforme la notion qualitative de « textes proches » ou « textes différents » en contrainte mathématique sur les vecteurs.

### 6.1 Régression sur la similarité

Si chaque paire possède une note de similarité $$y \in [0,1]$$, on peut minimiser :

$$
\mathcal{L}_{\text{MSE}}
= \left(\cos(u,v)-y\right)^2
$$

Cette approche demande des annotations graduées. Elle est naturelle pour la tâche STS, où l'on attribue une note de similarité à une paire de phrases.

### 6.2 Classification de paires

On peut également demander au modèle de prédire une classe de relation :

- entailment ;
- paraphrase ;
- contradiction ;
- paire non liée.

Une tête de classification peut recevoir une combinaison des deux embeddings, par exemple :

$$
 z = [u; v; |u-v|]
$$

puis produire des logits :

$$
 p(y \mid u,v) = \mathrm{softmax}(Wz+b)
$$

La cross-entropy pousse le modèle à produire des représentations permettant de distinguer les relations annotées.

### 6.3 Triplet loss

Avec un triplet constitué d'une ancre $$a$$, d'un positif $$p$$ et d'un négatif $$n$$, on veut que l'ancre soit plus proche du positif que du négatif d'une marge $$m>0$$.

Avec une distance $$d$$ :

$$
\mathcal{L}_{\text{triplet}}
= \max\left(0, d(a,p)-d(a,n)+m\right)
$$

Si l'on travaille directement avec le cosinus, une forme équivalente est :

$$
\mathcal{L}_{\text{cos-triplet}}
= \max\left(0,
\cos(a,n)-\cos(a,p)+m
\right)
$$

La perte est nulle lorsque le positif est suffisamment plus proche que le négatif.

### 6.4 Multiple Negatives Ranking Loss et apprentissage contrastif

Une approche moderne consiste à utiliser un batch de paires positives :

$$
(A_1,B_1), (A_2,B_2), \ldots, (A_B,B_B)
$$

Pour une phrase $$A_i$$, $$B_i$$ est le positif et les autres $$B_j$$ du batch servent de négatifs implicites. Avec une température $$\tau_T>0$$ :

$$
\mathcal{L}_i
= -\log
\frac{\exp\left(\mathrm{sim}(u_i,v_i)/\tau_T\right)}
{\sum_{j=1}^{B}
\exp\left(\mathrm{sim}(u_i,v_j)/\tau_T\right)}
$$

La loss du batch est généralement :

$$
\mathcal{L}
= \frac{1}{B}\sum_{i=1}^{B}\mathcal{L}_i
$$

Cette formulation apprend au modèle à placer le vrai partenaire plus haut que les autres candidats.

**Pourquoi est-elle efficace ?**

- Elle exploite plusieurs négatifs par exemple.
- Elle rapproche les positifs tout en séparant les autres éléments du batch.
- Elle correspond naturellement à une tâche de ranking ou de recherche.

**Risque important : les faux négatifs.** Si deux éléments différents du batch sont en réalité tous les deux pertinents, la loss peut demander au modèle de les éloigner artificiellement. La composition des batches et la qualité des paires sont donc déterminantes.

### 6.5 Le rôle de la température

La température contrôle la concentration de la distribution softmax :

- température faible : le modèle se concentre fortement sur le meilleur candidat ;
- température élevée : la distribution est plus douce.

Elle modifie l'échelle des gradients et la difficulté du ranking. Elle doit être considérée comme un hyperparamètre d'entraînement, et non comme un simple paramètre cosmétique.

### 6.6 Une fonction de perte ne crée pas de sens à partir de rien

La loss ne garantit pas une compréhension générale du langage. Elle déforme les paramètres du modèle à partir des exemples disponibles.

Si les données d'entraînement :

- ignorent une distinction importante ;
- contiennent des labels incohérents ;
- utilisent des négatifs trop faciles ;
- ne couvrent pas les formulations réelles ;

alors l'espace vectoriel peut être performant sur le benchmark mais mauvais sur le cas d'usage réel.

---

## 7. Pourquoi SBERT est beaucoup plus rapide qu'un CrossEncoder

### 7.1 Recherche avec un CrossEncoder

Supposons un texte requête $$q$$ et une collection de $$N$$ textes candidats $$d_1,\ldots,d_N$$.

Un CrossEncoder doit traiter :

$$
(q,d_1), (q,d_2), \ldots, (q,d_N)
$$

Le calcul contient une passe du Transformer par paire. Les tokens de la requête sont donc retraités à chaque comparaison.

### 7.2 Recherche avec un bi-encoder

Avec SBERT :

1. encoder la requête une fois : $$u=E(q)$$ ;
2. encoder les candidats une fois ou réutiliser leurs vecteurs ;
3. calculer les similarités par produits scalaires.

Le coût devient approximativement :

$$
O(C_{\text{encodeur}}(q))
+ O(N \times d)
$$

si les candidats sont déjà encodés.

La différence structurelle est donc :

| Architecture | Interaction entre les textes | Réutilisation des vecteurs | Usage typique |
|---|---|---|---|
| CrossEncoder | Dans le Transformer | Non | Reranking de quelques candidats |
| Bi-encoder / SBERT | Après encodage, dans l'espace vectoriel | Oui | Recherche, clustering, classification par prototypes |

### 7.3 Architecture hybride recommandée

Dans beaucoup de systèmes de recherche, les deux approches sont complémentaires :

```text
Requête
  │
  ▼
Bi-encoder : récupération rapide des top-k candidats
  │
  ▼
CrossEncoder : reranking précis de ces k candidats
  │
  ▼
Résultat final
```

On évite ainsi d'utiliser le CrossEncoder sur toute la collection tout en profitant de sa capacité à modéliser les interactions fines.

---

## 8. Les principales familles de modèles

### 8.1 Modèle généraliste vs modèle orienté tâche

Il n'existe pas un meilleur modèle absolu. Un modèle est bon par rapport à :

- la langue du texte ;
- la longueur des entrées ;
- la symétrie de la comparaison ;
- le type de recherche ;
- le budget de calcul ;
- le besoin de latence ;
- les données de calibration disponibles.

### 8.2 Tableau de comparaison

| Famille / modèle | Langues | Dimension indicative | Longueur indicative | Force principale | Point d'attention |
|---|---:|---:|---:|---|---|
| BERT brut + pooling | Selon le checkpoint | Variable | Selon le checkpoint | Base contextualisée flexible | Pas nécessairement entraîné pour la similarité |
| `all-MiniLM-L6-v2` | Anglais | 384 | 256 selon la carte du modèle | Très bon compromis vitesse / qualité en anglais | Pas adapté à un besoin multilingue par défaut |
| `all-mpnet-base-v2` | Anglais | 768 | 384 selon la carte du modèle | Qualité générale élevée pour phrases et paragraphes | Plus lourd et plus lent que MiniLM |
| `paraphrase-multilingual-MiniLM-L12-v2` | Multilingue | 384 | 128 selon sa carte du modèle | Similarité et paraphrases multilingues, modèle compact | Fenêtre courte ; vigilance sur les textes longs |
| `distiluse-base-multilingual-cased-v2` | Multilingue | 512 | Selon le checkpoint | Baseline multilingue légère et ancienne | À comparer empiriquement avec des modèles plus récents |
| `LaBSE` | Multilingue | 768 | Selon le checkpoint | Alignement de phrases et bitext mining | Pas nécessairement optimal pour chaque recherche monolingue |
| `multilingual-e5-*` | Multilingue | 384 à 1024 selon la variante | 512 selon la variante | Recherche dense et requête/document | Préfixes `query:` et `passage:` importants pour la recherche |
| `BAAI/bge-m3` | Plus de 100 langues | 1024 | Jusqu'à 8192 selon la carte | Dense, sparse et multi-vector ; textes longs | Plus lourd ; pipeline plus complexe |
| CrossEncoder | Selon le checkpoint | Pas d'embedding réutilisable standard | Selon le checkpoint | Score pair-à-pair précis | Coût élevé si la collection est grande |

Les dimensions et longueurs ne sont pas des indicateurs de qualité à eux seuls. Une dimension plus élevée peut augmenter le coût de stockage et de calcul sans améliorer la performance sur la tâche cible.

### 8.3 `paraphrase-multilingual-MiniLM-L12-v2`

Ce modèle est particulièrement intéressant comme baseline multilingue compacte : sa carte de modèle indique une représentation dense de dimension 384 et une architecture Sentence Transformer avec mean pooling et une longueur maximale de séquence de 128 tokens.

Son fonctionnement est cohérent avec une comparaison de phrases courtes et de paraphrases. Il faut en revanche mesurer explicitement son comportement si les entrées sont des paragraphes longs ou contiennent beaucoup de bruit.

### 8.4 `all-MiniLM-L6-v2`

`all-MiniLM-L6-v2` est une baseline anglophone très utilisée pour les phrases et courts paragraphes. Elle privilégie le compromis entre vitesse et qualité.

Le nombre de couches réduit par rapport à des encodeurs plus lourds diminue le coût d'inférence, mais le modèle ne doit pas être choisi pour du français ou du multilingue uniquement parce qu'il est populaire dans les exemples de code.

### 8.5 `all-mpnet-base-v2`

`all-mpnet-base-v2` utilise un encodeur plus lourd et produit des vecteurs de dimension 768. Sa carte de modèle indique un entraînement contrastif sur plus d'un milliard de paires et une longueur par défaut limitée à 384 word pieces.

Il peut améliorer la qualité sur des textes anglophones, mais le coût mémoire, la latence et la compatibilité linguistique doivent être mesurés sur les données réelles.

### 8.6 Les modèles multilingues par distillation

Une stratégie de construction multilingue consiste à partir d'un modèle performant dans une langue source, puis à apprendre à un modèle multilingue à placer une phrase traduite au même endroit que la phrase originale.

Pour une paire originale/traduction $$x^{(s)},x^{(t)}$$, le modèle cherche à minimiser une distance entre :

$$
E_{\text{multi}}(x^{(s)})
\quad \text{et} \quad
E_{\text{multi}}(x^{(t)})
$$

Cette approche permet d'aligner plusieurs langues dans un espace commun. Elle ne signifie pas que chaque registre, dialecte ou domaine métier a été vu directement pendant l'entraînement.

### 8.7 `multilingual-e5`

La famille E5 est orientée vers la recherche textuelle et l'apprentissage contrastif à grande échelle. La carte du modèle recommande des préfixes comme :

```text
query: texte recherché
passage: texte candidat
```

Ces préfixes ne sont pas décoratifs : ils correspondent au format utilisé pendant l'entraînement. Pour une tâche de recherche asymétrique, les supprimer peut dégrader les performances.

Pour une comparaison symétrique de deux phrases, il faut suivre les recommandations précises de la carte du modèle et tester le format d'entrée retenu.

### 8.8 `BAAI/bge-m3`

BGE-M3 se distingue par trois propriétés :

- **multilingue** : plus de 100 langues annoncées ;
- **multi-fonctionnel** : dense, sparse et multi-vector ;
- **multi-granulaire** : de la phrase au document long, avec une longueur annoncée jusqu'à 8192 tokens.

C'est une option intéressante lorsque la tâche nécessite à la fois recherche dense, signal lexical et traitement de documents longs. Elle est en revanche plus lourde qu'un MiniLM et introduit davantage de choix d'intégration.

### 8.9 CrossEncoder : modèle associé, mais architecture différente

Un CrossEncoder ne produit pas nécessairement un vecteur indépendant par texte. Il prend une paire et renvoie directement un score :

$$
(q,d) \longmapsto s(q,d)
$$

Il est souvent utilisé après un bi-encoder : le bi-encoder réduit la collection, puis le CrossEncoder réordonne un petit nombre de candidats. Les deux modèles ne sont donc pas concurrents dans tous les scénarios.

---

## 9. Le problème des textes longs

### 9.1 Troncature

Chaque checkpoint possède une longueur maximale. Si une séquence dépasse cette longueur, le tokenizer ou le modèle peut supprimer les tokens excédentaires.

Formellement, si la longueur réelle vaut $$L$$ et la limite $$L_{\max}$$ :

$$
L_{\text{utilisée}} = \min(L,L_{\max})
$$

Si l'information discriminante apparaît après $$L_{\max}$$, elle ne participe pas à l'embedding final.

La troncature n'est pas une perte progressive : les tokens supprimés ne contribuent plus du tout au calcul.

### 9.2 Pourquoi la moyenne peut diluer le sens

Même sans troncature, un long texte peut contenir :

- plusieurs sujets ;
- une signature ;
- un historique de conversation ;
- des mentions légales ;
- des éléments sans rapport avec la demande principale.

Le mean pooling agrège alors des informations hétérogènes. Le vecteur final peut représenter une moyenne sémantique du document plutôt que le passage réellement discriminant.

### 9.3 Stratégies de traitement

| Stratégie | Principe | Avantage | Risque |
|---|---|---|---|
| Nettoyage | Retirer les éléments non informatifs | Simple et peu coûteux | Le nettoyage peut supprimer un signal utile |
| Découpage en chunks | Encoder des fragments séparément | Respecte la limite du modèle | Gestion de plusieurs scores et de plusieurs contextes |
| Max pooling des scores | Conserver le meilleur score fragment/prototype | Détecte un passage très pertinent | Peut sélectionner un fragment hors contexte |
| Moyenne des embeddings | Agréger les vecteurs des chunks | Résultat global stable | Réintroduit une dilution |
| Pooling pondéré | Donner plus de poids aux fragments importants | Plus expressif | Nécessite une règle ou un modèle supplémentaire |
| Modèle long contexte | Utiliser un checkpoint adapté | Moins de découpage | Plus de mémoire et pas forcément meilleur sur phrases courtes |

La bonne solution doit être choisie sur une mesure expérimentale. Un modèle possédant une longue fenêtre ne garantit pas que le pooling global soit adapté à une décision locale.

### 9.4 Mesurer plutôt que supposer

Avant de modifier le pipeline, il faut enregistrer au minimum :

- le nombre de tokens avant troncature ;
- la limite du modèle ;
- le nombre de textes tronqués ;
- la position de l'information utile ;
- le score avant et après traitement des textes longs.

Cette instrumentation permet de distinguer trois causes souvent confondues :

1. la troncature ;
2. la dilution par agrégation ;
3. une confusion sémantique réelle entre catégories.

---

## 10. Comment écrire ou sélectionner de bons exemples

Cette section concerne les situations où le modèle pré-entraîné est utilisé sans fine-tuning local, par exemple avec des prototypes ou des exemples de référence.

### 10.1 Un exemple doit représenter une intention, pas un document complet

Un prototype doit exprimer clairement le concept à reconnaître. Il vaut mieux plusieurs formulations courtes et distinctes qu'un unique paragraphe contenant plusieurs sujets.

Mauvaise pratique :

```text
Bonjour, je vous contacte au sujet de plusieurs éléments, avec mon ancienne demande, ma signature, un historique et différentes informations annexes.
```

Meilleure pratique :

```text
Je souhaite mettre fin à mon abonnement.
```

Le texte réel peut rester long dans l'application, mais le catalogue de référence doit décrire proprement la notion recherchée.

### 10.2 Diversité contrôlée des positifs

Les positifs doivent couvrir des formulations variées :

- formulation directe ;
- formulation polie ;
- synonyme ;
- formulation indirecte ;
- variation morphologique ;
- variation de longueur raisonnable.

La diversité ne doit pas changer l'intention. Un exemple qui introduit une seconde demande ambiguë peut élargir artificiellement la classe.

### 10.3 Négatifs difficiles

Un négatif utile partage souvent une partie du vocabulaire avec la classe positive, mais exprime une décision différente.

```text
Positif : « Je souhaite modifier mes coordonnées bancaires. »
Négatif difficile : « Je souhaite comprendre une facture prélevée sur mon compte bancaire. »
```

Le négatif doit être réellement plausible. Des négatifs trop éloignés apprennent peu ; des négatifs mal étiquetés peuvent dégrader l'espace.

### 10.4 Adapter les exemples au registre réel

Les modèles multilingues sont généralement robustes à des variations modérées, mais ils peuvent être moins fiables sur :

- fautes nombreuses ;
- abréviations ;
- jargon non expliqué ;
- textes très fragmentés ;
- mélange de langues ;
- signatures ou réponses automatiques concaténées.

Il ne faut pas réécrire artificiellement toutes les données en français académique sans mesurer l'effet. La stratégie recommandée est de couvrir le registre réel dans les exemples, puis de vérifier la performance sur des données annotées représentatives.

### 10.5 Ce qu'il ne faut pas conclure

Ajouter un exemple positif ne « programme » pas une nouvelle règle dans le modèle. Cela modifie seulement les candidats auxquels un nouvel embedding sera comparé.

Le catalogue agit comme une représentation de référence, pas comme un fine-tuning du Transformer :

$$
\theta_{\text{modèle}} \text{ reste fixe}
$$

mais l'ensemble des prototypes $$P$$ est modifié :

$$
\hat{y}(x) = \arg\max_{p \in P} \cos(E(x),E(p))
$$

Il faut donc évaluer chaque ajout sur un jeu de validation séparé.

---

## 11. Utilisation pratique avec Sentence Transformers

### 11.1 Encodage et comparaison

```python
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer(
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

texts = [
    "Je souhaite arrêter mon abonnement.",
    "Je voudrais mettre fin à mon contrat.",
    "La livraison de ma commande est en retard.",
]

# La normalisation rend le produit scalaire équivalent au cosinus.
embeddings = model.encode(
    texts,
    normalize_embeddings=True,
    convert_to_numpy=True,
)

similarities = embeddings @ embeddings.T

print(embeddings.shape)
print(np.round(similarities, 3))
```

La matrice obtenue est :

$$
S = E E^T
$$

Si $$E \in \mathbb{R}^{n \times d}$$ contient des lignes normalisées, alors :

$$
S_{ij} = \cos(e_i,e_j)
$$

### 11.2 Comparer une requête à des prototypes

```python
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer(
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

query = "Je veux annuler mon contrat."
prototypes = [
    "Le client demande la fin de son abonnement.",
    "Le client souhaite modifier son adresse.",
    "Le client demande une explication sur une facture.",
]

query_embedding = model.encode(
    [query],
    normalize_embeddings=True,
    convert_to_numpy=True,
)[0]

prototype_embeddings = model.encode(
    prototypes,
    normalize_embeddings=True,
    convert_to_numpy=True,
)

scores = prototype_embeddings @ query_embedding
best_index = int(np.argmax(scores))

print("Candidat retenu :", prototypes[best_index])
print("Score :", float(scores[best_index]))
```

### 11.3 Score maximum parmi plusieurs prototypes

Une classe peut être décrite par plusieurs exemples. Pour une classe $$c$$ et ses prototypes $$P_c$$ :

$$
 s(x,c) = \max_{p \in P_c}
 \cos(E(x),E(p))
$$

Puis la décision devient :

$$
\hat{c} = \arg\max_{c} s(x,c)
$$

**Pourquoi utiliser un maximum ?** Une formulation particulière d'un texte peut être très proche d'un seul prototype alors que la moyenne de tous les prototypes éloignerait le représentant de la classe. Le maximum est donc adapté lorsque les exemples couvrent des formulations distinctes.

**Limite :** un prototype mal écrit ou trop général peut produire un faux meilleur score. Il faut surveiller les exemples gagnants et les négatifs difficiles.

### 11.4 Utilisation d'un CrossEncoder pour reranker

```python
from sentence_transformers import CrossEncoder

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2")

query = "Je souhaite arrêter mon abonnement."
candidates = [
    "Le client veut mettre fin à son contrat.",
    "Le client demande la date du prochain prélèvement.",
]

pairs = [(query, candidate) for candidate in candidates]
scores = reranker.predict(pairs)

for candidate, score in sorted(
    zip(candidates, scores),
    key=lambda item: item[1],
    reverse=True,
):
    print(f"{score:.3f} - {candidate}")
```

Le score d'un CrossEncoder n'a pas nécessairement la même échelle ni la même interprétation que le cosinus d'un bi-encoder. Les deux scores ne doivent pas être mélangés sans calibration.

---

## 12. Évaluer correctement un modèle d'embeddings

### 12.1 La métrique doit correspondre à la tâche

| Tâche | Question | Métriques adaptées |
|---|---|---|
| Similarité de phrases | Les scores respectent-ils l'ordre humain ? | Spearman, Pearson, MSE |
| Recherche | Le bon document apparaît-il tôt ? | Recall@k, MRR, nDCG@k |
| Classification par prototypes | La classe prédite est-elle correcte ? | Accuracy, macro-F1, matrice de confusion |
| Détection de cas inconnus | Le système sait-il s'abstenir ? | Precision/recall d'abstention, couverture, risk-coverage |
| Production | Le système est-il exploitable ? | Latence, mémoire, taux d'erreur, taux de revue |

### 12.2 Séparer les ensembles de données

Un jeu de données minimal doit distinguer :

- **entraînement** : utilisé pour apprendre ou ajuster le modèle ;
- **validation** : utilisé pour choisir les exemples, seuils et hyperparamètres ;
- **test** : utilisé une seule fois pour mesurer la généralisation finale.

Même sans fine-tuning, la séparation reste nécessaire. Si l'on choisit les prototypes et le seuil en observant toujours les mêmes exemples, on risque de suradapter le catalogue au jeu de validation.

### 12.3 Éviter les fuites de données

Les doublons ou quasi-doublons entre les ensembles donnent une estimation trop optimiste. Il faut vérifier :

- les duplications exactes ;
- les paraphrases très proches ;
- les messages issus d'une même conversation ;
- les exemples d'une même source ou période ;
- les textes copiés dans les prototypes.

### 12.4 Mesurer les erreurs par sous-groupes

Une moyenne globale peut cacher des faiblesses importantes. Il est utile de comparer les performances selon :

- longueur du texte ;
- langue ;
- registre ;
- présence de fautes ;
- présence de plusieurs demandes ;
- classe d'intention ;
- score de confiance ;
- présence d'un vocabulaire partagé avec une classe concurrente.

### 12.5 Calibrer l'abstention

Un système de classification ouverte doit pouvoir répondre : « aucune classe connue ne convient ». On peut utiliser un seuil de score, un seuil de marge ou les deux :

$$
\text{marge}(x) = s_1(x)-s_2(x)
$$

où $$s_1$$ et $$s_2$$ sont les deux meilleurs scores.

Une décision automatique peut exiger :

$$
 s_1(x) \geq \tau_s
\quad \text{et} \quad
s_1(x)-s_2(x) \geq \tau_m
$$

Les seuils doivent être déterminés avec des données comprenant des cas réellement inconnus, et non seulement des exemples faciles des classes connues.

---

## 13. Méthode de choix d'un modèle

### Étape 1 — Définir la tâche

Avant de comparer des modèles, préciser si l'on cherche à :

- comparer deux phrases de manière symétrique ;
- rechercher un document avec une requête ;
- classer un texte par prototypes ;
- reranker quelques candidats ;
- regrouper des textes ;
- traiter des documents longs ;
- fonctionner dans plusieurs langues.

### Étape 2 — Définir les contraintes

Documenter :

- langues supportées ;
- longueur typique et maximale ;
- latence cible ;
- mémoire disponible ;
- calcul CPU/GPU ;
- fonctionnement hors ligne ;
- licence ;
- besoin de reproductibilité ;
- volume de textes à indexer.

### Étape 3 — Sélectionner deux ou trois baselines

Une comparaison utile peut inclure :

1. un modèle compact et rapide ;
2. un modèle plus lourd supposé plus qualitatif ;
3. un modèle spécialisé dans la recherche ou le multilingue.

Il faut conserver le même jeu de test, les mêmes règles de prétraitement et la même procédure de calibration.

### Étape 4 — Comparer qualité et coût

Un tableau de décision doit inclure au minimum :

| Critère | Question |
|---|---|
| Qualité | Le modèle améliore-t-il la métrique de la tâche ? |
| Robustesse | Résiste-t-il aux longueurs et formulations réelles ? |
| Latence | Quel est le temps par texte et par batch ? |
| Mémoire | Quelle quantité de RAM/VRAM et de stockage est nécessaire ? |
| Index | Combien coûte le stockage des embeddings ? |
| Langues | Le modèle couvre-t-il réellement les langues nécessaires ? |
| Exploitabilité | Le score et les erreurs sont-ils analysables ? |
| Maintenance | Le modèle, sa version et ses paramètres sont-ils figés ? |

### Étape 5 — Figer la configuration

Pour rendre les résultats reproductibles, enregistrer :

- identifiant exact du modèle ;
- version des dépendances ;
- tokenizer ;
- longueur maximale ;
- pooling ;
- normalisation ;
- métrique ;
- batch size ;
- seuils ;
- version du catalogue ou de l'index.

Changer de modèle implique de recalculer les embeddings et de recalibrer les seuils. Les scores de deux modèles différents ne sont pas directement comparables.

---

## 14. Limites et erreurs fréquentes

### 14.1 « Plus le score est élevé, plus la réponse est vraie »

Faux. Le score mesure une proximité dans un espace appris. Il ne vérifie pas la vérité factuelle du texte et ne remplace pas une règle métier ou une validation.

### 14.2 « SBERT comprend tous les documents longs »

Faux. Un modèle de phrase est optimisé pour produire un vecteur compact. Un document long et multi-thématique peut nécessiter un découpage, une recherche par passages ou une architecture long contexte.

### 14.3 « Un modèle multilingue est aussi bon dans toutes les langues »

Faux. La couverture annoncée n'implique pas une qualité identique pour chaque langue, domaine et registre.

### 14.4 « Une dimension plus grande est toujours meilleure »

Faux. Une dimension supérieure augmente souvent les coûts. La qualité dépend de l'objectif d'entraînement, des données, de la langue et de la tâche.

### 14.5 « Le modèle apprend quand on ajoute un prototype »

Faux dans un pipeline sans fine-tuning. Le modèle reste figé ; seul l'ensemble des vecteurs de référence change.

### 14.6 « Le maximum de prototypes est sans risque »

Faux. Un prototype trop vague, trop long ou mal étiqueté peut devenir un attracteur erroné. Le système doit conserver le prototype gagnant pour permettre l'analyse des erreurs.

### 14.7 « Les négatifs doivent être très différents »

Faux. Les négatifs les plus instructifs sont souvent proches lexicalement mais différents sémantiquement. Ils doivent toutefois être correctement étiquetés.

---

## 15. Synthèse : ce qu'il faut retenir

### Niveau débutant

SBERT transforme chaque phrase en vecteur. Deux phrases sémantiquement proches doivent produire des vecteurs proches. On compare ensuite ces vecteurs avec le cosinus.

### Niveau intermédiaire

SBERT utilise un même encodeur partagé pour chaque texte. L'entraînement apprend une géométrie adaptée à la similarité. À l'inférence, les textes sont encodés séparément, ce qui permet de réutiliser les vecteurs et d'éviter de recalculer un réseau pour chaque paire.

### Niveau avancé

La qualité dépend du triplet :

$$
\text{représentation} + \text{objectif d'entraînement} + \text{données}
$$

Le pooling transforme une séquence de représentations tokenisées en un vecteur fixe. La normalisation L2 permet de réduire la similarité à un produit scalaire. Les pertes contrastives organisent l'espace en rapprochant les positifs et en repoussant les négatifs. Les modèles modernes diffèrent surtout par leurs données, leur objectif, leur taille, leur langue et leur tâche cible.

### Phrase de présentation

> **SBERT ne rend pas simplement BERT plus rapide : il réentraîne la représentation pour que la géométrie des vecteurs soit directement exploitable, puis sépare l'encodage de la comparaison afin de rendre la recherche sémantique réutilisable et scalable.**

---

## 16. Exercices de vérification

### Exercice 1 — Poids partagés

Deux phrases passent-elles dans deux modèles indépendants ?

**Réponse :** Non. Dans l'architecture siamoise, elles passent dans le même encodeur, utilisé deux fois avec les mêmes paramètres.

### Exercice 2 — Cosinus

Si $$u$$ et $$v$$ sont normalisés, quelle opération permet de calculer leur cosinus ?

**Réponse :** Le produit scalaire $$u^Tv$$.

### Exercice 3 — Recherche

Pourquoi encoder une collection de documents avant l'arrivée des requêtes ?

**Réponse :** Les embeddings des documents peuvent être calculés une seule fois puis réutilisés. À chaque requête, il suffit d'encoder la requête et de comparer les vecteurs.

### Exercice 4 — Texte long

Un modèle annonce une longueur maximale de 128 tokens. Que se passe-t-il avec un texte de 300 tokens si aucune segmentation n'est appliquée ?

**Réponse :** Les tokens au-delà de la limite sont tronqués. Le modèle ne peut pas utiliser l'information supprimée.

### Exercice 5 — Seuil

Un cosinus de `0.85` est-il une probabilité de 85 % ?

**Réponse :** Non. C'est un score de proximité géométrique. Il faut calibrer un seuil avec des données annotées.

### Exercice 6 — CrossEncoder

Pourquoi utiliser un CrossEncoder après un bi-encoder ?

**Réponse :** Le bi-encoder récupère rapidement un petit ensemble de candidats ; le CrossEncoder peut ensuite les comparer plus finement, à un coût limité.

---

## 17. Sources et références

### Papiers fondateurs

1. **Reimers, N. & Gurevych, I. (2019).** *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks.* EMNLP 2019. [arXiv:1908.10084](https://arxiv.org/abs/1908.10084)
2. **Devlin, J. et al. (2018).** *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.* [arXiv:1810.04805](https://arxiv.org/abs/1810.04805)
3. **Vaswani, A. et al. (2017).** *Attention Is All You Need.* [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)
4. **Reimers, N. & Gurevych, I. (2020).** *Making Monolingual Sentence Embeddings Multilingual using Knowledge Distillation.* EMNLP 2020. [arXiv:2004.09813](https://arxiv.org/abs/2004.09813)
5. **Gao, T. et al. (2021).** *SimCSE: Simple Contrastive Learning of Sentence Embeddings.* EMNLP 2021. [arXiv:2104.08821](https://arxiv.org/abs/2104.08821)
6. **Wang, L. et al. (2024).** *Multilingual E5 Text Embeddings: A Technical Report.* [arXiv:2402.05672](https://arxiv.org/abs/2402.05672)
7. **Muennighoff, N. et al. (2023).** *MTEB: Massive Text Embedding Benchmark.* EACL 2023. [arXiv:2210.07316](https://arxiv.org/abs/2210.07316)
8. **Khattab, O. & Zaharia, M. (2020).** *ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT.* [arXiv:2004.12832](https://arxiv.org/abs/2004.12832)

### Documentation et cartes de modèles

- [Sentence Transformers — documentation officielle](https://www.sbert.net/)
- [Semantic Textual Similarity — Sentence Transformers](https://www.sbert.net/docs/sentence_transformer/usage/semantic_textual_similarity.html)
- [`paraphrase-multilingual-MiniLM-L12-v2` — carte Hugging Face](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2)
- [`all-MiniLM-L6-v2` — carte Hugging Face](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
- [`all-mpnet-base-v2` — carte Hugging Face](https://huggingface.co/sentence-transformers/all-mpnet-base-v2)
- [`multilingual-e5-base` — carte Hugging Face](https://huggingface.co/intfloat/multilingual-e5-base)
- [`BAAI/bge-m3` — carte Hugging Face](https://huggingface.co/BAAI/bge-m3)
- [MTEB — benchmark et leaderboard](https://github.com/embeddings-benchmark/mteb)

### Références BibTeX principales

```bibtex
@inproceedings{reimers-2019-sentence-bert,
  title     = {Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks},
  author    = {Reimers, Nils and Gurevych, Iryna},
  booktitle = {Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing},
  year      = {2019},
  url       = {https://arxiv.org/abs/1908.10084}
}

@inproceedings{gao-2021-simcse,
  title     = {SimCSE: Simple Contrastive Learning of Sentence Embeddings},
  author    = {Gao, Tianyu and Yao, Xingcheng and Chen, Danqi},
  booktitle = {Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing},
  year      = {2021},
  url       = {https://arxiv.org/abs/2104.08821}
}
```

---

## 18. Checklist opérationnelle

Avant de mettre un modèle d'embedding en production :

- [ ] La tâche est-elle définie : similarité, recherche, classification ou reranking ?
- [ ] Le modèle couvre-t-il les langues nécessaires ?
- [ ] La longueur maximale est-elle connue et instrumentée ?
- [ ] Le pooling et la normalisation sont-ils explicitement configurés ?
- [ ] Le format d'entrée est-il conforme à la carte du modèle ?
- [ ] Les positifs et négatifs sont-ils représentatifs des données réelles ?
- [ ] Les faux négatifs et les doublons ont-ils été contrôlés ?
- [ ] Le seuil a-t-il été calibré sur un jeu séparé ?
- [ ] Les cas inconnus et l'abstention sont-ils évalués ?
- [ ] Les latences, la mémoire et le coût d'indexation sont-ils mesurés ?
- [ ] La version exacte du modèle et des dépendances est-elle enregistrée ?
- [ ] Les erreurs peuvent-elles être expliquées par le prototype ou le passage sélectionné ?

> **Principe final :** un embedding n'est pas une vérité numérique universelle. C'est une représentation optimisée pour un objectif et des données. Le bon modèle est celui qui obtient les meilleurs compromis mesurés sur la tâche réelle, avec une décision suffisamment explicable et maîtrisable.
