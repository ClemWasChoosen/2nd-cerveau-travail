# RAG (Retrieval-Augmented Generation) - Guide Complet

## Table des Matières

1. [Introduction aux RAG](#1-introduction-aux-rag)
2. [Architecture et Principes Fondamentaux](#2-architecture-et-principes-fondamentaux)
3. [Aspects Mathématiques](#3-aspects-mathématiques)
4. [Implémentation Pratique](#4-implémentation-pratique)
5. [Optimisations Avancées](#5-optimisations-avancées)
6. [Évaluation et Métriques](#6-évaluation-et-métriques)
7. [Cas d'Usage Réels](#7-cas-dusage-réels)
8. [Sources et Références](#8-sources-et-références)

---

## 1. Introduction aux RAG

### 1.1 Qu'est-ce qu'un RAG ?

**RAG (Retrieval-Augmented Generation)** est une architecture qui combine les mécanismes de recherche (retrieval) avec des modèles de langage génératifs pour améliorer la précision et la fiabilité des réponses générées.

**Définition formelle :**

Un système RAG est une fonction $$f: Q \rightarrow A$$ qui, pour une question $$Q$$, génère une réponse $$A$$ en deux étapes :

1. **Retrieval** : $$R(Q) = \{d_1, d_2, ..., d_k\}$$ où $$d_i$$ sont les documents les plus pertinents
2. **Generation** : $$G(Q, R(Q)) = A$$ où $$A$$ est la réponse générée

### 1.2 Pourquoi utiliser un RAG ?

#### Problèmes résolus par les RAG :

**1. Hallucinations des LLMs**
- Les LLMs peuvent inventer des informations non-factuelles
- Les RAG ancrent les réponses dans des documents réels et vérifiables

**2. Connaissances limitées**
- Les LLMs sont figés dans le temps (date de leur entraînement)
- Les RAG permettent d'accéder à des informations actualisées

**3. Domaines spécialisés**
- Les LLMs ont des connaissances générales
- Les RAG permettent d'injecter des connaissances spécifiques à un domaine (médical, légal, entreprise, etc.)

**4. Traçabilité**
- Les réponses des LLMs seuls sont difficiles à vérifier
- Les RAG fournissent des sources citables et vérifiables

#### Exemple concret :

**Sans RAG :**
- Question : "Quelle est notre politique de remboursement ?"
- Réponse LLM : "Généralement, les entreprises offrent 30 jours..." (réponse générique et potentiellement fausse)

**Avec RAG :**
- Question : "Quelle est notre politique de remboursement ?"
- Retrieval : Récupère le document officiel de politique de remboursement
- Réponse : "Selon notre politique (document XYZ), nous offrons un remboursement complet de 30 jours sans frais supplémentaires." (réponse précise et sourcée)

### 1.3 Comparaison : RAG vs Fine-tuning vs Prompting

| Critère | RAG | Fine-tuning | Prompting Simple |
|---------|-----|-------------|------------------|
| **Coût** | Moyen | Élevé | Faible |
| **Mise à jour** | Facile (ajout de documents) | Difficile (ré-entraînement) | N/A |
| **Précision domaine** | Élevée | Très élevée | Faible |
| **Traçabilité** | Excellente | Faible | Faible |
| **Latence** | Moyenne | Faible | Faible |

**Pourquoi choisir RAG ?**
- Besoin de sources vérifiables
- Données changeant fréquemment
- Budget limité pour le fine-tuning
- Besoin de contrôle sur les connaissances utilisées

---

## 2. Architecture et Principes Fondamentaux

### 2.1 Vue d'ensemble du pipeline RAG

Un système RAG comprend 4 phases principales :

```
┌──────────────┐
│   Question   │
│      Q       │
└──────┬───────┘
       │
       ▼
┌──────────────────────────┐
│    1. INDEXATION         │
│  (Preprocessing)         │
│  - Chunking              │
│  - Embedding             │
│  - Stockage Vector DB    │
└──────────┬───────────────┘
           │
           ▼
┌──────────────────────────┐
│    2. RETRIEVAL          │
│  - Embedding de Q        │
│  - Similarity Search     │
│  - Top-K retrieval       │
│  - Reranking (optionnel) │
└──────────┬───────────────┘
           │
           ▼
┌──────────────────────────┐
│    3. AUGMENTATION       │
│  - Construction prompt   │
│  - Q + Context           │
└──────────┬───────────────┘
           │
           ▼
┌──────────────────────────┐
│    4. GENERATION         │
│  - LLM génère réponse    │
│  - Post-processing       │
└──────────┬───────────────┘
           │
           ▼
┌──────────────┐
│   Réponse    │
│      A       │
└──────────────┘
```

### 2.2 Composants détaillés

#### 2.2.1 Indexation (Preprocessing)

**Objectif :** Préparer les documents pour la recherche vectorielle

**Étapes :**

1. **Chunking** : Découpage des documents en morceaux
   - Pourquoi ? Les embeddings ont des limites de tokens
   - Stratégies : Fixed-size, sémantique, récursif, adaptatif

2. **Embedding** : Conversion en vecteurs numériques
   - Pourquoi ? Permet la recherche par similarité sémantique
   - Modèles populaires : OpenAI `text-embedding-3-large`, `bge-large`, `e5-large`

3. **Stockage** : Sauvegarde dans une base vectorielle
   - Pourquoi ? Permet des recherches rapides sur millions de documents
   - Solutions : Pinecone, Qdrant, Chroma, Weaviate

#### 2.2.2 Retrieval (Recherche)

**Objectif :** Trouver les documents les plus pertinents pour la question

**Processus mathématique :**

1. Embedding de la question : $$q = E(Q)$$ où $$E$$ est le modèle d'embedding

2. Calcul de similarité avec tous les documents :
   $$sim(q, d_i) = \frac{q \cdot d_i}{||q|| \cdot ||d_i||}$$ (similarité cosinus)

3. Sélection Top-K :
   $$R(Q) = \{d_i | sim(q, d_i) \geq \tau \text{ ou } i \in TopK\}$$

**Pourquoi la similarité cosinus ?**
- Indépendante de la magnitude des vecteurs
- Mesure l'orientation (direction sémantique)
- Efficace computationnellement

#### 2.2.3 Augmentation

**Objectif :** Construire un prompt enrichi pour le LLM

**Template typique :**

```
You are a helpful assistant. Use the information provided in the context below to answer the user's question accurately.

Context:
{retrieved_documents}

Question:
{user_question}

Instructions:
- Base your answer strictly on the provided context
- If the answer is not in the context, say "I don't know based on the provided information"
- Cite sources when possible

Answer:
```

**Pourquoi cette structure ?**
- Sépare clairement le contexte de la question
- Donne des instructions explicites pour éviter les hallucinations
- Encourage la traçabilité

#### 2.2.4 Generation

**Objectif :** Produire la réponse finale

**Paramètres importants :**
- **Temperature** : Contrôle la créativité
  - Basse (0.1-0.3) : Réponses déterministes pour QA
  - Haute (0.7-1.0) : Réponses créatives pour génération
- **Max tokens** : Longueur maximale de la réponse
- **Top-p** : Contrôle la diversité du vocabulaire

---

## 3. Aspects Mathématiques

### 3.1 Embeddings et Espaces Vectoriels

#### 3.1.1 Qu'est-ce qu'un embedding ?

Un **embedding** est une représentation vectorielle dense d'un texte dans un espace de dimension $$d$$ (typiquement $$d = 768, 1024, 1536, 3072$$).

**Formellement :**
$$E: \text{Texte} \rightarrow \mathbb{R}^d$$

**Propriétés importantes :**
1. **Proximité sémantique** : Textes similaires → vecteurs proches
2. **Compositionnalité** : Relations sémantiques peuvent être capturées par opérations vectorielles

**Exemple :**
```python
from openai import OpenAI

client = OpenAI()

# Créer des embeddings
text1 = "Le chat dort sur le canapé"
text2 = "Un félin se repose sur le sofa"
text3 = "La voiture roule sur l'autoroute"

embedding1 = client.embeddings.create(
    input=text1,
    model="text-embedding-3-large"
).data[0].embedding

embedding2 = client.embeddings.create(
    input=text2,
    model="text-embedding-3-large"
).data[0].embedding

embedding3 = client.embeddings.create(
    input=text3,
    model="text-embedding-3-large"
).data[0].embedding

# Calculer les similarités
import numpy as np

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

print(f"Sim(text1, text2): {cosine_similarity(embedding1, embedding2):.3f}")  # ~0.85
print(f"Sim(text1, text3): {cosine_similarity(embedding1, embedding3):.3f}")  # ~0.45
```

**Pourquoi ça marche ?**
- Les modèles d'embedding sont entraînés sur des paires de textes similaires/dissimilaires
- Ils apprennent à projeter des concepts sémantiques similaires dans des régions proches de l'espace vectoriel

### 3.2 Métriques de Similarité

#### 3.2.1 Similarité Cosinus

**Formule :**
$`\text{cosine\_sim}(a, b) = \frac{a \cdot b}{||a|| \cdot ||b||} = \frac{\sum_{i=1}^{d} a_i b_i}{\sqrt{\sum_{i=1}^{d} a_i^2} \cdot \sqrt{\sum_{i=1}^{d} b_i^2}}`$

**Propriétés :**
- Valeurs : $$[-1, 1]$$
- $$1$$ = vecteurs identiques en direction
- $$0$$ = vecteurs orthogonaux (aucune relation)
- $$-1$$ = vecteurs opposés

**Pourquoi pour les embeddings ?**
- Les embeddings sont normalisés ($$||v|| \approx 1$$)
- La direction (sémantique) est plus importante que la magnitude

#### 3.2.2 Distance Euclidienne

**Formule :**
$`\text{euclidean\_dist}(a, b) = \sqrt{\sum_{i=1}^{d} (a_i - b_i)^2}`$

**Propriétés :**
- Valeurs : $$[0, \infty)$$
- $$0$$ = vecteurs identiques
- Plus grande = plus différents

**Comparaison :**
```python
import numpy as np

a = np.array([1, 0, 0])
b = np.array([0.7, 0.7, 0])
c = np.array([2, 0, 0])

# Cosinus : ne change pas avec la magnitude
cos_ab = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
cos_ac = np.dot(a, c) / (np.linalg.norm(a) * np.linalg.norm(c))
print(f"Cosine sim(a,b): {cos_ab:.3f}")  # 0.707
print(f"Cosine sim(a,c): {cos_ac:.3f}")  # 1.000 (même direction!)

# Euclidienne : sensible à la magnitude
euc_ab = np.linalg.norm(a - b)
euc_ac = np.linalg.norm(a - c)
print(f"Euclidean dist(a,b): {euc_ab:.3f}")  # 1.000
print(f"Euclidean dist(a,c): {euc_ac:.3f}")  # 1.000
```

**Quand utiliser quoi ?**
- **Cosinus** : Embeddings normalisés (cas standard en RAG)
- **Euclidienne** : Quand la magnitude compte (rare en NLP)

#### 3.2.3 Dot Product

**Formule :**
$`\text{dot\_product}(a, b) = \sum_{i=1}^{d} a_i b_i`$

**Relation avec cosinus :**
Si $$a$$ et $$b$$ sont normalisés ($$||a|| = ||b|| = 1$$) :
$`\text{dot\_product}(a, b) = \text{cosine\_sim}(a, b)`$

**Pourquoi utiliser le dot product ?**
- Plus rapide à calculer (pas de normalisation)
- Équivalent au cosinus si embeddings pré-normalisés

### 3.3 Approximate Nearest Neighbor (ANN)

#### Problème à résoudre

Pour $$N$$ documents et $$d$$ dimensions :
- Recherche exacte : $$O(N \cdot d)$$ - trop lent pour $$N$$ grand
- Besoin d'une approche approximative mais rapide

#### HNSW (Hierarchical Navigable Small World)

**Concept :**
- Graphe multi-couches avec des connexions "small world"
- Recherche commence en haut (peu de nœuds) et descend progressivement

**Complexité :**
- Construction : $$O(N \log N)$$
- Recherche : $$O(\log N)$$

**Pourquoi ça fonctionne ?**
- Exploite la propriété que les vecteurs similaires forment des clusters
- Les couches supérieures permettent des "sauts" rapides entre clusters
- Les couches inférieures permettent un raffinement local

**Trade-off :**
- Précision vs Vitesse
- Contrôlé par les paramètres `ef_search` et `M` (nombre de connexions)

---

## 4. Implémentation Pratique

### 4.1 RAG Simple avec LangChain

```python
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader

# 1. CHARGEMENT ET CHUNKING
loader = TextLoader("mes_documents.txt")
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,      # Taille des chunks en caractères
    chunk_overlap=200,    # Chevauchement pour préserver contexte
    separators=["\n\n", "\n", ".", " ", ""]  # Séparateurs par ordre de priorité
)
chunks = text_splitter.split_documents(documents)
print(f"Nombre de chunks créés: {len(chunks)}")

# 2. EMBEDDING ET INDEXATION
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large",
    dimensions=1536  # Dimension des vecteurs
)

# Création de la base vectorielle
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"  # Sauvegarde locale
)

# 3. CRÉATION DU RETRIEVER
retriever = vectorstore.as_retriever(
    search_type="similarity",  # Ou "mmr" pour Maximum Marginal Relevance
    search_kwargs={"k": 4}     # Récupérer top-4 documents
)

# 4. CRÉATION DE LA CHAÎNE QA
llm = OpenAI(temperature=0.2, model="gpt-4")

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",  # "stuff", "map_reduce", "refine", "map_rerank"
    retriever=retriever,
    return_source_documents=True  # Retourner les sources
)

# 5. UTILISATION
question = "Quelle est notre politique de remboursement ?"
result = qa_chain({"query": question})

print(f"Question: {question}")
print(f"Réponse: {result['result']}")
print(f"\nSources:")
for i, doc in enumerate(result['source_documents']):
    print(f"  [{i+1}] {doc.page_content[:100]}...")
```

**Explication des paramètres importants :**

- `chunk_size=1000` : Pourquoi 1000 ?
  - Assez grand pour capturer du contexte complet
  - Assez petit pour rester pertinent et dans les limites des embeddings
  - Règle générale : 500-1500 caractères selon le domaine

- `chunk_overlap=200` : Pourquoi un overlap ?
  - Évite de couper des informations importantes à la frontière
  - 10-20% du chunk_size est une bonne pratique

- `k=4` : Pourquoi 4 documents ?
  - Trade-off : Plus de contexte vs bruit
  - Dépend de la taille des chunks et du context window du LLM
  - Tester empiriquement : 3-5 est souvent optimal

### 4.2 RAG Custom avec Contrôle Total

```python
import openai
from typing import List, Dict
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

class CustomRAG:
    def __init__(self, collection_name: str = "my_docs"):
        self.client = QdrantClient(":memory:")  # Ou url="http://localhost:6333" pour Qdrant serveur
        self.collection_name = collection_name
        self.embedding_model = "text-embedding-3-large"
        self.embedding_dimension = 1536
        
        # Créer la collection
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=self.embedding_dimension,
                distance=Distance.COSINE
            )
        )
    
    def chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """Découpe le texte en chunks avec chevauchement"""
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            
            # Essayer de couper à une phrase complète
            if end < len(text):
                last_period = chunk.rfind('.')
                if last_period > chunk_size * 0.5:  # Au moins 50% du chunk
                    end = start + last_period + 1
                    chunk = text[start:end]
            
            chunks.append(chunk.strip())
            start = end - overlap
        
        return chunks
    
    def embed_text(self, text: str) -> List[float]:
        """Crée l'embedding d'un texte"""
        response = openai.embeddings.create(
            input=text,
            model=self.embedding_model
        )
        return response.data[0].embedding
    
    def index_documents(self, documents: List[str]):
        """Indexe une liste de documents"""
        points = []
        
        for doc_id, doc in enumerate(documents):
            # Chunking
            chunks = self.chunk_text(doc)
            
            # Embedding et ajout à Qdrant
            for chunk_id, chunk in enumerate(chunks):
                embedding = self.embed_text(chunk)
                
                point = PointStruct(
                    id=doc_id * 1000 + chunk_id,  # ID unique
                    vector=embedding,
                    payload={
                        "text": chunk,
                        "doc_id": doc_id,
                        "chunk_id": chunk_id
                    }
                )
                points.append(point)
        
        # Upload en batch pour performance
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
        
        print(f"✅ {len(points)} chunks indexés")
    
    def retrieve(self, query: str, top_k: int = 4) -> List[Dict]:
        """Récupère les documents les plus pertinents"""
        # Embedding de la requête
        query_embedding = self.embed_text(query)
        
        # Recherche dans Qdrant
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=top_k,
            with_payload=True
        )
        
        # Formater les résultats
        retrieved_docs = []
        for result in results:
            retrieved_docs.append({
                "text": result.payload["text"],
                "score": result.score,
                "doc_id": result.payload["doc_id"],
                "chunk_id": result.payload["chunk_id"]
            })
        
        return retrieved_docs
    
    def generate(self, query: str, context: List[Dict]) -> str:
        """Génère une réponse avec le LLM"""
        # Construire le contexte
        context_text = "\n\n".join([
            f"[Document {doc['doc_id']}, chunk {doc['chunk_id']}, score: {doc['score']:.3f}]\n{doc['text']}"
            for doc in context
        ])
        
        # Construire le prompt
        prompt = f"""Vous êtes un assistant serviable. Utilisez les informations du contexte ci-dessous pour répondre à la question de l'utilisateur de manière précise.

Contexte:
{context_text}

Question:
{query}

Instructions:
- Basez votre réponse strictement sur le contexte fourni
- Si la réponse n'est pas dans le contexte, dites "Je ne sais pas selon les informations fournies"
- Citez les sources quand possible (ex: "Selon le document X...")

Réponse:"""
        
        # Appel au LLM
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "Vous êtes un assistant serviable et précis."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=500
        )
        
        return response.choices[0].message.content
    
    def query(self, question: str, top_k: int = 4) -> Dict:
        """Pipeline complet RAG"""
        # 1. Retrieval
        retrieved_docs = self.retrieve(question, top_k=top_k)
        
        # 2. Generation
        answer = self.generate(question, retrieved_docs)
        
        return {
            "question": question,
            "answer": answer,
            "sources": retrieved_docs
        }

# UTILISATION
rag = CustomRAG(collection_name="company_docs")

# Indexer des documents
documents = [
    "Notre politique de remboursement offre un remboursement complet sous 30 jours sans frais supplémentaires.",
    "Pour les retours, contactez notre service client au 01 23 45 67 89.",
    "Les produits défectueux sont remplacés gratuitement dans les 90 jours."
]

rag.index_documents(documents)

# Poser une question
result = rag.query("Quelle est la politique de remboursement ?")

print(f"Question: {result['question']}")
print(f"\nRéponse: {result['answer']}")
print(f"\nSources:")
for i, source in enumerate(result['sources']):
    print(f"  [{i+1}] Score: {source['score']:.3f}")
    print(f"      {source['text'][:100]}...")
```

**Pourquoi cette architecture ?**
1. **Modularité** : Chaque étape (chunking, embedding, retrieval, generation) est séparée
2. **Flexibilité** : Facile de changer un composant (ex: remplacer Qdrant par Pinecone)
3. **Contrôle** : Accès complet aux paramètres et métadonnées
4. **Debugging** : Peut inspecter chaque étape du pipeline

### 4.3 Frameworks et Outils

#### Comparaison des solutions

| Outil | Avantages | Inconvénients | Use Case |
|-------|-----------|---------------|----------|
| **LangChain** | Abstraction complète, intégrations multiples | Peut être verbose, breaking changes | Prototypage rapide |
| **LlamaIndex** | Optimisé pour RAG, excellente indexation | Moins flexible | RAG complexe |
| **Haystack** | Production-ready, pipelines | Courbe d'apprentissage | Production |
| **Custom** | Contrôle total | Plus de code | Besoins spécifiques |

#### Bases de données vectorielles

| DB | Type | Avantages | Inconvénients |
|----|------|-----------|---------------|
| **Pinecone** | Cloud | Managed, scalable, simple | Coût, vendor lock-in |
| **Qdrant** | Self-hosted/Cloud | Rapide, filtres puissants | Setup initial |
| **Chroma** | Embedded | Facile, gratuit, local | Pas pour production scale |
| **Weaviate** | Self-hosted/Cloud | GraphQL, modules ML | Complexité |
| **Milvus** | Self-hosted/Cloud | Très scalable | Configuration complexe |

---

## 5. Optimisations Avancées

### 5.1 Stratégies de Chunking

Le chunking est **critique** car il détermine la qualité de votre retrieval.

#### 5.1.1 Fixed-Size Chunking

**Principe :** Découper en morceaux de taille fixe

```python
def fixed_size_chunking(text: str, chunk_size: int = 1000, overlap: int = 200):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks
```

**Avantages :**
- Simple et rapide
- Chunks de taille uniforme

**Inconvénients :**
- Peut couper des phrases/paragraphes
- Ignore la structure sémantique

**Quand utiliser :**
- Textes uniformes (logs, transcriptions)
- Prototypage rapide

#### 5.1.2 Semantic Chunking

**Principe :** Découper aux frontières sémantiques (phrases, paragraphes)

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", ". ", " ", ""]  # Hiérarchie de séparateurs
)

# Essaie d'abord de couper aux paragraphes (\n\n)
# Si trop grand, essaie phrases (.)
# Si encore trop grand, mots ( )
chunks = splitter.split_text(text)
```

**Avantages :**
- Préserve le sens
- Meilleure cohérence

**Inconvénients :**
- Chunks de taille variable
- Plus complexe

**Quand utiliser :**
- Documents structurés (articles, docs)
- Qualité > uniformité

#### 5.1.3 Pourquoi le chunking est important ?

**Exemple concret :**

Document original :

Notre politique de remboursement est la suivante : tous les clients peuvent demander un remboursement complet dans les 30 jours suivant l'achat. Aucun frais supplémentaire ne sera appliqué. Pour les retours après 30 jours, un crédit magasin sera offert.

**Mauvais chunking (coupe en plein milieu) :**
- Chunk 1 : "Notre politique de remboursement est la suivante : tous les clients peuvent demander un remboursement complet dans les 30 jours suivant l'achat. Aucun frais supplé"
- Chunk 2 : "mentaire ne sera appliqué. Pour les retours après 30 jours, un crédit magasin sera offert."

→ Information fragmentée, contexte perdu

**Bon chunking (frontières sémantiques) :**
- Chunk 1 : "Notre politique de remboursement est la suivante : tous les clients peuvent demander un remboursement complet dans les 30 jours suivant l'achat. Aucun frais supplémentaire ne sera appliqué."
- Chunk 2 : "Pour les retours après 30 jours, un crédit magasin sera offert."

→ Information complète et contextuelle

### 5.2 Hybrid Search

**Problème :** La recherche vectorielle pure manque parfois les correspondances exactes de mots-clés.

**Solution :** Combiner recherche vectorielle (sémantique) + recherche lexicale (BM25)

#### BM25 (Best Matching 25)

**Formule :**
$$\text{score}(D, Q) = \sum_{i=1}^{n} IDF(q_i) \cdot \frac{f(q_i, D) \cdot (k_1 + 1)}{f(q_i, D) + k_1 \cdot (1 - b + b \cdot \frac{|D|}{avgdl})}$$

Où :
- $$f(q_i, D)$$ = fréquence du terme $$q_i$$ dans le document $$D$$
- $$|D|$$ = longueur du document
- $$avgdl$$ = longueur moyenne des documents
- $$k_1, b$$ = paramètres (typiquement $$k_1=1.5, b=0.75$$)

**Pourquoi BM25 ?**
- Capture les correspondances exactes de mots-clés
- Bon pour les noms propres, acronymes, codes
- Complément à la recherche sémantique

#### Implémentation Hybrid Search

```python
from rank_bm25 import BM25Okapi
import numpy as np

class HybridRetriever:
    def __init__(self, documents: List[str]):
        self.documents = documents
        
        # BM25
        tokenized_docs = [doc.lower().split() for doc in documents]
        self.bm25 = BM25Okapi(tokenized_docs)
        
        # Vector search (supposons embeddings pré-calculés)
        self.embeddings = [embed_text(doc) for doc in documents]
    
    def retrieve(self, query: str, top_k: int = 10, alpha: float = 0.5):
        """
        alpha : poids de la recherche vectorielle (1-alpha pour BM25)
        """
        # BM25 scores
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        
        # Normaliser BM25 scores [0, 1]
        bm25_scores = (bm25_scores - bm25_scores.min()) / (bm25_scores.max() - bm25_scores.min() + 1e-6)
        
        # Vector search scores
        query_embedding = embed_text(query)
        vector_scores = np.array([
            cosine_similarity(query_embedding, doc_emb) 
            for doc_emb in self.embeddings
        ])
        
        # Combiner les scores
        hybrid_scores = alpha * vector_scores + (1 - alpha) * bm25_scores
        
        # Top-K
        top_indices = np.argsort(hybrid_scores)[-top_k:][::-1]
        
        return [(self.documents[i], hybrid_scores[i]) for i in top_indices]

# UTILISATION
retriever = HybridRetriever(documents)
results = retriever.retrieve(
    "remboursement 30 jours",
    top_k=3,
    alpha=0.7  # 70% vectoriel, 30% BM25
)
```

**Quand utiliser Hybrid Search ?**
- Documents techniques avec beaucoup d'acronymes
- Recherche de noms propres
- Domaines où les mots-clés exacts sont importants (légal, médical)

**Pourquoi le paramètre alpha ?**
- $$\alpha = 1$$ : Uniquement vectoriel (sémantique pure)
- $$\alpha = 0$$ : Uniquement BM25 (mots-clés)
- $$\alpha = 0.5-0.7$$ : Bon équilibre général

### 5.3 Reranking

**Problème :** La recherche vectorielle peut manquer de nuances contextuelles.

**Solution :** Réordonnancer les résultats avec un modèle cross-encoder.

#### Cross-Encoder vs Bi-Encoder

**Bi-Encoder (utilisé pour retrieval initial) :**
- Encode query et documents séparément
- Rapide : $$O(N)$$ pour $$N$$ documents
- Mais : pas d'interaction entre query et document

**Cross-Encoder (pour reranking) :**
- Encode query + document ensemble
- Lent : $$O(N)$$ appels au modèle
- Mais : meilleure compréhension de la pertinence

```python
from sentence_transformers import CrossEncoder

class RerankedRetriever:
    def __init__(self):
        self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    
    def rerank(self, query: str, documents: List[str], top_k: int = 3):
        # Créer des paires (query, doc)
        pairs = [[query, doc] for doc in documents]
        
        # Scorer avec le cross-encoder
        scores = self.cross_encoder.predict(pairs)
        
        # Réordonnancer
        ranked_indices = np.argsort(scores)[-top_k:][::-1]
        
        return [(documents[i], scores[i]) for i in ranked_indices]

# PIPELINE COMPLET
# 1. Retrieval vectoriel : récupérer top-20
initial_results = vector_search(query, top_k=20)

# 2. Reranking : affiner à top-3
reranker = RerankedRetriever()
final_results = reranker.rerank(
    query,
    [doc['text'] for doc in initial_results],
    top_k=3
)
```

**Pourquoi ce pipeline en 2 étapes ?**
1. **Retrieval vectoriel** : Rapide, filtre $$10^6$$ docs → 20
2. **Reranking** : Précis, affine 20 → 3

**Impact sur la performance :**
- +10-20% de précision (mesurée par MRR@3)
- Coût : +100-200ms de latence

### 5.4 Query Expansion

**Problème :** La question de l'utilisateur peut être mal formulée ou ambiguë.

**Solution :** Enrichir la requête avec des reformulations.

#### Technique 1 : Multi-Query

```python
def generate_multi_queries(original_query: str) -> List[str]:
    """Génère plusieurs versions de la requête"""
    prompt = f"""Générez 3 versions différentes de cette question pour améliorer la recherche :

Question originale : {original_query}

Versions alternatives (une par ligne) :"""

    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.8  # Un peu de créativité
    )
    
    queries = [original_query]
    queries.extend(response.choices[0].message.content.strip().split('\n'))
    
    return queries

# UTILISATION
original = "politique remboursement"
expanded_queries = generate_multi_queries(original)
# ['politique remboursement', 
#  'quelles sont les conditions de retour',
#  'comment obtenir un remboursement',
#  'délai pour retourner un produit']

# Rechercher avec toutes les queries et fusionner résultats
all_results = []
for query in expanded_queries:
    results = retrieve(query, top_k=5)
    all_results.extend(results)

# Dédupliquer et réordonnancer
final_results = deduplicate_and_rank(all_results, top_k=5)
```

**Pourquoi ça marche ?**
- Capture différentes façons de poser la même question
- Augmente le recall (probabilité de trouver les bons documents)

#### Technique 2 : HyDE (Hypothetical Document Embeddings)

**Principe :** Générer une réponse hypothétique et l'utiliser pour la recherche.

```python
def hyde_retrieval(query: str):
    # 1. Générer une réponse hypothétique
    prompt = f"""Générez une réponse détaillée et informative à cette question, même si vous n'êtes pas sûr :

Question : {query}

Réponse hypothétique :"""

    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    
    hypothetical_answer = response.choices[0].message.content
    
    # 2. Utiliser la réponse pour la recherche (plus proche des documents réels)
    results = retrieve(hypothetical_answer, top_k=5)
    
    return results
```

**Pourquoi c'est puissant ?**
- Les réponses hypothétiques ressemblent sémantiquement aux vrais documents
- Meilleur matching que la question brute
- Particulièrement utile pour les questions complexes

---

## 6. Évaluation et Métriques

L'évaluation d'un RAG nécessite de tester **séparément** le retriever et le generator.

### 6.1 Métriques pour le Retriever

#### 6.1.1 Contextual Relevancy

**Définition :** Proportion des documents récupérés qui sont pertinents pour la question.

**Formule :**
$$\text{Contextual Relevancy} = \frac{\text{Nombre de docs pertinents récupérés}}{\text{Nombre total de docs récupérés}}$$

**Calcul avec LLM-as-a-judge :**

```python
def compute_contextual_relevancy(query: str, retrieved_docs: List[str]) -> float:
    relevant_count = 0
    
    for doc in retrieved_docs:
        # Demander au LLM si le doc est pertinent
        prompt = f"""Est-ce que ce document est pertinent pour répondre à la question ?

Question : {query}

Document : {doc}

Répondez uniquement par OUI ou NON."""

        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        
        if "OUI" in response.choices[0].message.content.upper():
            relevant_count += 1
    
    return relevant_count / len(retrieved_docs)

# EXEMPLE
query = "Quelle est la politique de remboursement ?"
retrieved_docs = [
    "Nous offrons un remboursement complet sous 30 jours.",  # Pertinent
    "Contactez le service client au 01 23 45 67 89.",       # Pertinent
    "Nos horaires d'ouverture sont 9h-18h."                 # NON pertinent
]

score = compute_contextual_relevancy(query, retrieved_docs)
print(f"Contextual Relevancy: {score:.2f}")  # 0.67
```

**Pourquoi cette métrique ?**
- Mesure la précision du retriever
- Détecte le bruit (documents non pertinents)

#### 6.1.2 Contextual Recall

**Définition :** Proportion d'informations de la réponse idéale qui sont présentes dans les documents récupérés.

**Formule :**
$$\text{Contextual Recall} = \frac{\text{Infos de la réponse idéale trouvées}}{\text{Total infos dans réponse idéale}}$$

```python
def compute_contextual_recall(expected_answer: str, retrieved_docs: List[str]) -> float:
    """Nécessite une réponse de référence (ground truth)"""
    
    # Extraire les faits de la réponse attendue
    prompt = f"""Extrayez tous les faits et informations clés de cette réponse sous forme de liste :

Réponse : {expected_answer}

Liste des faits (un par ligne) :"""

    response = openai.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
    
    expected_facts = response.choices[0].message.content.strip().split('\n')
    
    # Vérifier quels faits sont dans les docs récupérés
    context = "\n\n".join(retrieved_docs)
    found_facts = 0
    
    for fact in expected_facts:
        prompt = f"""Est-ce que cette information est présente dans le contexte ?

Information : {fact}

Contexte : {context}

Répondez OUI ou NON."""

        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        
        if "OUI" in response.choices[0].message.content.upper():
            found_facts += 1
    
    return found_facts / len(expected_facts)

# EXEMPLE
expected_answer = "Nous offrons un remboursement complet sous 30 jours sans frais."
retrieved_docs = ["Remboursement possible sous 30 jours."]  # Manque "sans frais"

score = compute_contextual_recall(expected_answer, retrieved_docs)
print(f"Contextual Recall: {score:.2f}")  # < 1.0 car info manquante
```

**Pourquoi cette métrique ?**
- Mesure la complétude du retrieval
- Détecte les informations manquantes

#### 6.1.3 Contextual Precision

**Définition :** Les documents pertinents sont-ils classés en premier ?

**Formule :**
$$\text{Contextual Precision@K} = \frac{1}{K} \sum_{k=1}^{K} \frac{\text{Docs pertinents dans top-k}}{k}$$

**Pourquoi important ?**
- L'ordre des résultats compte
- Un bon doc en position 10 est moins utile qu'en position 1

### 6.2 Métriques pour le Generator

#### 6.2.1 Answer Relevancy

**Définition :** La réponse est-elle pertinente pour la question ?

```python
def compute_answer_relevancy(query: str, answer: str) -> float:
    """Mesure si la réponse répond bien à la question"""
    
    prompt = f"""Sur une échelle de 0 à 1, évaluez si cette réponse est pertinente pour la question :

Question : {query}

Réponse : {answer}

Score de pertinence (0-1) :"""

    response = openai.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    
    # Extraire le score
    score_text = response.choices[0].message.content.strip()
    try:
        score = float(score_text)
    except:
        # Parsing plus robuste si nécessaire
        score = 0.5
    
    return score
```

#### 6.2.2 Faithfulness (Non-hallucination)

**Définition :** La réponse est-elle fidèle au contexte fourni ? (pas d'hallucinations)

```python
def compute_faithfulness(answer: str, context: List[str]) -> float:
    """Détecte les hallucinations"""
    
    # Extraire les affirmations de la réponse
    prompt = f"""Extrayez toutes les affirmations factuelles de cette réponse :

Réponse : {answer}

Affirmations (une par ligne) :"""

    response = openai.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
    
    claims = response.choices[0].message.content.strip().split('\n')
    
    # Vérifier chaque affirmation dans le contexte
    context_text = "\n\n".join(context)
    supported_claims = 0
    
    for claim in claims:
        prompt = f"""Cette affirmation est-elle supportée par le contexte ?

Affirmation : {claim}

Contexte : {context_text}

Répondez OUI (supportée), NON (contredite), ou INCONNU (pas d'info)."""

        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        
        result = response.choices[0].message.content.upper()
        if "OUI" in result:
            supported_claims += 1
    
    return supported_claims / len(claims) if claims else 1.0

# EXEMPLE
answer = "Nous offrons un remboursement complet sous 30 jours sans frais supplémentaires."
context = ["Remboursement possible sous 30 jours."]  # Ne mentionne pas "sans frais"

score = compute_faithfulness(answer, context)
print(f"Faithfulness: {score:.2f}")  # < 1.0 car ajout d'info
```

**Pourquoi critique ?**
- Les hallucinations sont le problème #1 des RAG
- Une réponse fidèle mais incomplète > réponse complète mais fausse

### 6.3 Framework d'Évaluation avec DeepEval

```python
from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    ContextualRelevancyMetric,
    ContextualRecallMetric,
    ContextualPrecisionMetric
)

# Créer des cas de test
test_cases = []

questions = [
    "Quelle est la politique de remboursement ?",
    "Comment contacter le service client ?",
    "Quels sont les horaires ?"
]

for question in questions:
    # Exécuter le RAG
    retrieved_context = retrieve(question)
    actual_output = generate(question, retrieved_context)
    
    # Créer le test case
    test_case = LLMTestCase(
        input=question,
        actual_output=actual_output,
        retrieval_context=retrieved_context,
        expected_output="Réponse de référence"  # Si disponible
    )
    test_cases.append(test_case)

# Évaluer avec toutes les métriques
results = evaluate(
    test_cases=test_cases,
    metrics=[
        # Retriever
        ContextualRelevancyMetric(threshold=0.7),
        ContextualRecallMetric(threshold=0.7),
        ContextualPrecisionMetric(threshold=0.7),
        
        # Generator
        AnswerRelevancyMetric(threshold=0.7),
        FaithfulnessMetric(threshold=0.7)
    ]
)

# Afficher les résultats
print(results)
```

**Rapport type :**

Test Cases: 3  
✓ Passed: 2  
✗ Failed: 1  
Metrics:  

    Contextual Relevancy: 0.85 ± 0.10
    Contextual Recall: 0.78 ± 0.12
    Contextual Precision: 0.82 ± 0.08
    Answer Relevancy: 0.91 ± 0.06
    Faithfulness: 0.73 ± 0.15 ⚠️ Below threshold!

Failed Test Case:
Question: "Quels sont les horaires ?"
Issue: Faithfulness = 0.60 (threshold: 0.70)
Reason: Answer mentioned "24/7" but context only states "9h-18h"

### 6.4 Métriques de Production

En production, surveiller :

1. **Latence** : Temps de réponse total
   - Retrieval : < 200ms
   - Generation : < 2s
   - Total : < 3s

2. **Throughput** : Requêtes/seconde
   - Dépend de l'infrastructure
   - Optimiser avec batching

3. **Coût** : $$$/1000 requêtes
   - Embeddings : ~$0.10
   - LLM calls : ~$1-5
   - Vector DB : ~$0.05

4. **User Feedback** : Thumbs up/down
   - Taux de satisfaction > 80%
   - Utiliser pour améliorer le système

```python
import time
from typing import Dict

class RAGMonitor:
    def __init__(self):
        self.metrics = {
            'total_queries': 0,
            'total_latency': 0,
            'failures': 0,
            'positive_feedback': 0,
            'negative_feedback': 0
        }
    
    def log_query(self, latency: float, success: bool, feedback: int = 0):
        """
        feedback: 1 (positif), -1 (négatif), 0 (pas de feedback)
        """
        self.metrics['total_queries'] += 1
        self.metrics['total_latency'] += latency
        
        if not success:
            self.metrics['failures'] += 1
        
        if feedback == 1:
            self.metrics['positive_feedback'] += 1
        elif feedback == -1:
            self.metrics['negative_feedback'] += 1
    
    def get_stats(self) -> Dict:
        total = self.metrics['total_queries']
        if total == 0:
            return {}
        
        feedback_total = self.metrics['positive_feedback'] + self.metrics['negative_feedback']
        
        return {
            'avg_latency_ms': (self.metrics['total_latency'] / total) * 1000,
            'success_rate': 1 - (self.metrics['failures'] / total),
            'satisfaction_rate': (self.metrics['positive_feedback'] / feedback_total) if feedback_total > 0 else None,
            'total_queries': total
        }

# UTILISATION
monitor = RAGMonitor()

def query_with_monitoring(question: str, user_feedback: int = 0):
    start = time.time()
    try:
        result = rag.query(question)
        success = True
    except Exception as e:
        success = False
        result = None
    
    latency = time.time() - start
    monitor.log_query(latency, success, user_feedback)
    
    return result

# Après 1000 requêtes
stats = monitor.get_stats()
print(f"Latence moyenne: {stats['avg_latency_ms']:.2f} ms")
print(f"Taux de succès: {stats['success_rate']:.2%}")
print(f"Satisfaction: {stats['satisfaction_rate']:.2%}")
```

---

## 7. Cas d'Usage Réels

### 7.1 Chatbot Support Client

**Contexte :**
- Base de connaissances : FAQ, documentation produit, politiques
- Volume : 10,000+ documents
- Objectif : Répondre aux questions clients 24/7

**Architecture :**

```python
class CustomerSupportRAG:
    def __init__(self):
        self.rag = CustomRAG(collection_name="customer_support")
        
        # Charger et indexer la base de connaissances
        documents = self.load_knowledge_base()
        self.rag.index_documents(documents)
        
        # Configuration spécifique
        self.chunk_size = 500  # Chunks plus petits pour réponses précises
        self.top_k = 3         # Peu de contexte pour éviter confusion
    
    def load_knowledge_base(self):
        """Charger FAQ, docs, politiques"""
        docs = []
        
        # FAQ
        faq = load_faq_from_database()
        docs.extend(faq)
        
        # Documentation produit
        product_docs = load_product_docs_from_cms()
        docs.extend(product_docs)
        
        # Politiques (remboursement, livraison, etc.)
        policies = load_policies_from_files()
        docs.extend(policies)
        
        return docs
    
    def answer_customer_query(self, query: str, customer_id: str = None):
        """Répondre à une question client"""
        
        # Retrieval avec filtres si besoin
        retrieved = self.rag.retrieve(query, top_k=self.top_k)
        
        # Generation avec ton adapté
        prompt = f"""Vous êtes un agent du service client professionnel et amical.

Contexte :
{format_context(retrieved)}

Question client :
{query}

Instructions :
- Répondez de manière claire et concise
- Soyez empathique et professionnel
- Si vous ne savez pas, suggérez de contacter le support
- Incluez des liens vers la documentation si pertinent

Réponse :"""
        
        answer = self.generate(prompt)
        
        # Logger pour amélioration continue
        self.log_interaction(query, answer, retrieved, customer_id)
        
        return {
            'answer': answer,
            'sources': retrieved,
            'confidence': self.compute_confidence(retrieved)
        }
    
    def compute_confidence(self, retrieved_docs):
        """Score de confiance basé sur les similarités"""
        if not retrieved_docs:
            return 0.0
        
        avg_score = sum(doc['score'] for doc in retrieved_docs) / len(retrieved_docs)
        
        # Heuristique : confiance faible si scores < 0.7
        if avg_score < 0.7:
            return 'low'
        elif avg_score < 0.85:
            return 'medium'
        else:
            return 'high'
```

**Améliorations spécifiques :**

1. **Feedback Loop :**
```python
def collect_feedback(query: str, answer: str, helpful: bool):
    """Collecter le feedback utilisateur"""
    if not helpful:
        # Cas négatif : stocker pour analyse
        save_to_improvement_queue({
            'query': query,
            'answer': answer,
            'timestamp': datetime.now(),
            'helpful': False
        })
        
        # Trigger notification pour équipe
        notify_team_negative_feedback(query, answer)
```

2. **Escalade automatique :**
```python
def should_escalate(confidence: str, query: str) -> bool:
    """Décider si escalader vers agent humain"""
    
    # Toujours escalader si confiance faible
    if confidence == 'low':
        return True
    
    # Mots-clés sensibles
    sensitive_keywords = ['remboursement', 'annulation', 'plainte', 'problème']
    if any(kw in query.lower() for kw in sensitive_keywords):
        return True
    
    return False
```

### 7.2 Q&A sur Documents Internes (Entreprise)

**Contexte :**
- Documents : Rapports, présentations, emails, wikis internes
- Sécurité : Accès basé sur les permissions
- Objectif : Recherche d'informations instantanée

**Défis spécifiques :**

1. **Multi-formats** : PDF, Word, PowerPoint, etc.
2. **Permissions** : Alice peut voir doc X, Bob non
3. **Freshness** : Documents mis à jour fréquemment

**Architecture :**

```python
class EnterpriseRAG:
    def __init__(self):
        self.rag = CustomRAG(collection_name="enterprise_docs")
        
    def ingest_document(self, file_path: str, metadata: Dict):
        """Ingérer un nouveau document avec métadonnées"""
        
        # 1. Extraction du texte selon le format
        if file_path.endswith('.pdf'):
            text = extract_text_from_pdf(file_path)
        elif file_path.endswith('.docx'):
            text = extract_text_from_docx(file_path)
        elif file_path.endswith('.pptx'):
            text = extract_text_from_pptx(file_path)
        else:
            text = read_text_file(file_path)
        
        # 2. Chunking avec préservation de structure
        chunks = self.semantic_chunking(text)
        
        # 3. Enrichir avec métadonnées
        for chunk in chunks:
            chunk_metadata = {
                'source_file': file_path,
                'department': metadata.get('department'),
                'access_level': metadata.get('access_level', 'public'),
                'created_date': metadata.get('created_date'),
                'author': metadata.get('author'),
                'tags': metadata.get('tags', [])
            }
            
            # 4. Indexer avec métadonnées
            self.rag.index_chunk(chunk, chunk_metadata)
    
    def query_with_permissions(self, query: str, user_id: str):
        """Recherche avec respect des permissions"""
        
        # 1. Récupérer les permissions utilisateur
        user_permissions = get_user_permissions(user_id)
        
        # 2. Retrieval avec filtres
        retrieved = self.rag.retrieve(
            query,
            filters={
                'access_level': {'$in': user_permissions}
            },
            top_k=5
        )
        
        # 3. Generation
        answer = self.generate(query, retrieved)
        
        return {
            'answer': answer,
            'sources': self.format_sources(retrieved)
        }
    
    def format_sources(self, retrieved_docs):
        """Formater les sources pour affichage"""
        sources = []
        for doc in retrieved_docs:
            sources.append({
                'title': doc.metadata['source_file'].split('/')[-1],
                'author': doc.metadata['author'],
                'date': doc.metadata['created_date'],
                'excerpt': doc.text[:200] + '...',
                'url': generate_doc_link(doc.metadata['source_file'])
            })
        return sources
```

**Fonctionnalités avancées :**

1. **Automatic Refresh :**
```python
def watch_document_changes():
    """Surveiller les changements de documents et réindexer"""
    
    # Utiliser un file watcher (ex: watchdog)
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler
    
    class DocumentHandler(FileSystemEventHandler):
        def on_modified(self, event):
            if not event.is_directory:
                print(f"Document modifié: {event.src_path}")
                
                # Supprimer anciens chunks
                rag.delete_document(event.src_path)
                
                # Réindexer
                metadata = load_metadata(event.src_path)
                rag.ingest_document(event.src_path, metadata)
    
    observer = Observer()
    observer.schedule(DocumentHandler(), path='/docs', recursive=True)
    observer.start()
```

2. **Smart Summarization :**
```python
def summarize_multiple_sources(query: str, sources: List[Dict]):
    """Synthétiser info de plusieurs sources"""
    
    prompt = f"""Synthétisez les informations de plusieurs sources pour répondre à : {query}

Sources :
{format_sources_with_citations(sources)}

Créez une réponse unifiée en :
1. Intégrant toutes les infos pertinentes
2. Citant les sources [1], [2], etc.
3. Signalant les contradictions éventuelles

Synthèse :"""

    return generate(prompt)
```

### 7.3 Autres Cas d'Usage

#### Legal Document Analysis
- Recherche de jurisprudence
- Analyse de contrats
- Vérification de conformité

#### Medical Q&A
- Recherche de littérature médicale
- Aide au diagnostic (avec disclaimers)
- Drug interactions

#### Code Documentation
- Recherche dans le codebase
- API documentation
- Troubleshooting

---

## 8. Sources et Références

### 8.1 Articles Scientifiques

1. **RAG: Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks**
   - Lewis et al., 2020
   - Paper fondateur du RAG
   - [arXiv:2005.11401](https://arxiv.org/abs/2005.11401)

2. **A Comprehensive Survey of Retrieval-Augmented Generation (RAG): Evolution, Current Landscape and Future Directions**
   - Gupta et al., 2024
   - Survey complet et à jour
   - [arXiv:2410.12837](https://arxiv.org/abs/2410.12837)

3. **Dense Passage Retrieval for Open-Domain Question Answering**
   - Karpukhin et al., 2020
   - Base du retrieval dense
   - [arXiv:2004.04906](https://arxiv.org/abs/2004.04906)

4. **REALM: Retrieval-Augmented Language Model Pre-Training**
   - Guu et al., 2020
   - Pré-entraînement avec retrieval
   - [arXiv:2002.08909](https://arxiv.org/abs/2002.08909)

### 8.2 Ressources Techniques

1. **LangChain Documentation**
   - [https://python.langchain.com/docs/](https://python.langchain.com/docs/)
   - Framework RAG le plus utilisé

2. **LlamaIndex (GPT Index)**
   - [https://docs.llamaindex.ai/](https://docs.llamaindex.ai/)
   - Optimisé pour RAG

3. **DeepEval - RAG Evaluation**
   - [https://github.com/confident-ai/deepeval](https://github.com/confident-ai/deepeval)
   - Framework d'évaluation

4. **Chunking Strategies Guide (Databricks)**
   - [Guide Complet sur les Stratégies de Chunking](https://community.databricks.com/t5/technical-blog/the-ultimate-guide-to-chunking-strategies-for-rag-applications/ba-p/113089)
   - Best practices de chunking

### 8.3 Lectures Complémentaires

1. **Embedding Models:**
   - MTEB Leaderboard: [https://huggingface.co/spaces/mteb/leaderboard](https://huggingface.co/spaces/mteb/leaderboard)
   - Comparaison des modèles d'embedding

2. **Vector Databases:**
   - Pinecone Learning Center: [https://www.pinecone.io/learn/](https://www.pinecone.io/learn/)
   - Qdrant Documentation: [https://qdrant.tech/documentation/](https://qdrant.tech/documentation/)

3. **RAG Evaluation:**
   - RAGAS Framework: [https://github.com/explodinggradients/ragas](https://github.com/explodinggradients/ragas)
   - Confident AI Blog: [https://www.confident-ai.com/blog/rag-evaluation-metrics-answer-relevancy-faithfulness-and-more](https://www.confident-ai.com/blog/rag-evaluation-metrics-answer-relevancy-faithfulness-and-more)

### 8.4 Outils et Frameworks

| Outil | Usage | Lien |
|-------|-------|------|
| LangChain | Framework RAG complet | [GitHub](https://github.com/langchain-ai/langchain) |
| LlamaIndex | Indexation et retrieval | [GitHub](https://github.com/run-llama/llama_index) |
| Chroma | Vector DB embedded | [GitHub](https://github.com/chroma-core/chroma) |
| Qdrant | Vector DB production | [GitHub](https://github.com/qdrant/qdrant) |
| Pinecone | Vector DB cloud | [Website](https://www.pinecone.io/) |
| DeepEval | Évaluation LLM/RAG | [GitHub](https://github.com/confident-ai/deepeval) |

---

## Conclusion

Les systèmes RAG représentent une approche puissante pour augmenter les capacités des LLMs avec des connaissances externes, vérifiables et à jour. 

**Points clés à retenir :**

1. **Architecture en 4 phases** : Indexation, Retrieval, Augmentation, Generation
2. **Chunking critique** : Détermine la qualité du retrieval
3. **Embeddings** : Représentation vectorielle pour similarité sémantique
4. **Optimisations** : Hybrid search, reranking, query expansion
5. **Évaluation** : Mesurer séparément retriever et generator
6. **Production** : Monitoring, feedback, amélioration continue

**Prochaines étapes :**
- Implémenter un RAG simple sur vos propres données
- Expérimenter avec différentes stratégies de chunking
- Mesurer et optimiser avec les métriques d'évaluation
- Itérer basé sur le feedback utilisateur

Les RAG sont un domaine en évolution rapide. Restez à jour avec les nouvelles recherches et techniques !
