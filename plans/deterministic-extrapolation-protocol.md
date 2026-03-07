# Test d'Extrapolation Déterministe - "Routing Puzzle"

## 🎯 Objectif du Test

Valider la capacité du système TT-Distill à apprendre une règle mathématique **stricte, contre-intuitive, absente du pré-entraînement** via la distillation du Système 2 (MACA) vers le Système 1 (Réflexe), puis extrapoler cette règle sur des données inédites.

---

## 🧩 Règle Secrète (Synthétique)

**Formule**: `Cœur = (Thread_ID × 7) % 5`

Cette règle est:
- **Arbitraire**: Le multiplicateur 7 et le modulo 5 n'ont aucune signification statistique
- **Contre-intuitive**: Les modèles de langage tendent à faire des moyennes ou des approximations
- **Absente du pré-entraînement**: Cette combinaison spécifique n'existe nulle part sur internet

### Exemples d'Entraînement (S2 voit)

| Thread_ID | Calcul | Cœur Attendu |
|-----------|--------|--------------|
| 2 | 2×7=14, 14%5=**4** | 4 |
| 3 | 3×7=21, 21%5=**1** | 1 |
| 4 | 4×7=28, 28%5=**3** | 3 |

### Threads de Test (Inédits - S1 doit extrapoler)

| Thread_ID | Calcul | Cœur Attendu |
|-----------|--------|--------------|
| 8 | 8×7=56, 56%5=**1** | 1 |
| 9 | 9×7=63, 63%5=**3** | 3 |
| 10 | 10×7=70, 70%5=**0** | 0 |

**Réponse attendue**: `[1, 3, 0]`

---

## 🏗️ Architecture du Test

```mermaid
flowchart TB
    subgraph Phase1["Phase 1: Baseline (Sans adaptateur)"]
        B1[Modèle Qwen2.5-1.5B] --> B2[Prompt: 'Assigne les cœurs NPU pour les threads [8, 9, 10]']
        B2 --> B3[Hallucination attendue<br/>ex: [1, 2, 3] ou moyenne]
    end

    subgraph Phase2["Phase 2: Distillation (MACA → DoRA)"]
        S2[Agents S2 deliberation] --> S2_1[Intentions: Thread 2→4, 3→1, 4→3]
        S2_1 --> MACA[Moteur MACA<br/>Consensus tensoriel]
        MACA --> Sinkhorn[Algorithme Sinkhorn<br/>Barycentre de Wasserstein]
        Sinkhorn --> Bridge[TT-Distill Bridge<br/>SVD Factorization]
        Bridge --> Adapter[Adaptateur DoRA<br/>test_adapter.bin ~15MB]
    end

    subgraph Phase3["Phase 3: Extrapolation (Avec adaptateur)"]
        Adapter --> S1[Modèle + Adaptateur]
        S1 --> Prompt[Prompt: 'Assigne les cœurs NPU pour les threads [8, 9, 10]']
        Prompt --> Response[Extrapolation attendue: [1, 3, 0]]
    end

    Phase1 --> Phase2 --> Phase3
```

---

## 📋 Étapes d'Exécution

### Étape 1: Vérification des Prérequis

```bash
# Vérifier l'existence des fichiers
ls -lh qwen2.5-1.5b-instruct-q8_0.gguf
ls -lh test_adapter.bin  # Peut ne pas exister encore
```

**Fichiers requis**:
- [`qwen2.5-1.5b-instruct-q8_0.gguf`](qwen2.5-1.5b-instruct-q8_0.gguf): Modèle de base (~1.5B paramètres)
- [`test_adapter.bin`](test_adapter.bin): Adaptateur DoRA (généré par le test)

### Étape 2: Exécution du Test

```bash
cd /Users/morad/Projects/project-manager
python tests/test_deterministic_extrapolation.py
```

### Étape 3: Analyse des Résultats

Le test produit trois phases de résultats:

#### Phase 1 - Baseline (Sans adaptateur)
- **Comportement attendu**: Hallucination ou approximation statistique
- **Précision attendue**: < 30%
- **Latence**: ~10-15 ms par token

#### Phase 2 - Distillation (MACA → DoRA)
- **Score de consensus**: > 0.9 (seuil de validation)
- **Taille adaptateur**: ~15 MB (cible)
- **Format**: Pickle avec tenseurs `lora_a` et `lora_b`

#### Phase 3 - Extrapolation (Avec adaptateur)
- **Précision attendue**: ≥ 60% (succès), ≥ 30% (partiel)
- **Réponse attendue**: `[1, 3, 0]`
- **Fréquence réflexe**: > 60 Hz (latence < 13 ms)

---

## 🎯 Critères de Succès

| Métrique | Cible | Statut |
|----------|-------|--------|
| Précision extrapolation | ≥ 60% | ⚠️ À valider |
| Latence moyenne | < 13 ms | ⚠️ À valider |
| Fréquence réflexe | > 60 Hz | ⚠️ À valider |
| Score consensus MACA | > 0.9 | ⚠️ À valider |
| Taille adaptateur | ~15 MB | ⚠️ À valider |

---

## 🔧 Configuration Technique

### Paramètres du Modèle
- **Modèle**: Qwen2.5-1.5B-Instruct (GGUF Q8_0)
- **Hidden size**: 2048
- **Context length**: 2048 tokens
- **GPU layers**: 45 (accélération GPU)

### Paramètres MACA
- **Latent dim**: 2048 (aligné avec hidden_size)
- **Seq len**: 32 (longueur du rollout latent)
- **Sinkhorn epsilon**: 1e-3
- **Sinkhorn iterations**: 50

### Paramètres DoRA
- **Adapter size**: 15 MB
- **Rank**: 16
- **SVD**: Activé pour factorisation optimale

---

## 🧪 Scénarios de Test

### Scénario A: Test Complet (Recommandé)
1. Exécuter le test depuis zéro (génération de l'adaptateur)
2. Valider la précision d'extrapolation
3. Mesurer la latence et la fréquence réflexe

### Scénario B: Test avec Adaptateur Existant
```bash
# Si test_adapter.bin existe déjà
python tests/test_deterministic_extrapolation.py
```

### Scénario C: Debug Mode
```bash
# Ajouter des logs détaillés
python -v tests/test_deterministic_extrapolation.py
```

---

## 📊 Interprétation des Résultats

### Succès (✅)
- Précision ≥ 60% sur les threads inédits [8, 9, 10]
- Réponse proche de `[1, 3, 0]`
- Latence < 13 ms (fréquence > 60 Hz)
- Le modèle a **effectivement appris** la règle par distillation

### Succès Partiel (⚠️)
- Précision entre 30% et 60%
- Le modèle montre une **compréhension partielle**
- Peut indiquer une distillation incomplète ou un bruit dans le consensus

### Échec (❌)
- Précision < 30%
- Réponse aléatoire ou hallucinée
- **Hypothèses**:
  - Le consensus MACA n'a pas capturé la règle
  - La distillation SVD a perdu l'information
  - Le modèle n'extrapole pas correctement

---

## 🔍 Debugging Guide

### Problème: Adaptateur trop petit (< 10 MB)
**Cause**: Rank trop faible ou hidden_size incorrect
**Solution**: Augmenter `rank` dans [`DeterministicExtrapolationTest.__init__()`](tests/test_deterministic_extrapolation.py:76)

### Problème: Précision baseline > 0%
**Cause**: Le modèle a appris une approximation statistique
**Solution**: Vérifier que la règle est vraiment absente du pré-entraînement

### Problème: Consensus score < 0.9
**Cause**: Les agents S2 divergent trop
**Solution**: Augmenter le nombre d'agents ou réduire le bruit dans les initial states

### Problème: Latence > 20 ms
**Cause**: Modèle non accéléré GPU ou adaptateur trop lourd
**Solution**: Vérifier `n_gpu_layers=45` et réduire `rank` si nécessaire

---

## 📝 Notes d'Implémentation

### Code Clé
- [`test_deterministic_extrapolation.py`](tests/test_deterministic_extrapolation.py:59): Classe principale du test
- [`src/orchestration/maca.py`](src/orchestration/maca.py:407): [`TTDistillBridge`](src/orchestration/maca.py:407) pour la distillation
- [`src/orchestration/maca.py`](src/orchestration/maca.py:90): [`SinkhornBarycenter`](src/orchestration/maca.py:90) pour le consensus

### Points d'Attention
1. **Reproductibilité**: Utiliser `np.random.seed(42)` dans la distillation
2. **Format de réponse**: Le prompt doit exiger une réponse purement numérique
3. **Parsing**: [`_parse_cores()`](tests/test_deterministic_extrapolation.py:389) extrait les nombres de la réponse

---

## 🚀 Prochaines Étapes

1. **Exécuter le test** avec `python tests/test_deterministic_extrapolation.py`
2. **Analyser les résultats** de chaque phase
3. **Ajuster les paramètres** si nécessaire (rank, latent_dim, etc.)
4. **Valider la réussite** selon les critères ci-dessus

---

## 📚 Références

- **TT-Distill**: Two-Level Cognitive Systems (Soatto & Achille, Feb 2026)
- **MACA**: Multi-Agent Consensus Alignment via Sinkhorn Barycenter
- **DoRA**: Decomposed Low-Rank Adaptation
- **SVD**: Singular Value Decomposition pour l'optimisation de rank