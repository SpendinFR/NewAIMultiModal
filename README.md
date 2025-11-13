# NewAIMultiModal

Ce dépôt décrit une architecture complète et entraînable de bout en bout pour apprendre automatiquement des concepts multimodaux, les rendre manipulables via des interventions causales, puis les utiliser pour raisonner et générer du texte conditionné par un LLM. Le projet est structuré comme un plan d'implémentation détaillé : composants, algorithmes d'entraînement, pipeline de données, fonctions de coût, hyperparamètres et métriques.

## 1. Vision d'ensemble

Le système se compose des blocs suivants :

1. **Perception / Encoder multimodal** — encodeur multitour (texte, vision, état proprioceptif) transformant les entrées en un ensemble de features \(F = \{f_j\}\).
2. **Concept Extractor (Slot Attention + Bottleneck)** — découvre automatiquement \(K\) concepts sous forme de slots structurés (type, vecteur continu, score d'existence, code discret optionnel).
3. **World Graph / GNN** — construit un graphe des relations entre concepts et produit un état structuré \(S\).
4. **Dynamics / Causal Module** — apprend les transitions \(T(c, a) \rightarrow c'\) et supporte les interventions \(do(c_i = v)\).
5. **Reasoner neuro-symbolique** — applique des opérateurs différentiables (SELECT, RELATE, APPLY_RULE) pour réaliser des chaînes de raisonnement internes.
6. **Surface Realizer (Decoder / LLM conditionné)** — génère du texte fluent conditionné sur \(S\) et l'historique.
7. **Controller / Intervention Trainer** — orchestre les interventions synthétiques/réelles et impose des contraintes d'identifiabilité causale.

L'ensemble est entraîné de manière conjointe, avec possibilité de phases spécifiques pour stabiliser l'apprentissage.

## 2. Format des concepts

Chaque concept \(c_i\) est stocké comme un tuple structuré :

- `slot_i.type` — identifiant (appris) du type d'entité/attribut/relation.
- `slot_i.vec` — vecteur continu \(\in \mathbb{R}^d\).
- `slot_i.exist` — score d'existence \(s_i \in [0, 1]\) activé par une sigmoïde.
- `slot_i.sym` — code discret optionnel obtenu via Gumbel-Softmax (permet des concepts symboliques).

Les slots sont permutation-invariants et peuvent rester inutilisés grâce au score d'existence.

## 3. Modules détaillés

### 3.1 Perception encoder

- **Texte** : transformer (6–12 couches) sur un vocabulaire BPE partagé. Sorties = embeddings de tokens.
- **Vision** : CNN ou ViT produisant des embeddings de patches \(p \times d\).
- **État / proprioception** : MLP sur vecteurs d'état.
- **Fusion** : concaténation sous forme d'ensemble \(F\), + positional encodings spécifiques au modality.

### 3.2 Concept Extractor (Slot Attention + Bottleneck)

- Slot Attention (Locatello et al.) avec \(K \in [8, 64]\) slots.
- Chaque slot produit `vec_k` et `exist_k`.
- Un petit MLP prédit les paramètres \((\mu, \sigma)\) d'un latent de type VAE ou des logits Gumbel pour un code discret.
- Les reconstructions (L_rec) utilisent un décodeur miroir pour reconstituer les tokens masqués, patches visuels, ou états.

### 3.3 Disentanglement & Bottleneck

- Perte KL (\(\beta\)-VAE) pour encourager la factorisation.
- InfoNCE multi-vues pour stabiliser les slots à travers augmentations / modalités.
- Pénalité de sparsité sur `exist_k` (L_slot).

### 3.4 Graph Builder & GNN

- Construction d'un graphe complet ou sparse via attention (score \(a_{ij} = \text{softmax}(MLP([c_i, c_j]))\)).
- GNN (3 couches) de type Graph Attention Network ou Gated Graph Neural Network.
- Production de l'état structuré \(S = (C, E)\) utilisable par les modules suivants.

### 3.5 Dynamics / Causal Module

- Module de transition \(T: (C, a) \rightarrow C'\) basé sur message passing conditionné par l'action.
- Têtes d'intervention pour prédire les effets \(do(c_i = v)\) sur les autres slots.
- Entraînement via L_dyn (prédiction de l'état suivant) et L_interv (erreur sous intervention réelle/synthétique).

### 3.6 Reasoner (Neuro-symbolic)

- Implémenté comme un Transformer sur des « tokens » de slots + opérateurs.
- Opérations SELECT/RELATE/APPLY_RULE paramétrées, permettant de composer des chaînes de pensée internes et des tests contrefactuels.
- Les états intermédiaires réutilisent \(S\) et alimentent le LLM conditionné.

### 3.7 Surface Realizer (Decoder / LLM)

- Décodeur Transformer (type T5/LLama) recevant \(S\) via cross-attention (encodé comme séquence).
- Conditionnement additionnel par l'historique textuel.
- Distillation depuis un LLM enseignant possible pour enrichir les cibles.

### 3.8 Controller / Intervention Trainer

- Sélectionne des slots à intervenir (aléatoire, heuristique sur l'incertitude).
- Génère des interventions : modifications directes dans un simulateur, augmentations réalistes, ou contrefactuels générés.
- Supervise L_interv + invariances (slots non descendants invariants).

## 4. Fonction de coût globale

La perte totale est :
\[
\mathcal{L} = \lambda_{rec} \mathcal{L}_{rec} + \lambda_{slot} \mathcal{L}_{slot} + \lambda_{contr} \mathcal{L}_{contr} + \lambda_{dyn} \mathcal{L}_{dyn} + \lambda_{interv} \mathcal{L}_{interv} + \lambda_{text} \mathcal{L}_{text} + \lambda_{KL} \mathcal{L}_{KL}
\]

- **\(\mathcal{L}_{rec}\)** : reconstruction multimodale (tokens masqués, patches, états).
- **\(\mathcal{L}_{slot}\)** : sparsité/existence (penalise \(s_i\) élevés inutiles).
- **\(\mathcal{L}_{contr}\)** : InfoNCE multi-vues (positifs = même slot sous augmentation, négatifs = autres slots).
- **\(\mathcal{L}_{dyn}\)** : MSE ou CPC sur \(T(c,a)\).
- **\(\mathcal{L}_{interv}\)** : supervision des effets d'interventions réelles/synthétiques.
- **\(\mathcal{L}_{text}\)** : cross-entropie pour la génération conditionnée.
- **\(\mathcal{L}_{KL}\)** : terme \(\beta\)-VAE.
- Optionnel : **\(\mathcal{L}_{symbolic}\)** pour contraindre certains slots avec règles logiques.

## 5. Algorithme d'auto-apprentissage des concepts

1. **Phase 0 — Pré-entraînement multimodal** : self-supervised (masked modeling, reconstruction, contrastif) pour stabiliser les slots.
2. **Phase 1 — Dynamics + interventions synthétiques** : apprentissage de \(T\) avec trajectoires simulées et interventions contrôlées.
3. **Phase 2 — Disentanglement par invariances** : minimiser \(\|c_j(s) - c_j(s^{do(i)})\|\) pour les slots non descendants.
4. **Phase 3 — Grounding supervisé (optionnel)** : aligner certains slots avec labels humains via MSE / cross-entropy.
5. **Phase 4 — Entraînement joint avec le LLM** : décodeur conditionné sur les slots, distillation depuis un LLM professeur, vérification contrefactuelle (modifie \(c_i\) → texte attendu).

## 6. Pipeline de données

- **Sources réelles** : scènes visuelles annotées, logs robotiques, dialogues.
- **Sources synthétiques** : simulateurs 2D/3D (CLEVR, ProcTHOR), générateurs textuels.
- **Augmentations/interventions** : modifications d'attributs (couleur, position), remplacement d'entités, swaps d'objets, contrefactuels textuels.
- **Batchs mixtes** : assembler dans chaque batch un mélange réel/synthétique pour éviter l'oubli.
- **Gestion des trajectoires** : buffer de replay pour transitions et interventions.

## 7. Hyperparamètres indicatifs

| Hyperparamètre | Valeur de départ |
| --- | --- |
| `slot_dim` | 256 |
| `K` | 16 |
| `beta_vae` | 0.5 (ajuster 0.1–2) |
| `temperature_gumbel` | 0.5 → annealing vers 0.1 |
| `lr_pretrain` | 1e-4 (AdamW) |
| `lr_finetune` | 3e-5 |
| `batch_size` | 128 (réduire si séquences longues) |
| `infoNCE_tau` | 0.07 |
| `gnn_layers` | 3 |
| `decoder_layers` | 12 |
| `dropout` | 0.1 |

## 8. Métriques d'évaluation

- **Intervention Consistency (IC)** : proportion d'interventions dont l'effet prédit sur les slots (et le texte généré) est correct.
- **Slot Interpretability Score** : accuracy de probes linéaires sur des labels connus.
- **Grounding Accuracy** : IoU/détection pour la localisation visuelle des slots.
- **Causal Precision/Recall** : comparaison du graphe appris vs. graphe de vérité (en simulation).
- **Compositional Generalization** : réussite sur combinaisons attributaires tenues hors entraînement.
- **Planning Success Rate** : taux de succès d'un planificateur utilisant \(S\) pour atteindre des objectifs.
- **Qualité linguistique** : BLEU/ROUGE + évaluations humaines.

## 9. Pièges & bonnes pratiques

- Les interventions sont essentielles pour l'identifiabilité. Sans elles, les slots peuvent tourner arbitrairement.
- Attention à la fidélité du simulateur : des interventions trop artificielles créent des shortcuts.
- Le décodeur peut ignorer les slots : surveiller l'écart de performance entre « LLM seul » vs « LLM conditionné ». Augmenter \(\lambda_{interv}\) ou ajouter des pénalités si nécessaire.
- Commencer dans des environnements simples (gridworld, CLEVR) pour valider les objectifs avant de scaler.
- Suivre en continu la métrique IC, les probes et la qualité texte pour détecter les effondrements.

## 10. Exemple d'expérience complète (CLEVR-like)

1. Générer des scènes 3D (objets, couleurs, positions) + descriptions (enseignant LLM).
2. Pré-entraîner l'encodeur + slots sur reconstruction et contrastif.
3. Collecter des trajectoires d'actions (déplacer, peindre, empiler), avec interventions explicites.
4. Apprendre la dynamique \(T\) et les contraintes d'invariance.
5. Distiller un LLM enseignant pour générer : descriptions, réponses causales (« Si je peins... »).
6. Évaluer IC, Slot Interpretability, Planning Success.

Le dossier `configs/` fournit un exemple de configuration pour lancer ce pipeline.

## 11. Implémentation de référence

Le répertoire `src/newaimultimodal/` contient une implémentation légère mais fonctionnelle des concepts ci-dessus :

- `models/perception.py` encode les modalités (texte, vision symbolique, état) via des projections déterministes.
- `models/slots.py` applique un Slot Attention simplifié et expose les vecteurs `slots` + les scores `existence`.
- `models/graph.py`, `models/dynamics.py`, `models/reasoner.py` et `models/decoder.py` matérialisent respectivement le graphe causal, les transitions, le raisonneur neuro-symbolique et le décodeur textuel.
- `data/sources.py` et `data/buffer.py` fournissent un générateur CLEVR-like ainsi qu'un buffer de trajectoires pour les phases dynamiques.
- `training/trainer.py` orchestre les phases décrites ci-dessus et calcule des métriques légères (`intervention_consistency`, `slot_interpretability`, `grounding_accuracy`).
- Pour rester exécutable hors-ligne, `configs/base.yaml` est écrit en JSON (valide YAML) et le chargeur accepte automatiquement YAML ou JSON.

Pour lancer une expérience rapide (quelques itérations de démonstration) :

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
python train.py --config configs/base.yaml --phase phase0_pretrain --max_steps 3
```

Les journaux affichent les pertes et métriques toutes les `evaluation_interval` itérations. Les sorties et artefacts sont stockés dans `outputs/<nom_experience>/`.
