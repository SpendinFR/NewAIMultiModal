# Pipeline d'entraînement détaillé

Ce document décrit les étapes pratiques pour entraîner le système NewAIMultiModal end-to-end.

## 1. Préparation des données

1. **Collecte réelle** : exporter des scènes (images + descriptions textuelles + états) dans `data/clevr_real`.
2. **Simulation** : lancer un simulateur CLEVR-like pour générer `num_scenes` configurations + actions/interventions. Stocker dans `data/sim_trajs`.
3. **Augmentations/interventions** :
   - `color_swap`: modifie aléatoirement la couleur d'un objet.
   - `object_replace`: remplace un objet par un autre (mêmes attributs sauf un).
   - `text_counterfactual`: utiliser un LLM pour réécrire la description sous intervention.
4. **Buffer de trajectoires** : insérer toutes les transitions `(state_t, action_t, intervention, state_{t+1})`.

## 2. Phase 0 — Pré-entraînement

Commande indicative :
```
python train.py --config configs/base.yaml --phase phase0_pretrain
```
Objectifs actifs : reconstruction (vision+texte), InfoNCE, sparsité slots, KL.

## 3. Phase 1 — Dynamics + interventions

1. Activer le simulateur pour générer des trajectoires additionnelles.
2. Entraîner le module de transition avec L_dyn + L_interv.
```
python train.py --config configs/base.yaml --phase phase1_dynamics --replay data/sim_trajs
```
3. Vérifier la métrique `intervention_consistency` toutes les 5k itérations.

## 4. Phase 2 — Invariances causales

- Échantillonner des paires `(s, s^{do(i)})`.
- Minimiser la distance des slots non descendants.
- Ajuster `lambda_interv` si l'invariance stagne.

## 5. Phase 3 — Grounding supervisé (optionnel)

- Charger des labels faibles (ex. attributs d'objets) et projeter sur certains slots.
- Utiliser un loss MSE/cross-entropy additionnel.

## 6. Phase 4 — Joint + LLM conditionné

1. Connecter le décodeur (LLM) et activer L_text.
2. Distiller un enseignant : fournir les slots + interventions au LLM professeur pour générer les cibles.
3. Ajouter des tests contrefactuels : modifie un slot `c_i` et vérifier que le texte change localement.

## 7. Évaluation continue

- `python evaluate.py --config configs/base.yaml --metrics intervention_consistency slot_interpretability`
- Pour la planification : `python plan.py --config configs/base.yaml --goal spec.json`.

## 8. Notes de monitoring

- `slots_dashboard.py` : visualiser `exist_k`, attention, attributs.
- `reasoner_trace.py` : inspecter les chaînes de raisonnement (opérateurs SELECT/RELATE).
- `decoder_alignment.py` : mesurer la dépendance du texte au slots via counterfactual KL.
