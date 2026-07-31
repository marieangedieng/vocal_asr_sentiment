# Resultats d'evaluation

Les scripts `eval/evaluate_asr.py` et `eval/evaluate_sentiment.py` ecrivent ici les resultats reels apres execution.

Dans cet environnement, les jeux de donnees Hugging Face et les poids des modeles ne sont pas telecharges automatiquement. Lancez les scripts apres installation des dependances et acces reseau:

```bash
python eval/evaluate_asr.py
python eval/evaluate_sentiment.py
```


## Evaluation ASR - Common Voice FR

- Echantillons: 25
- WER: 0.3177

## Evaluation sentiment - Allocine

- Echantillons: 40
- Approximation: Allocine est binaire, donc aucun exemple neutre n'est mesure.
- Accuracy: 0.8500
- F1 macro: 0.5957
