from __future__ import annotations

from pathlib import Path

from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score

from app.pipeline.sentiment import FrenchSentimentClassifier


RESULTS_PATH = Path("eval/results.md")


def map_allocine_label(label: int) -> str:
    # Allocine est binaire: 0 negatif, 1 positif. La classe neutre n'existe pas dans ce dataset.
    return "negatif" if label == 0 else "positif"


def main(limit: int = 40) -> None:
    dataset = load_dataset("allocine", split=f"test[:{limit}]")
    classifier = FrenchSentimentClassifier()

    y_true: list[str] = []
    y_pred: list[str] = []
    for sample in dataset:
        y_true.append(map_allocine_label(int(sample["label"])))
        y_pred.append(classifier.predict(sample["review"]).label)

    accuracy = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, labels=["negatif", "neutre", "positif"], average="macro", zero_division=0)

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS_PATH.open("a", encoding="utf-8") as file:
        file.write("\n## Evaluation sentiment - Allocine\n\n")
        file.write(f"- Echantillons: {len(y_true)}\n")
        file.write("- Approximation: Allocine est binaire, donc aucun exemple neutre n'est mesure.\n")
        file.write(f"- Accuracy: {accuracy:.4f}\n")
        file.write(f"- F1 macro: {f1_macro:.4f}\n")

    print(f"accuracy={accuracy:.4f} f1_macro={f1_macro:.4f} sur {len(y_true)} echantillons")


if __name__ == "__main__":
    main()

