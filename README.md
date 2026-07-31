# Detection automatique de sentiment dans des appels vocaux

## 1. Presentation du projet

Ce projet d'examen implemente un pipeline francais de bout en bout pour analyser des appels clients:

```text
Audio .wav/.mp3
  -> pretraitement audio: mono, 16 kHz, normalisation, controle silence/duree
  -> ASR Wav2Vec 2.0: transcription
  -> DistilCamemBERT sentiment: probabilites 1 a 5 etoiles
  -> mapping metier: negatif / neutre / positif + confiance agregee
```

L'application expose une API FastAPI (`POST /predict`, `GET /health`) et une interface Gradio. Gradio ne duplique pas la logique ML: elle envoie le fichier audio a l'API via HTTP.

## 2. Modeles utilises

ASR: [`jonatasgrosman/wav2vec2-large-xlsr-53-french`](https://huggingface.co/jonatasgrosman/wav2vec2-large-xlsr-53-french). Ce modele Wav2Vec 2.0 XLSR large est fine-tune pour la reconnaissance vocale francaise et impose une entree audio a 16 kHz. Il est adapte au sujet car il evite tout entrainement from scratch, fournit une bonne base ASR francaise et reste exploitable sur une RTX d'entree/milieu de gamme avec une seule instance chargee. La fiche Hugging Face indique une licence Apache-2.0 et environ 0.3B parametres; le depot contient des poids lourds, donc prevoir plusieurs Go de cache disque et environ 2 a 4 Go de VRAM en inference selon backend et precision.

Sentiment: [`cmarkea/distilcamembert-base-sentiment`](https://huggingface.co/cmarkea/distilcamembert-base-sentiment). Ce modele est un DistilCamemBERT francais fine-tune sur Amazon Reviews et Allocine. Il est plus leger qu'un CamemBERT complet, ce qui reduit la latence et la memoire. La fiche Hugging Face indique une licence MIT, 68.1M parametres et des poids PyTorch/Safetensors autour de 272 Mo. Le modele renvoie 5 classes (`1 star` a `5 stars`); le code agrège explicitement les probabilites:

| Etoiles | Classe finale |
|---|---|
| 1-2 | `negatif` |
| 3 | `neutre` |
| 4-5 | `positif` |

La confiance retournee est la somme des probabilites de la classe finale, pas la probabilite d'une seule etoile.

## 3. Prerequis

- Python 3.9 minimum; tests effectues ici avec Python 3.12.3.
- `ffmpeg` recommande pour un decodage MP3 fiable, surtout dans Docker.
- Espace disque: prevoir 5 a 8 Go pour l'environnement Python, les caches Hugging Face et les poids.
- GPU optionnel: CUDA est utilise automatiquement si `torch.cuda.is_available()` vaut `True`; sinon inference CPU. Sur CPU, l'ASR peut etre lente.
- VRAM estimee: ASR 2 a 4 Go, sentiment moins de 1 Go. Le pipeline charge chaque modele une seule fois.

## 4. Installation pas a pas

```bash
git clone <URL_DU_REPO>
cd projet-sentiment-vocal
```

Option 1 - avec `venv`:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Sous Windows PowerShell:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Option 2 - avec `conda` si Anaconda ou Miniconda est deja installe:

```bash
conda create -n sentiment-vocal python=3.11 -y
conda activate sentiment-vocal
pip install -r requirements.txt
cp .env.example .env
```

Sous Windows PowerShell:

```powershell
conda create -n sentiment-vocal python=3.11 -y
conda activate sentiment-vocal
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Dans cet environnement Windows, l'installation a necessite:

```powershell
python -m pip install --user --trusted-host pypi.org --trusted-host files.pythonhosted.org -r requirements.txt pytest==8.3.3
```

## 5. Lancement de l'API

```bash
uvicorn app.api.main:app --host 0.0.0.0 --port 8000 --reload
```

Adresses:

- Swagger: `http://localhost:8000/docs`
- Healthcheck: `http://127.0.0.1:8000/health`

Healthcheck valide localement via `TestClient` le 24 juillet 2026:

```json
{
  "status": "ok",
  "asr_model": "jonatasgrosman/wav2vec2-large-xlsr-53-french",
  "sentiment_model": "cmarkea/distilcamembert-base-sentiment",
  "models_loaded": false,
  "device": "cpu"
}
```

`models_loaded=false` signifie que le healthcheck n'a pas encore declenche le telechargement/chargement des poids.

## 6. Exemple d'appel API

Avec `curl`:

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -F "file=@data/samples/positif.wav"
```

Avec Python:

```python
import requests

with open("data/samples/positif.wav", "rb") as audio:
    response = requests.post(
        "http://127.0.0.1:8000/predict",
        files={"file": ("positif.wav", audio, "audio/wav")},
        timeout=600,
    )

print(response.status_code)
print(response.json())
```

Structure JSON retournee par l'API apres chargement des modeles:

```json
{
  "transcription": "texte transcrit par Wav2Vec2",
  "sentiment": "positif | neutre | negatif",
  "confidence": 0.91,
  "duration_sec": 4.23
}
```

La forme exacte ci-dessus est celle garantie par le schema. Les valeurs metier reelles pour les trois audios de demo doivent etre generees avec:

```bash
python scripts/run_samples.py
```

Dans l'environnement actuel, les poids Hugging Face n'etaient pas presents localement et le telechargement complet ASR n'a pas ete lance car il pese plusieurs Go.

## 7. Lancement de l'interface Gradio

Demarrer l'API d'abord, puis:

```bash
API_URL=http://localhost:8000 python app/gradio_app.py
```

Sous Windows PowerShell:

```powershell
$env:API_URL = "http://localhost:8000"
python app/gradio_app.py
```

Adresse locale: `http://127.0.0.1:7860`. L'application ecoute sur `0.0.0.0:7860`, donc elle peut aussi etre atteinte via l'adresse IP locale de la machine si le pare-feu l'autorise.

## 8. Lancement via Docker

```bash
docker compose up --build
```

Services:

- API FastAPI: `http://127.0.0.1:8000`, Swagger sur `http://127.0.0.1:8000/docs`
- Gradio: `http://127.0.0.1:7860`

Le volume Docker `hf-cache` conserve les poids Hugging Face entre les redemarrages.

## 9. Demo publique

Le dossier `hf_space/` contient les fichiers de deploiement Hugging Face Spaces. Aucun deploiement public reel n'a pu etre effectue dans cet environnement, car il manque un compte/token Hugging Face et une confirmation du nom du Space.

Commandes a lancer avec un compte Hugging Face:

```bash
huggingface-cli login
huggingface-cli repo create sentiment-vocal-client --type space --space_sdk gradio
git clone https://huggingface.co/spaces/<USER>/sentiment-vocal-client
cp -r app hf_space/app.py hf_space/README.md hf_space/requirements.txt sentiment-vocal-client/
cd sentiment-vocal-client
git add .
git commit -m "Deploy voice sentiment demo"
git push
```

URL attendue apres creation: `https://huggingface.co/spaces/<USER>/sentiment-vocal-client`.

## 10. Resultats de l'evaluation quantitative

Scripts fournis:

- `eval/evaluate_asr.py`: WER sur `mozilla-foundation/common_voice_17_0`, config `fr`, split `test[:25]`.
- `eval/evaluate_sentiment.py`: accuracy et F1 macro sur `tblard/allocine`, split `test[:40]`.

Ces datasets servent uniquement a l'evaluation, pas a l'entrainement.

| Evaluation | Dataset | Commande | Resultat local |
|---|---|---|---|
| ASR WER | Common Voice 17.0 FR | `python eval/evaluate_asr.py` | Non execute: poids ASR non telecharges |
| Sentiment accuracy/F1 | Allocine | `python eval/evaluate_sentiment.py` | Non execute: poids sentiment non telecharges |

Les resultats reels sont ecrits dans `eval/results.md`.

## 11. Cas d'usage

- Prioriser les rappels clients en detectant rapidement les appels negatifs.
- Mesurer la satisfaction apres une campagne support ou commerciale.
- Construire un tableau de bord qualite a partir de transcriptions et scores de sentiment.

## 12. Limites connues

- Projet francais uniquement.
- Sensible au bruit de fond, aux accents tres eloignes du corpus d'entrainement et aux appels multi-locuteurs.
- La synthese vocale des samples n'est pas un vrai appel client; elle sert uniquement a la demo sans donnees confidentielles.
- Le mapping 5 etoiles vers 3 classes est utile metier mais reste une approximation.
- Latence elevee possible en CPU, surtout pour l'ASR Wav2Vec2 large.
- Le deploiement public doit etre finalise par le proprietaire du compte Hugging Face.

## 13. Structure du projet

```text
.
├── app/
│   ├── api/
│   │   ├── main.py
│   │   └── schemas.py
│   ├── pipeline/
│   │   ├── asr.py
│   │   ├── config.py
│   │   ├── errors.py
│   │   ├── pipeline.py
│   │   ├── preprocessing.py
│   │   └── sentiment.py
│   └── gradio_app.py
├── data/
│   └── samples/
│       ├── negatif.wav
│       ├── neutre.wav
│       └── positif.wav
├── eval/
│   ├── evaluate_asr.py
│   ├── evaluate_sentiment.py
│   └── results.md
├── hf_space/
│   ├── README.md
│   ├── app.py
│   └── requirements.txt
├── scripts/
│   ├── create_demo_audio.ps1
│   └── run_samples.py
├── tests/
│   ├── test_api.py
│   └── test_pipeline.py
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

## 14. Samples de demonstration

Les fichiers ci-dessous ont ete generes localement avec `scripts/create_demo_audio.ps1` par synthese vocale Windows.

| Fichier | Phrase source | Sentiment attendu | Resultat pipeline |
|---|---|---|---|
| `data/samples/positif.wav` | Je suis tres satisfait du service, la personne au telephone a ete rapide et claire. | positif | A generer avec `python scripts/run_samples.py` |
| `data/samples/negatif.wav` | Je suis vraiment mecontent, mon probleme dure depuis plusieurs jours et personne ne me rappelle. | negatif | A generer avec `python scripts/run_samples.py` |
| `data/samples/neutre.wav` | Je vous appelle pour connaitre les horaires et verifier le statut de mon dossier. | neutre | A generer avec `python scripts/run_samples.py` |

## Validation locale effectuee

```text
python -m pytest -q
7 passed in 4.55s
```

Le test couvre le pretraitement audio, les erreurs de format/silence, le mapping sentiment et les endpoints `/health` et `/predict` avec pipeline mocke.

## References et liens utiles

- Modele ASR: https://huggingface.co/jonatasgrosman/wav2vec2-large-xlsr-53-french
- Modele sentiment: https://huggingface.co/cmarkea/distilcamembert-base-sentiment
- Dataset Common Voice 17.0: https://huggingface.co/datasets/mozilla-foundation/common_voice_17_0
- Dataset Allocine: https://huggingface.co/datasets/tblard/allocine
- FastAPI: https://fastapi.tiangolo.com/
- Gradio: https://www.gradio.app/
- Hugging Face Spaces: https://huggingface.co/docs/hub/spaces
