# Detection automatique de sentiment vocal

## Liens du projet

- Repository GitHub : https://github.com/marieangedieng/vocal_asr_sentiment
- Demo publique Render : https://vocal-asr-sentiment.onrender.com

## Presentation

Ce projet implemente un pipeline francais de bout en bout pour analyser un fichier audio client et retourner une transcription ainsi qu'un sentiment global.

```text
Audio .mp3/.wav
  -> validation et pretraitement audio
  -> transcription ASR avec Wav2Vec2 francais
  -> classification de sentiment avec DistilCamemBERT
  -> resultat JSON: transcription, sentiment, confiance, duree
```

Le projet expose trois manieres d'utiliser le pipeline :

- execution directe sur les audios de demonstration avec `run_samples.py`
- API FastAPI avec endpoint `/predict`
- interface Gradio qui appelle l'API

## Modeles utilises

Le module ASR utilise le modele Hugging Face :

```text
jonatasgrosman/wav2vec2-large-xlsr-53-french
```

Ce modele transcrit l'audio francais. Il attend un signal audio en `16000 Hz`; le pipeline convertit donc les fichiers entrants vers ce taux d'echantillonnage.

Le module sentiment utilise :

```text
cmarkea/distilcamembert-base-sentiment
```

Ce modele retourne une prediction en 5 niveaux de type etoiles. Le projet regroupe ces scores en trois classes metier :

| Scores du modele | Classe finale |
|---|---|
| 1 ou 2 etoiles | negatif |
| 3 etoiles | neutre |
| 4 ou 5 etoiles | positif |

La confiance retournee correspond a la probabilite agregee de la classe finale.

## Architecture du projet

```text
.
├── app/
│   ├── api/
│   │   ├── main.py              # Application FastAPI, routes /health et /predict
│   │   └── schemas.py           # Schemas Pydantic des reponses API
│   ├── pipeline/
│   │   ├── asr.py               # Chargement Wav2Vec2 et transcription audio -> texte
│   │   ├── config.py            # Configuration: modeles, duree max, seuil silence, device CPU/GPU
│   │   ├── errors.py            # Exceptions metier retournees proprement par l'API
│   │   ├── pipeline.py          # Orchestration complete audio -> transcription -> sentiment
│   │   ├── preprocessing.py     # Validation, decodage, mono, resampling 16 kHz, normalisation
│   │   └── sentiment.py         # Classification texte -> negatif/neutre/positif
│   └── gradio_app.py            # Interface Gradio cliente de l'API FastAPI
├── data/
│   └── samples/                 # Audios MP3 de demonstration positif/neutre/negatif
├── eval/
│   ├── evaluate_asr.py          # Evaluation ASR sur Common Voice FR, resultats dans results.md
│   ├── evaluate_sentiment.py    # Evaluation sentiment sur Allocine, resultats dans results.md
│   └── results.md               # Historique des resultats d'evaluation
├── Dockerfile                   # Image Docker avec ffmpeg et libsndfile
├── docker-compose.yml           # Lancement API + Gradio en deux services
├── environment.yml              # Environnement conda reproductible
├── requirements.txt             # Dependances Python pip
├── render_app.py                # Entree Render: FastAPI interne + Gradio public
├── run_samples.py               # Execution du pipeline sur data/samples
└── README.md
```

## Prerequis

- Python 3.11 recommande.
- Connexion internet au premier lancement pour telecharger les modeles Hugging Face.
- Espace disque disponible pour les caches Hugging Face.
- `ffmpeg` et `ffprobe` pour lire correctement les formats audio compresses comme MP3.
- GPU optionnel. Si CUDA est disponible, PyTorch l'utilise automatiquement; sinon l'inference se fait sur CPU.

Sur CPU, le chargement et la transcription avec Wav2Vec2 peuvent etre lents. Le premier lancement est aussi plus long car les modeles sont telecharges puis mis en cache.

## Installation depuis zero

### 1. Cloner le repository

```bash
git clone https://github.com/marieangedieng/vocal_asr_sentiment.git
cd vocal_asr_sentiment
```

### 2. Creer le fichier d'environnement

Le projet fournit un exemple :

```text
.env.example
```

Copier ce fichier vers `.env`.

Windows PowerShell :

```powershell
Copy-Item .env.example .env
```

Linux ou macOS :

```bash
cp .env.example .env
```

Variables principales :

```text
API_URL=http://localhost:8000
ASR_MODEL_ID=jonatasgrosman/wav2vec2-large-xlsr-53-french
SENTIMENT_MODEL_ID=cmarkea/distilcamembert-base-sentiment
MAX_DURATION_SEC=300
SILENCE_RMS_THRESHOLD=0.003
```

### 3. Installation avec Anaconda ou Miniconda

Si Anaconda ou Miniconda est installe, utiliser `environment.yml`.

```bash
conda env create -f environment.yml
conda activate sentiment-vocal
```

Pour mettre a jour un environnement deja cree :

```bash
conda env update -f environment.yml --prune
conda activate sentiment-vocal
```

Cette methode installe aussi `ffmpeg` et `libsndfile`.

Verification :

```bash
python --version
ffmpeg -version
ffprobe -version
```

### 4. Installation sans conda avec venv et pip

Dans ce cas, `requirements.txt` installe les dependances Python, mais `ffmpeg` doit etre installe separement au niveau systeme.

Windows PowerShell :

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Linux :

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

macOS :

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Installer `ffmpeg` sans conda :

Windows avec winget :

```powershell
winget install Gyan.FFmpeg
```

Windows avec Chocolatey :

```powershell
choco install ffmpeg
```

Linux Ubuntu/Debian :

```bash
sudo apt update
sudo apt install ffmpeg
```

macOS avec Homebrew :

```bash
brew install ffmpeg
```

Verification :

```bash
ffmpeg -version
ffprobe -version
```

## Reproduction et tests

### 1. Tester le pipeline sur les audios samples

Le fichier `run_samples.py` execute le pipeline complet sur les fichiers `.mp3` presents dans :

```text
data/samples
```

Commande :

```bash
python run_samples.py
```

Pour chaque fichier audio, le script affiche un resultat JSON :

```json
{
  "transcription": "texte transcrit",
  "sentiment": "positif",
  "confidence": 0.91,
  "duration_sec": 4.23
}
```

Ce test valide le fonctionnement local complet :

```text
fichier audio -> preprocessing -> ASR -> sentiment -> resultat
```

### 2. Evaluer les modules dans le dossier eval

Le dossier `eval/` contient deux evaluations independantes.

#### 2.1 Evaluation ASR

Script :

```text
eval/evaluate_asr.py
```

Objectif :

```text
Evaluer la qualite de transcription audio -> texte
```

Dataset :

```text
Common Voice FR
```

Le script utilise une lecture en streaming et une limite d'exemples pour eviter de telecharger un dataset trop lourd.

Commande recommandee :

```bash
python -m eval.evaluate_asr --limit 25
```

Le resultat est ajoute dans :

```text
eval/results.md
```

Metrique :

```text
WER - Word Error Rate
```

Remarque : le depot officiel `mozilla-foundation/common_voice_17_0` sur Hugging Face peut ne plus exposer directement les fichiers de donnees. Le script utilise donc un miroir compatible par defaut.

#### 2.2 Evaluation sentiment

Script :

```text
eval/evaluate_sentiment.py
```

Objectif :

```text
Evaluer la classification texte -> sentiment
```

Dataset :

```text
Allocine
```

Commande :

```bash
python -m eval.evaluate_sentiment
```

Le resultat est ajoute dans :

```text
eval/results.md
```

Metriques :

```text
accuracy
F1 macro
```

Important : Allocine est un dataset binaire positif/negatif. Il ne contient pas de classe neutre. L'evaluation sentiment ne mesure donc pas reellement la classe neutre.

Ces deux evaluations sont separees :

- Common Voice sert a evaluer l'ASR, car il contient des audios et des transcriptions.
- Allocine sert a evaluer le sentiment, car il contient des textes et des labels de sentiment.

Il ne s'agit pas d'une evaluation end-to-end audio -> sentiment, car le projet ne dispose pas d'un dataset public unique contenant audio, transcription et label de sentiment.

### 3. Tester l'API et l'application

#### 3.1 Tester l'API FastAPI

Lancer l'API depuis la racine du projet :

```bash
uvicorn app.api.main:app --reload --host 0.0.0.0 --port 8000
```

Ouvrir la documentation interactive :

```text
http://localhost:8000/docs
```

Tester d'abord :

```text
GET /health
```

La reponse attendue ressemble a :

```json
{
  "status": "ok",
  "asr_model": "jonatasgrosman/wav2vec2-large-xlsr-53-french",
  "sentiment_model": "cmarkea/distilcamembert-base-sentiment",
  "models_loaded": false,
  "device": "cpu"
}
```

Ensuite tester :

```text
POST /predict
```

Dans Swagger, cliquer sur `Try it out`, choisir un fichier audio depuis `data/samples`, puis executer la requete ou ajouter un audio depuis data/samples et liser le résultat.

Exemple avec `curl` :

```bash
curl -X POST "http://localhost:8000/predict" \
  -F "file=@data/samples/positif_1.mp3"
```

Windows PowerShell :

```powershell
curl.exe -X POST "http://localhost:8000/predict" `
  -F "file=@data/samples/positif_1.mp3"
```

#### 3.2 Tester l'interface Gradio

L'API doit rester lancee dans un premier terminal.

Donc Terminal 1 :

```bash
uvicorn app.api.main:app --reload --host 0.0.0.0 --port 8000
```

Dans un deuxieme terminal, lancer Gradio :

```bash
python app/gradio_app.py
```

Ouvrir :

```text
http://localhost:7860
```

Dans l'interface, selectionner un fichier audio depuis :

```text
data/samples
```

L'application affiche :

- la transcription produite par l'ASR
- le sentiment final
- la confiance
- la duree analysee

Si l'API n'est pas lancee sur `localhost:8000`, definir `API_URL`.

Windows PowerShell :

```powershell
$env:API_URL = "http://localhost:8000"
python app/gradio_app.py
```

Linux ou macOS :

```bash
API_URL=http://localhost:8000 python app/gradio_app.py
```

## Lancement avec Docker

Docker installe `ffmpeg` et `libsndfile1` via le `Dockerfile`.

Commande :

```bash
docker compose up --build
```

Services :

- API FastAPI : `http://localhost:8000`
- Swagger : `http://localhost:8000/docs`
- Gradio : `http://localhost:7860`

## Deploiement public

### Render

Le fichier :

```text
render_app.py
```

lance FastAPI en interne sur `127.0.0.1:8000`, puis lance Gradio sur le port public fourni par Render avec la variable `PORT`.

Parametres Render :

```text
Build command:
pip install -r requirements.txt

Start command:
python render_app.py
```

Il faut aussi s'assurer que `ffmpeg` est installe sur l'environnement de deploiement. Si la plateforme utilise Docker, le `Dockerfile` du projet l'installe deja.

URL Render :

```text
https://vocal-asr-sentiment.onrender.com
```

Remarque importante : sur le plan gratuit Render, le service peut se mettre en veille apres une periode d'inactivite. Au premier acces, il faut donc attendre environ 50 secondes avant que la page Gradio se charge. La page publique permet de verifier le deploiement de l'interface, mais l'analyse complete d'un fichier audio peut ne pas aboutir sur cette offre gratuite, car l'espace disque disponible ne suffit pas toujours a telecharger et mettre en cache les modeles Hugging Face utilises par le pipeline.

## Cas d'usage professionnels

Ce projet peut servir de base dans plusieurs contextes professionnels.

Centres d'appel et support client :

- detecter rapidement les appels clients negatifs
- prioriser les demandes urgentes ou les rappels
- mesurer la satisfaction apres une interaction avec le support

Banque, assurance et telecoms :

- analyser les reclamations clients
- suivre les motifs d'insatisfaction
- aider les equipes qualite a auditer les conversations

E-commerce et services numeriques :

- analyser les retours vocaux clients
- identifier les tendances positives ou negatives apres une campagne
- enrichir un tableau de bord satisfaction

Sante, education et administration :

- qualifier des demandes entrantes
- detecter des signaux d'insatisfaction dans des messages vocaux
- produire une premiere synthese exploitable par un agent humain

Dans un cadre reel, ce type de systeme doit rester une aide a la decision. Les predictions doivent etre interpretees avec prudence, surtout pour les cas sensibles.

## Difficultes rencontrees

Plusieurs difficultes techniques sont apparues pendant la mise en place.

Gestion de l'environnement Python :

- utiliser le mauvais interpreteur Python peut provoquer des erreurs comme `ModuleNotFoundError: No module named 'torch'`
- il faut lancer les scripts avec le Python de l'environnement actif

Formats audio :

- les fichiers MP3 demandent `ffmpeg` et `ffprobe`
- Gradio peut tenter de decoder l'audio avant l'appel API; l'interface utilise donc un upload de fichier pour eviter certains problemes de decodage

Datasets d'evaluation :

- le dataset Common Voice officiel sur Hugging Face peut ne plus exposer directement les fichiers
- l'evaluation ASR utilise une lecture limitee pour eviter des telechargements trop lourds
- l'evaluation sentiment utilise Allocine, qui ne fournit pas de classe neutre

Contraintes de deploiement :

- Hugging Face Spaces Gradio peut ne pas etre disponible gratuitement selon le compte
- les modeles sont lourds pour certaines plateformes gratuites
- le premier demarrage peut etre long car les modeles doivent etre telecharges

## Limites connues

- Le projet est specialise pour la langue francaise.
- La transcription peut etre degradee par le bruit, les accents forts, les voix superposees ou une mauvaise qualite audio.
- Le pipeline traite un fichier audio comme une seule sequence; il ne fait pas de diarisation locuteur.
- Le modele sentiment analyse la transcription textuelle, pas directement l'intonation ou la prosodie.
- Une erreur ASR peut influencer directement la prediction de sentiment.
- Le mapping 5 etoiles vers 3 classes est une simplification metier.
- La classe neutre est difficile a evaluer avec Allocine, car ce dataset est binaire.
- L'inference CPU peut etre lente.
- Le systeme ne doit pas etre utilise seul pour des decisions sensibles sans validation humaine.

## References

- GitHub : https://github.com/marieangedieng/vocal_asr_sentiment
- Modele ASR : https://huggingface.co/jonatasgrosman/wav2vec2-large-xlsr-53-french
- Modele sentiment : https://huggingface.co/cmarkea/distilcamembert-base-sentiment
- Common Voice : https://commonvoice.mozilla.org/fr/datasets
- Allocine dataset : https://huggingface.co/datasets/allocine
- FastAPI : https://fastapi.tiangolo.com/
- Gradio : https://www.gradio.app/
- Render : https://render.com/
