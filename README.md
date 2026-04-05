# YouTube Fitness Influencer Comment Analysis

This project is a full sentiment-analysis workflow for YouTube fitness comments. It trains multiple NLP models on a fitness comment dataset, selects the best model automatically, serves predictions through a FastAPI backend, and exposes the results through a Chrome extension popup that fetches real comments from the YouTube Data API using a video ID and API key.

## What This Project Does

- Trains and compares 9 text-classification models on a labeled fitness-comments dataset.
- Selects the best model using macro F1, then saves it for inference.
- Fetches YouTube comments from the YouTube Data API by video ID.
- Predicts positive, neutral, and negative sentiment for each fetched comment.
- Shows sentiment counts, percentages, top terms, and a word-cloud-style term view in a Chrome extension.
- Includes a Streamlit dashboard for local dataset inspection and model leaderboard review.

## Dataset

Sample labeled training data is included at:

`data/raw/youtube_fitness_comments.csv`

Expected columns:

- `influencer`
- `video_title`
- `comment_text`
- `like_count`
- `reply_count`
- `days_since_post`
- `sentiment`

## Model Lineup

The training pipeline compares these 9 models:

1. Logistic Regression
2. Linear SVC (Calibrated)
3. SGD Classifier (Calibrated)
4. Passive Aggressive (Calibrated)
5. Ridge Classifier (Calibrated)
6. Multinomial Naive Bayes
7. Complement Naive Bayes
8. Bernoulli Naive Bayes
9. Random Forest

The best model is saved to:

`outputs/models/best_fitness_sentiment_model.pkl`

## Project Structure

```text
.
|-- api.py
|-- app.py
|-- extension/
|   |-- manifest.json
|   |-- popup.css
|   |-- popup.html
|   `-- popup.js
|-- data/
|-- outputs/
|-- src/
|   |-- fitness_sentiment/
|   |   |-- modeling.py
|   |   |-- text_utils.py
|   |   `-- youtube_api.py
|   |-- data_prep.py
|   |-- eda.py
|   |-- model_comparison.py
|   `-- train_model.py
`-- requirements.txt
```

## Setup

Install dependencies:

```powershell
pip install -r requirements.txt
```

Prepare the dataset:

```powershell
python src/data_prep.py
```

Train and save the best model:

```powershell
python src/train_model.py
```

Optional: print the leaderboard without saving again:

```powershell
python src/model_comparison.py
```

Start the backend API:

```powershell
uvicorn api:app --reload
```

Optional: launch the local Streamlit dashboard:

```powershell
streamlit run app.py
```

## Deploy The API

The easiest production setup for this project is:

- deploy the FastAPI backend
- keep the Chrome extension as the frontend
- paste the deployed backend URL into the extension popup

This repository now includes:

- `Dockerfile`
- `.dockerignore`
- `requirements-api.txt`
- `render.yaml`

These files are meant for hosting the API on a Docker-friendly service such as Render.

### Deploy on Render

1. Push this repository to GitHub.
2. Open Render and create a new `Web Service`.
3. Connect your GitHub repository:
   - `yt-fitness-influencer-comment-analysis`
4. Render will detect the included `Dockerfile`.
5. Deploy the service.
6. After deployment, copy your public backend URL.

Example deployed URL:

```text
https://yt-fitness-influencer-comment-analysis-api.onrender.com
```

Test the deployed API:

- `/`
- `/health`
- `/docs`
- `/model/summary`

Example:

```text
https://your-service-name.onrender.com/health
```

### Use The Deployed API In The Chrome Extension

After your backend is live:

1. Open the Chrome extension popup.
2. Set `Backend URL` to your deployed API URL.
3. Paste your YouTube Data API key.
4. Open a YouTube video tab.
5. Run analysis.

The extension will then:

- fetch comments for the current video
- send them to the deployed model API
- show sentiment charts, top terms, and comment-level predictions

### Important Deployment Notes

- The deployed API uses the saved `SGD Classifier (Calibrated, Optuna Tuned)` model.
- The popup summary still works even if training metric files are not present on the host.
- DistilBERT checkpoints are intentionally excluded from deployment because they are too large for a lightweight first deploy.
- Your YouTube API key is not stored in the repository. Enter it only in the extension popup.

## Chrome Extension Workflow

1. Open `chrome://extensions/`
2. Enable `Developer mode`
3. Click `Load unpacked`
4. Select the `extension/` folder from this project
5. Start the FastAPI server locally
6. Open the extension popup
7. Enter:
   - your backend URL, usually `http://127.0.0.1:8000`
   - your YouTube Data API key
   - the target YouTube video ID
8. Click `Fetch and Analyze`

The popup will show:

- video metadata
- positive, neutral, and negative counts
- percentage bars
- top terms and word-cloud-style terms
- per-comment predictions with confidence

## API Endpoints

- `GET /health`
- `GET /model/summary`
- `POST /predict`
- `POST /youtube/analyze`

Example payload for `POST /youtube/analyze`:

```json
{
  "api_key": "YOUR_YOUTUBE_API_KEY",
  "video_id": "VIDEO_ID",
  "max_comments": 80
}
```

## Outputs

- `outputs/metrics/model_leaderboard.csv`
- `outputs/metrics/best_model_summary.json`
- `outputs/metrics/best_model_classification_report.txt`
- `outputs/models/best_fitness_sentiment_model.pkl`

## Notes

- The extension does not call YouTube directly; it talks to your local FastAPI backend.
- Your API key stays in the extension's local storage on your machine.
- You can replace the sample fitness dataset later with a larger labeled dataset to improve accuracy.
