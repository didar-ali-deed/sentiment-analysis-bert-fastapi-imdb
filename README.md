# Sentiment Analysis with BERT

A robust and scalable pipeline for fine-tuning BERT on the IMDb movie review dataset for binary sentiment classification (Positive/Negative). This project provides a web-based application with a FastAPI backend and a React frontend, enabling real-time sentiment predictions with interactive visualizations.

## Overview

This project implements a complete machine learning pipeline to fine-tune a BERT model (`bert-base-uncased`) for sentiment analysis on the IMDb dataset. It includes data preprocessing, model training, API deployment, and a user-friendly web interface with visualizations.

- **Dataset**: IMDb movie reviews (25,000 training, 25,000 test samples).
- **Model**: `bert-base-uncased`, fine-tuned for 3 epochs.
- **Performance**: Test Accuracy: 93.66%, F1-Score: 93.66%.
- **Technologies**: Hugging Face Transformers, PyTorch, Datasets, FastAPI, React, Chart.js.
- **Hardware**: Trained on NVIDIA RTX A3000 GPU.

## Features

- **End-to-End Pipeline**: Download, preprocess, train, and deploy a BERT model for sentiment analysis.
- **FastAPI Backend**: RESTful API for real-time sentiment predictions.
- **React Frontend**: Responsive UI with text input and probability visualizations.
- **Visualizations**: Bar charts displaying Positive/Negative sentiment probabilities.
- **Scalable Deployment**: Backend deployable on Render, frontend on Vercel.

## Project Structure

```
sentiment-analysis-with-bert/
├── backend/                    # FastAPI backend code
│   ├── models/                 # Fine-tuned BERT model files
│   ├── app.py                  # API implementation
│   ├── requirements.txt        # Backend dependencies
│   └── start.sh                # Render start script
├── data/                       # Raw and tokenized datasets, exploration plots
├── frontend/                   # React frontend code
├── models/                     # Original trained model and tokenizer
├── results/                    # Training metrics and predictions
├── scripts/                    # Pipeline scripts
│   ├── download_dataset.py     # Downloads IMDb dataset
│   ├── preprocess_dataset.py   # Preprocesses and tokenizes data
│   ├── training.py             # Trains the BERT model
│   └── deploy.py               # Deploys model for inference
├── .gitignore                  # Git ignore rules
├── README.md                   # Project documentation
└── requirements.txt            # Project-wide Python dependencies
```

## Prerequisites

- **Python**: 3.11
- **Node.js**: 18.x or later
- **Git**: For version control
- **Conda**: For environment management
- **Hardware**: GPU recommended for training (e.g., NVIDIA RTX A3000)

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/didar-ali-deed/sentiment-analysis-bert-fastapi-imdb.git
cd sentiment-analysis-bert-fastapi-imdb
```

### 2. Set Up the Backend

1. Create and activate a Conda environment:
   ```bash
   conda create -n myenv python=3.11
   conda activate myenv
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the pipeline scripts:
   ```bash
   python scripts/download_dataset.py
   python scripts/preprocess_dataset.py
   python scripts/training.py
   python scripts/deploy.py
   ```

### 3. Set Up the Frontend

1. Install Node.js: Download from https://nodejs.org/ (LTS version recommended).
2. Navigate to the frontend directory and install dependencies:
   ```bash
   cd frontend
   npm install
   ```
3. Start the frontend:
   ```bash
   npm start
   ```
4. Access the app at `http://localhost:3000`.

### 4. Run the Backend Locally

1. Navigate to the backend directory:
   ```bash
   cd backend
   ```
2. Start the FastAPI server:
   ```bash
   uvicorn app:app --reload --host 0.0.0.0 --port 8000
   ```
3. Test the API:
   ```bash
   curl -X POST "http://127.0.0.1:8000/predict/" -H "Content-Type: application/json" -d '{"text": "This movie was absolutely fantastic!"}'
   ```

## API Usage

The FastAPI backend provides a RESTful endpoint for sentiment predictions.

- **Endpoint**: `POST /predict/`
- **Request Body**:
  ```json
  { "text": "Your movie review here" }
  ```
- **Response**:
  ```json
  {
    "text": "Input text",
    "predicted_label": "Positive/Negative",
    "score": 0.9996
  }
  ```
- **Example**:
  ```bash
  curl -X POST "http://127.0.0.1:8000/predict/" -H "Content-Type: application/json" -d '{"text": "This movie was absolutely fantastic!"}'
  ```
- **Health Check**: `GET /health` (returns `{"status": "healthy"}`)

## Deployment

### Backend Deployment (Render)

- **Setup**:
  1. Copy `models/bert-imdb-final` (excluding `*.safetensors`) to `backend/models/`.
  2. Create `backend/requirements.txt` and `backend/start.sh`.
  3. Update `.gitignore` to exclude `*.safetensors`.
  4. Push code to GitHub: `https://github.com/didar-ali-deed/imdb-sentiment-analysis-with-bert`.
  5. Deploy via Render with `backend/` as root directory. Upload `model.safetensors` separately during deployment.

### Frontend Deployment (Vercel)

- Instructions to be added after backend deployment.

## Results

- **Model Location**: `models/bert-imdb-final/`
- **Example Predictions**:
  | Text | Predicted Sentiment | Score |
  |------|--------------------|-------|
  | "This movie was absolutely fantastic!" | Positive | 0.9996 |
  | "The plot was boring and the acting was terrible." | Negative | 0.9997 |

## Visualizations

The React frontend includes a bar chart (powered by Chart.js) to visualize sentiment probabilities.

- **Setup**: Install dependencies in `frontend/`:
  ```bash
  npm install chart.js react-chartjs-2
  ```
- **Feature**: Displays Positive/Negative probability distribution after each prediction.

## Contributing

Contributions are welcome! Please:

1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/YourFeature`).
3. Commit changes (`git commit -m "Add YourFeature"`).
4. Push to the branch (`git push origin feature/YourFeature`).
5. Open a pull request.

## Author

**Didar Ali**  
Computer Systems Engineering Student, UET Peshawar

- LinkedIn: [Didar Ali](https://www.linkedin.com/in/didar-ali-deed/)
- GitHub: [didar-ali-deed](https://github.com/didar-ali-deed)

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Hugging Face for the Transformers library
- IMDb dataset for providing a robust benchmark
- Render and Vercel for deployment platforms
