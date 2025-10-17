import React, { useState } from 'react';
import axios from 'axios';
import { Bar } from 'react-chartjs-2';
import { Chart as ChartJS, CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend } from 'chart.js';
import './App.css';

// Register Chart.js components
ChartJS.register(CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend);

function App() {
  const [text, setText] = useState('');
  const [prediction, setPrediction] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError(null);
    setPrediction(null);
    setLoading(true);

    try {
      const response = await axios.post('http://127.0.0.1:8000/predict/', { text });
      setPrediction(response.data);
    } catch (err) {
      setError(err.response?.data?.detail || 'An error occurred');
    } finally {
      setLoading(false);
    }
  };

  // Chart data
  const chartData = prediction
    ? {
        labels: ['Positive', 'Negative'],
        datasets: [
          {
            label: 'Sentiment Probability',
            data: [
              prediction.predicted_label === 'Positive' ? prediction.score : 1 - prediction.score,
              prediction.predicted_label === 'Negative' ? prediction.score : 1 - prediction.score,
            ],
            backgroundColor: ['#4CAF50', '#FF5733'],
            borderColor: ['#388E3C', '#C0392B'],
            borderWidth: 1,
          },
        ],
      }
    : null;

  return (
    <div className="App">
      <h1>Sentiment Analysis with BERT</h1>
      <form onSubmit={handleSubmit}>
        <textarea
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder="Enter your movie review..."
          rows="5"
          cols="50"
        />
        <br />
        <button type="submit" disabled={loading}>
          {loading ? 'Predicting...' : 'Predict Sentiment'}
        </button>
      </form>
      {error && <p style={{ color: 'red' }}>{error}</p>}
      {prediction && (
        <div>
          <h3>Prediction Result:</h3>
          <p><strong>Text:</strong> {prediction.text}</p>
          <p><strong>Sentiment:</strong> {prediction.predicted_label}</p>
          <p><strong>Confidence Score:</strong> {(prediction.score * 100).toFixed(2)}%</p>
          {chartData && (
            <div style={{ maxWidth: '400px', margin: '20px auto' }}>
              <Bar
                data={chartData}
                options={{
                  responsive: true,
                  plugins: {
                    legend: { position: 'top' },
                    title: { display: true, text: 'Sentiment Probability Distribution' },
                  },
                  scales: {
                    y: { beginAtZero: true, max: 1, title: { display: true, text: 'Probability' } },
                  },
                }}
              />
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export default App;