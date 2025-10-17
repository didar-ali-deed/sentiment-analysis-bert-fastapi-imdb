import React, { useState } from 'react';
import axios from 'axios';
import { Bar } from 'react-chartjs-2';
import { Chart as ChartJS, CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend } from 'chart.js';
import { TextField, Button, CircularProgress, Typography, Box, Paper, Switch } from '@mui/material';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import { motion } from 'framer-motion';
import { ToastContainer, toast } from 'react-toastify';
import 'react-toastify/dist/ReactToastify.css';
import './App.css';

// Register Chart.js components
ChartJS.register(CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend);

function App() {
  const [text, setText] = useState('');
  const [prediction, setPrediction] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);
  const [darkMode, setDarkMode] = useState(false);

  const theme = createTheme({
    palette: {
      mode: darkMode ? 'dark' : 'light',
      primary: { main: '#3b82f6' },
      secondary: { main: '#ef4444' },
    },
  });

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!text.trim()) {
      toast.error('Please enter a movie review.');
      return;
    }
    setError(null);
    setPrediction(null);
    setLoading(true);

    try {
      const response = await axios.post('http://127.0.0.1:8000/predict/', { text });
      setPrediction(response.data);
      toast.success('Sentiment predicted successfully!');
    } catch (err) {
      setError(err.response?.data?.detail || 'An error occurred');
      toast.error(err.response?.data?.detail || 'An error occurred');
    } finally {
      setLoading(false);
    }
  };

  const handleClear = () => {
    setText('');
    setPrediction(null);
    setError(null);
    toast.info('Input cleared.');
  };

  // Chart data with gradient colors
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
            backgroundColor: (context) => {
              const ctx = context.chart.ctx;
              const gradient = ctx.createLinearGradient(0, 0, 0, 400);
              const isPositive = prediction.predicted_label === 'Positive';
              gradient.addColorStop(0, darkMode ? (isPositive ? '#86efac' : '#f87171') : (isPositive ? '#22c55e' : '#ef4444'));
              gradient.addColorStop(1, darkMode ? (isPositive ? '#22c55e' : '#ef4444') : (isPositive ? '#86efac' : '#f87171'));
              return gradient;
            },
            borderColor: darkMode ? ['#22c55e', '#ef4444'] : ['#15803d', '#b91c1c'],
            borderWidth: 1,
          },
        ],
      }
    : null;

  return (
    <ThemeProvider theme={theme}>
      <Box className={`App ${darkMode ? 'dark' : ''}`}>
        <ToastContainer position="top-right" autoClose={3000} />
        <motion.div
          initial={{ opacity: 0, y: -50 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
        >
          <Typography variant="h3" color="primary" gutterBottom>
            Sentiment Analysis with BERT
          </Typography>
          <Switch checked={darkMode} onChange={() => setDarkMode(!darkMode)} />
        </motion.div>

        <Paper elevation={4} sx={{ padding: 4, marginBottom: 4, backgroundColor: darkMode ? '#1f2937' : '#ffffff' }}>
          <form onSubmit={handleSubmit}>
            <TextField
              fullWidth
              multiline
              rows={5}
              value={text}
              onChange={(e) => setText(e.target.value)}
              placeholder="Enter your movie review..."
              variant="outlined"
              label="Movie Review"
              error={!!error}
              helperText={error || 'Type your review and click Predict Sentiment'}
              sx={{ marginBottom: 3 }}
            />
            <Box sx={{ display: 'flex', gap: 2, justifyContent: 'center' }}>
              <motion.div whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}>
                <Button
                  type="submit"
                  variant="contained"
                  color="primary"
                  disabled={loading}
                  startIcon={loading ? <CircularProgress size={20} /> : null}
                  sx={{ paddingX: 4, paddingY: 1.5 }}
                >
                  {loading ? 'Predicting...' : 'Predict Sentiment'}
                </Button>
              </motion.div>
              <motion.div whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}>
                <Button
                  variant="outlined"
                  color="secondary"
                  onClick={handleClear}
                  sx={{ paddingX: 4, paddingY: 1.5 }}
                >
                  Clear
                </Button>
              </motion.div>
            </Box>
          </form>
        </Paper>

        {prediction && (
          <motion.div
            initial={{ opacity: 0, scale: 0.8 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.5 }}
          >
            <Paper elevation={4} sx={{ padding: 4, marginBottom: 4, backgroundColor: darkMode ? '#1f2937' : '#ffffff' }}>
              <Typography variant="h5" gutterBottom>
                Prediction Result
              </Typography>
              <Box
                className={`sentiment-emoji sentiment-${prediction.predicted_label.toLowerCase()}${darkMode ? '.dark' : ''}`}
              >
                {prediction.predicted_label === 'Positive' ? '😊' : '😔'}
              </Box>
              <Typography variant="body1">
                <strong>Text:</strong> {prediction.text}
              </Typography>
              <Typography variant="body1">
                <strong>Sentiment:</strong>{' '}
                <span style={{ color: prediction.predicted_label === 'Positive' ? '#22c55e' : '#ef4444' }}>
                  {prediction.predicted_label}
                </span>
              </Typography>
              <Typography variant="body1">
                <strong>Confidence Score:</strong> {(prediction.score * 100).toFixed(2)}%
              </Typography>
              {chartData && (
                <Box sx={{ maxWidth: '500px', margin: '24px auto' }}>
                  <Bar
                    data={chartData}
                    options={{
                      responsive: true,
                      plugins: {
                        legend: { position: 'top', labels: { color: darkMode ? '#e0e7ff' : '#1f2937' } },
                        title: {
                          display: true,
                          text: 'Sentiment Probability Distribution',
                          font: { size: 16 },
                          color: darkMode ? '#e0e7ff' : '#1f2937',
                        },
                      },
                      scales: {
                        y: {
                          beginAtZero: true,
                          max: 1,
                          title: { display: true, text: 'Probability', color: darkMode ? '#e0e7ff' : '#1f2937' },
                          ticks: { color: darkMode ? '#e0e7ff' : '#1f2937' },
                        },
                        x: { ticks: { color: darkMode ? '#e0e7ff' : '#1f2937' } },
                      },
                      animation: {
                        duration: 1200,
                        easing: 'easeOutBounce',
                      },
                    }}
                  />
                </Box>
              )}
            </Paper>
          </motion.div>
        )}
      </Box>
    </ThemeProvider>
  );
}

export default App;