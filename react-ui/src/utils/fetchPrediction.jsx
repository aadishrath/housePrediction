// Utility function to send prediction request to Flask backend

export const fetchPrediction = async (model, features) => {
  const response = await fetch('http://localhost:5000/predict', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ model, features })
  });

  if (!response.ok) throw new Error('Prediction failed');
  return await response.json();
};
