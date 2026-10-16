// Utility function to send prediction request to Flask backend

export const fetchPrediction = async (model, features, apiUrl) => {
  const response = await fetch(`${apiUrl}/predict`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ model, features })
  });

  const result = await response.json();
  if (!response.ok) throw new Error(result.error || 'Prediction failed.');
  return result;
};
