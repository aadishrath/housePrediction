import React, { Component } from "react";
import "./App.css";
import ModelSelector from "./components/ModelSelector";
import DynamicForm from "./components/DynamicForm";
import PredictionResult from "./components/PredictionResult";
import PriceChart from "./components/PriceChart";
import { fetchPrediction } from "./utils/fetchPrediction";

// Main app component
class App extends Component {
  state = {
    models: [],
    selectedModel: "",
    features: [],
    formData: {},
    prediction: null,
    sampleInput: null,
  };

  // Load available models on mount
  async componentDidMount() {
    try {
      const res = await fetch("http://localhost:5000/models");
      const models = await res.json();

      if (models.length === 0) throw new Error("No models available");

      const defaultModel = models[0].replace(/\s+/g, "_").toLowerCase();
      this.setState({ models, selectedModel: defaultModel });
    } catch (err) {
      console.error("Failed to load models:", err);
    }
  }

  // Load features when model changes
  async componentDidUpdate(_, prevState) {
    const { selectedModel } = this.state;
    if (prevState.selectedModel !== selectedModel && selectedModel) {
      try {
        const [featuresRes, sampleRes] = await Promise.all([
          fetch(`http://localhost:5000/features/${selectedModel}`), // fetch features
          fetch(`http://localhost:5000/sample/${selectedModel}`),   // fetch sample input
        ]);

        const features = await featuresRes.json();
        const sampleInput = await sampleRes.json();

        this.setState({ features, sampleInput, formData: {}, prediction: null });
      } catch (err) {
        console.error("Failed to load features or sample input:", err);
      }
    }

  }

  // Handle model selection change
  handleModelChange = (e) => {
    this.setState({ selectedModel: e.target.value });
  };

  // Handle input change for each feature
  handleInputChange = (feature, value) => {
    this.setState((prev) => ({
      formData: { ...prev.formData, [feature]: value },
    }));
  };

  // Submit form and fetch prediction
  handleSubmit = async () => {
    try {
      const result = await fetchPrediction(this.state.selectedModel, this.state.formData);
      this.setState({ prediction: result });
    } catch (err) {
      alert("Prediction failed: " + err.message);
    }
  };

  // Clear the form fields
  handleClear = () => {
    this.setState({ formData: {}, prediction: null });
  };


  render() {
    const { models, selectedModel, features, formData, prediction, sampleInput } = this.state;

    return (
      <div className="app-container">
        <h1>🏠 House Price Predictor</h1>

        {/* Model selector dropdown */}
        <div className="highlighted-dropdown">
          <ModelSelector models={models} selectedModel={selectedModel} onChange={this.handleModelChange} />
        </div>

        {/* Two-column layout for sample + form */}
        {features.length > 0 && (
          <div className="input-container">
            
            {/* Left column: Sample input */}
            <div className="sample-input-section">
              <h2>📋 Sample Input</h2>
              {sampleInput ? (
                <>
                  <pre style={{backgroundColor: "#fff", padding: "10px", borderRadius: "6px", overflowX: "auto"}}>
                    {JSON.stringify(sampleInput, null, 2)}
                  </pre>
                  <button
                    style={{marginTop: "10px", padding: "8px 16px", backgroundColor: "#0078d4", color: "white", border: "none", borderRadius: "5px", cursor: "pointer"}}
                    onClick={() => this.setState({ formData: sampleInput })}>
                    Use Sample Input
                  </button>
                </>
              ) : (
                <p>Loading sample input...</p>
              )}
            </div>

            {/* Right column: Dynamic form */}
            <div className="form-section">
              <DynamicForm features={features} formData={formData} onInputChange={this.handleInputChange} />
              <div className="button-container">
                <button onClick={this.handleSubmit} className="submit-button">
                  Submit
                </button>
                <button onClick={this.handleClear} className="clear-button">
                  Clear
                </button>
              </div>
            </div>
          </div>
        )}

        {/* Prediction result and chart */}
        {prediction && (
          <div  className="chart-container">
            <PredictionResult price={prediction.predicted_price} similarPrices={prediction.similar_prices} />
            <PriceChart predictedPrice={prediction.predicted_price} similarPrices={prediction.similar_prices} />
          </div>
        )}
      </div>
    );
  }
}

export default App;
