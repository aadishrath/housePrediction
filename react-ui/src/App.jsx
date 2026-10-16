import { useCallback, useEffect, useState } from "react";
import "./App.css";
import ModelSelector from "./components/ModelSelector";
import DynamicForm from "./components/DynamicForm";
import PredictionResult from "./components/PredictionResult";
import PriceChart from "./components/PriceChart";
import ModelComparison from "./components/ModelComparison";
import { fetchPrediction } from "./utils/fetchPrediction";
import { validateInputs } from "./utils/validateInputs";

const API_URL = import.meta.env.VITE_API_URL || "/api";
const money = (value) =>
  new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    maximumFractionDigits: 0,
  }).format(value);

function App() {
  const [models, setModels] = useState([]);
  const [selectedModel, setSelectedModel] = useState("");
  const [features, setFeatures] = useState([]);
  const [metadata, setMetadata] = useState(null);
  const [formData, setFormData] = useState({});
  const [fieldErrors, setFieldErrors] = useState({});
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(true);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    let active = true;
    fetch(`${API_URL}/models`)
      .then(async (response) => {
        if (!response.ok)
          throw new Error("Could not connect to the prediction service.");
        return response.json();
      })
      .then((available) => {
        if (!active) return;
        setModels(available);
        if (!available.length)
          setError(
            "No trained models found. Run the training step in the project README.",
          );
        else setSelectedModel(available[0].id);
      })
      .catch(
        (cause) =>
          active &&
          setError(`${cause.message} Start the API and refresh this page.`),
      )
      .finally(() => active && setLoading(false));
    return () => {
      active = false;
    };
  }, []);

  useEffect(() => {
    if (!selectedModel) return;
    let active = true;
    setLoading(true);
    setError("");
    Promise.all([
      fetch(`${API_URL}/features/${selectedModel}`).then(async (response) => {
        const result = await response.json();
        if (!response.ok)
          throw new Error(result.error || "Could not load the model inputs.");
        return result;
      }),
      fetch(`${API_URL}/sample/${selectedModel}`).then((response) =>
        response.json(),
      ),
    ])
      .then(([nextFeatures, sample]) => {
        if (!active) return;
        if (nextFeatures.error || sample.error)
          throw new Error(nextFeatures.error || sample.error);
        setFeatures(nextFeatures.fields);
        setMetadata(nextFeatures.metadata);
        setFormData(sample);
        setFieldErrors({});
        setPrediction(null);
      })
      .catch(
        (cause) =>
          active &&
          setError(cause.message || "Could not load the selected model."),
      )
      .finally(() => active && setLoading(false));
    return () => {
      active = false;
    };
  }, [selectedModel]);

  const handleInputChange = useCallback((feature, value) => {
    setFormData((current) => ({ ...current, [feature]: value }));
    setFieldErrors((current) => ({ ...current, [feature]: undefined }));
  }, []);

  const useSample = async () => {
    try {
      const response = await fetch(`${API_URL}/sample/${selectedModel}`);
      const sample = await response.json();
      if (!response.ok)
        throw new Error(sample.error || "Could not load a sample.");
      setFormData(sample);
      setFieldErrors({});
      setPrediction(null);
      setError("");
    } catch (cause) {
      setError(cause.message);
    }
  };

  const clearDetails = () => {
    setFormData({});
    setFieldErrors({});
    setPrediction(null);
    setError("");
  };

  const handleSubmit = async (event) => {
    event.preventDefault();
    setError("");
    const validationErrors = validateInputs(features, formData);
    setFieldErrors(validationErrors);
    if (Object.keys(validationErrors).length) {
      setError(
        "Review the highlighted property details and correct them before estimating.",
      );
      return;
    }
    setSubmitting(true);
    try {
      const result = await fetchPrediction(selectedModel, formData, API_URL);
      setPrediction(result);
    } catch (cause) {
      setError(cause.message);
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <main className="app-container">
      <section className="hero">
        <div>
          <p className="eyebrow">USA · HOME PRICE ESTIMATES</p>
          <h1>
            Estimate a home’s value <span>from its details.</span>
          </h1>
          <p className="hero-copy">
            Trained on {metadata?.rows_used?.toLocaleString() || "millions of"}{" "}
            valid sale and listing records across the United States. Enter a
            home’s characteristics and choose whether you want a completed-sale
            or asking-price estimate.
          </p>
        </div>
      </section>

      {error && (
        <div className="notice" role="alert">
          {error}
        </div>
      )}
      {loading && (
        <div className="loading-card" role="status">
          Preparing your estimate…
        </div>
      )}

      {!loading && models.length > 0 && (
        <>
          <section className="model-row" aria-label="Prediction model">
            <div>
              <p className="eyebrow">PREDICTION MODEL</p>
              <p className="muted">
                Compared on a held-out sample.{" "}
                {models[0]?.metrics?.mae != null &&
                  `Lowest holdout MAE: ${money(models[0].metrics.mae)}.`}
              </p>
            </div>
            <ModelSelector
              models={models}
              selectedModel={selectedModel}
              onChange={(event) => setSelectedModel(event.target.value)}
            />
          </section>

          {features.length > 0 && (
            <form className="workspace" onSubmit={handleSubmit}>
              <section className="form-panel">
                <div className="section-heading">
                  <div>
                    <p className="eyebrow">PROPERTY DETAILS</p>
                    <h2>Describe the property</h2>
                  </div>
                  <button
                    type="button"
                    className="text-button"
                    onClick={useSample}
                  >
                    Load a sample
                  </button>
                </div>
                <DynamicForm
                  features={features}
                  formData={formData}
                  errors={fieldErrors}
                  onInputChange={handleInputChange}
                />
                <div className="form-actions">
                  <button
                    className="primary-button"
                    type="submit"
                    disabled={submitting || loading}
                  >
                    {submitting ? "Calculating…" : "Estimate value"}
                    <span aria-hidden="true">→</span>
                  </button>
                  <button
                    className="secondary-button"
                    type="button"
                    onClick={clearDetails}
                  >
                    Clear details
                  </button>
                </div>
              </section>
              <aside className="context-panel">
                <div className="context-icon" aria-hidden="true">
                  ⌖
                </div>
                <p className="eyebrow">PRICE CONTEXT</p>
                <h2>Choose what kind of price to estimate.</h2>
                <p className="muted">
                  Sale prices and asking prices are different outcomes. Pick a
                  price context, then enter the property and location details.
                </p>
                <div className="context-note">
                  <span className="status-dot" /> Trained across all included
                  records
                </div>
              </aside>
            </form>
          )}

          {prediction && (
            <section className="results-section" aria-live="polite">
              <div className="section-heading">
                <div>
                  <p className="eyebrow">STATUS-CONDITIONED ESTIMATE</p>
                  <h2>
                    {prediction.price_context === "sold"
                      ? "Estimated completed-sale price"
                      : "Estimated asking price"}
                  </h2>
                </div>
              </div>
              <div className="results-grid">
                <PredictionResult
                  price={prediction.predicted_price}
                  low={prediction.estimated_low}
                  high={prediction.estimated_high}
                  radius={prediction.validation_band_radius}
                  context={prediction.price_context}
                />
                <div className="chart-card">
                  <PriceChart
                    predictedPrice={prediction.predicted_price}
                    low={prediction.estimated_low}
                    high={prediction.estimated_high}
                  />
                </div>
              </div>
              <p className="disclaimer">
                The range reflects an absolute error band measured on a random
                held-out sample from this dataset. The source does not provide a
                reliable current listing/sale observation date, so estimates are
                not current-market appraisals or offers.
              </p>
            </section>
          )}
        </>
      )}
      {models.length > 0 && <ModelComparison models={models} />}
      <footer className="page-footer">
        A practical look at machine learning and local housing data.
      </footer>
    </main>
  );
}

export default App;
