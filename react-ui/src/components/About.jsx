import { useEffect, useState } from "react";
import FeatureRationale from "./FeatureRationale";

const API_URL = import.meta.env.VITE_API_URL || "/api";

function About() {
  const [details, setDetails] = useState(null);
  const [error, setError] = useState("");
  useEffect(() => {
    let active = true;
    fetch(`${API_URL}/models`)
      .then((response) => response.json())
      .then(async (models) => {
        if (!models.length) throw new Error("No trained models are available.");
        const response = await fetch(`${API_URL}/features/${models[0].id}`);
        const result = await response.json();
        if (!response.ok)
          throw new Error(result.error || "Could not load model details.");
        if (active) setDetails(result);
      })
      .catch((cause) => active && setError(cause.message));
    return () => {
      active = false;
    };
  }, []);

  return (
    <main className="about-container">
      <p className="eyebrow">ABOUT ADIREALITY</p>
      <h1>One national dataset. Property-specific estimates.</h1>
      <p>
        AdiReality uses the Realtor.com USA Real Estate dataset supplied with
        this project. A model learns from all retained records and predicts the
        price associated with property and location details entered on the
        estimate page. It does not build a month-by-month trend.
      </p>

      <h2>What price does the model estimate?</h2>
      <p>
        The dataset's <code>price</code> column has different meanings by{" "}
        <code>status</code>. For <strong>sold</strong>, it is a completed sale
        price. For <strong>for_sale</strong> and <strong>ready_to_build</strong>
        , it is an asking/list price. The form asks you to select the matching
        price context, and the status is included as a model input so the
        estimate is conditional on that meaning. Asking prices are
        seller/listing expectations, not confirmed transaction prices.
      </p>

      <h2>Data and modeling</h2>
      <p>
        The raw file contains broker and street identifiers, status, price, room
        counts, lot size, city/state/ZIP, home size, and previous sale date.
        Training uses status, beds, baths, home size, lot size, and location.
        Broker/street identifiers are excluded because they identify records or
        agents rather than portable property attributes. Previous sale date is
        excluded because it is not the current listing or sale date, is
        incomplete, and includes invalid values. Numeric missing values are
        imputed within each training fold; missing geography or unusable target
        records are excluded. Implausible prices and property measurements are
        filtered using the documented ranges.
      </p>
      <p>
        Geographic categories use cross-fitted target encoding to avoid one-hot
        expansion of millions of records and thousands of locations. Six
        candidates are compared: Linear Regression as a baseline, Linear SVR
        trained with an epsilon-insensitive SGD solver, a PyTorch neural
        network, histogram gradient boosting, random forest, and XGBoost. The
        neural network and XGBoost use CUDA when available and fall back to CPU;
        saved artifacts run on CPU. A random, status-stratified holdout compares
        candidates; after that comparison, each saved model is refit using every
        retained row. The holdout is useful for comparing candidates on this
        snapshot, but it is not a test on future listings or unseen regions.
      </p>

      {details ? (
        <FeatureRationale
          featureGuide={details.feature_guide}
          rawFeatures={details.raw_features}
        />
      ) : (
        error && (
          <p className="notice" role="status">
            {error}
          </p>
        )
      )}

      <h2>Interpreting results</h2>
      <p>
        The estimate and validation range summarize model output and errors on
        held-out records. They are not a formal appraisal, offer, or guaranteed
        range for one home. The source does not provide a reliable current
        observation date, so this dataset cannot establish today's market value
        or infer a temporal price trend. Performance can vary by state, city,
        property type, and price context.
      </p>

      <h2>Data source</h2>
      <p>
        <a
          href="https://www.kaggle.com/datasets/ahmedshahriarsakib/usa-real-estate-dataset"
          target="_blank"
          rel="noreferrer"
        >
          USA Real Estate Dataset on Kaggle
        </a>
        . The project expects the local source file to be named{" "}
        <code>house_data.csv</code>; it is excluded from version control because
        of its size.
      </p>
      <h2>Built with</h2>
      <ul>
        <li>React and Vite</li>
        <li>Flask API</li>
        <li>scikit-learn and XGBoost</li>
      </ul>
    </main>
  );
}

export default About;
