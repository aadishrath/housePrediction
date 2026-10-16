function money(value) {
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    maximumFractionDigits: 0,
  }).format(value);
}

function PredictionResult({ price, low, high, radius, context }) {
  const label =
    context === "sold" ? "Completed-sale estimate" : "Asking-price estimate";
  return (
    <div className="estimate-card">
      <p className="eyebrow">{label.toUpperCase()}</p>
      <p className="estimate-value">{money(price)}</p>
      <div className="estimate-divider" />
      <p className="range-label">Approximate validation range</p>
      <p className="range-value">
        {money(low)} <span>to</span> {money(high)}
      </p>
      <p className="muted small-copy">
        An 80% absolute-error band measured on the held-out sample (±
        {money(radius)}). It describes model performance across records, not a
        guarantee for this home.
      </p>
    </div>
  );
}

export default PredictionResult;
