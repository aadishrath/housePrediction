import { useMemo } from "react";

function money(value) {
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    maximumFractionDigits: 0,
  }).format(value);
}

function PriceChart({ predictedPrice, low, high }) {
  const scale = useMemo(() => Math.max(high, 1), [high]);
  return (
    <div className="chart-inner">
      <p className="eyebrow">MODEL OUTPUT</p>
      <h3>Estimate and validation range</h3>
      <div
        className="estimate-range-visual"
        role="img"
        aria-label={`Estimate ${money(predictedPrice)}; validation range ${money(low)} to ${money(high)}`}
      >
        <div className="range-track">
          <span
            className="range-fill"
            style={{
              left: `${(low / scale) * 100}%`,
              width: `${Math.max(0, ((high - low) / scale) * 100)}%`,
            }}
          />
          <span
            className="range-marker"
            style={{ left: `${(predictedPrice / scale) * 100}%` }}
          />
        </div>
        <div className="range-endpoints">
          <span>{money(low)}</span>
          <strong>{money(predictedPrice)}</strong>
          <span>{money(high)}</span>
        </div>
      </div>
      <p className="muted chart-caption">
        This band comes from held-out records for the selected model.
      </p>
    </div>
  );
}

export default PriceChart;
