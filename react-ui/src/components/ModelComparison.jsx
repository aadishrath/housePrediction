const money = (value) =>
  new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    maximumFractionDigits: 0,
  }).format(value);

function ModelComparison({ models }) {
  return (
    <section className="model-report" aria-labelledby="model-report-title">
      <div className="section-heading">
        <div>
          <p className="eyebrow">HELD-OUT MODEL RESULTS</p>
          <h2 id="model-report-title">Compare model performance</h2>
          <p className="muted report-intro">
            Lower error scores are better; higher R² is better. The highlighted
            model has the lowest MAE on the same mixed-status validation split.
          </p>
        </div>
      </div>
      <div className="table-scroll">
        <table className="metrics-table">
          <thead>
            <tr>
              <th scope="col">Model</th>
              <th scope="col">R²</th>
              <th scope="col">MAE</th>
              <th scope="col">Median AE</th>
              <th scope="col">RMSE</th>
              <th scope="col">MAPE</th>
              <th scope="col">80% band</th>
              <th scope="col">Training</th>
            </tr>
          </thead>
          <tbody>
            {models.map(({ id, label, metrics = {}, best }) => (
              <tr key={id} className={best ? "best-model-row" : undefined}>
                <th scope="row">
                  {label}
                  {best && <span className="best-badge">Best MAE</span>}
                </th>
                <td>{metrics.r2 == null ? "—" : metrics.r2.toFixed(3)}</td>
                <td>{metrics.mae == null ? "—" : money(metrics.mae)}</td>
                <td>
                  {metrics.median_ae == null ? "—" : money(metrics.median_ae)}
                </td>
                <td>{metrics.rmse == null ? "—" : money(metrics.rmse)}</td>
                <td>
                  {metrics.mape == null
                    ? "—"
                    : `${(metrics.mape * 100).toFixed(1)}%`}
                </td>
                <td>
                  {metrics.validation_band_radius_80 == null
                    ? "—"
                    : `±${money(metrics.validation_band_radius_80)}`}
                </td>
                <td>{metrics.training_device || "CPU"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="muted metric-definitions">
        <strong>R²</strong> is the share of observed price variation explained.{" "}
        <strong>MAE</strong> is average absolute price error.{" "}
        <strong>Median AE</strong> is the typical absolute error.{" "}
        <strong>RMSE</strong> penalizes large misses more. <strong>MAPE</strong>{" "}
        is average error as a percent of observed price. Metrics combine
        completed-sale and asking-price records in the same status-stratified
        holdout; the score is a model comparison, not an expected error for any
        one location or price context.
      </p>
    </section>
  );
}

export default ModelComparison;
