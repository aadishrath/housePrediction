function FeatureRationale({ featureGuide, rawFeatures }) {
  return (
    <details className="feature-report" open>
      <summary>
        <span><span className="eyebrow">DATA AND FEATURE CHOICES</span><strong>What the models learn from</strong></span>
        <span className="summary-hint">Show or hide details</span>
      </summary>
      <div className="raw-feature-list">
        <h3>Raw dataset columns ({rawFeatures.length})</h3>
        <p className="muted">These are the columns from the source file. Price is the target, interpreted using status: sold records are completed sale prices; other statuses are asking prices.</p>
        <div className="feature-chips">{rawFeatures.map((feature) => <code key={feature}>{feature}</code>)}</div>
      </div>
      <div className="feature-reasons">
        <section>
          <h3>Used for training ({featureGuide.included.length})</h3>
          <ul>{featureGuide.included.map((feature) => <li key={feature.id}><strong>{feature.id}</strong><span>{feature.reason}</span></li>)}</ul>
        </section>
        <section>
          <h3>Dropped from model inputs ({featureGuide.excluded.length})</h3>
          <ul>{featureGuide.excluded.map((feature) => <li key={feature.id}><strong>{feature.id}</strong><span>{feature.reason}</span></li>)}</ul>
        </section>
      </div>
      <p className="muted feature-note">The model inputs are the same for every candidate, so holdout metrics can be compared directly. Location values are encoded as categories, not ordered numbers.</p>
    </details>
  );
}

export default FeatureRationale;
