function ModelSelector({ models, selectedModel, onChange }) {
  return (
    <label className="model-select">
      <span className="sr-only">Select prediction model</span>
      <select value={selectedModel} onChange={onChange}>
        {models.map((model) => (
          <option key={model.id} value={model.id}>
            {model.best ? "★ " : ""}
            {model.label}
            {model.metrics?.mae != null
              ? ` · $${Math.round(model.metrics.mae / 1000)}k MAE`
              : ""}
          </option>
        ))}
      </select>
      <span className="select-chevron" aria-hidden="true">
        ⌄
      </span>
    </label>
  );
}

export default ModelSelector;
