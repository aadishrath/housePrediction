function DynamicForm({ features, formData, errors, onInputChange }) {
  const status = formData.status;
  return (
    <div className="feature-grid">
      {features.map((field) => (
        <label className="form-group" key={field.id}>
          <span>{field.label}</span>
          {field.type === "select" ? (
            <select
              value={formData[field.id] ?? ""}
              onChange={(event) => onInputChange(field.id, event.target.value)}
              required={field.required !== false}
              aria-label={field.label}
              aria-invalid={Boolean(errors[field.id])}
              aria-describedby={errors[field.id] ? `${field.id}-error` : undefined}
            >
              <option value="" disabled>Choose {field.label.toLowerCase()}</option>
              {field.options.map((option) => (
                <option key={option.value} value={option.value}>{option.label}</option>
              ))}
            </select>
          ) : field.type === "text" ? (
            <input
              type="text"
              minLength={field.min_length}
              maxLength={field.max_length}
              value={formData[field.id] ?? ""}
              onChange={(event) => onInputChange(field.id, event.target.value)}
              required={field.required !== false}
              aria-label={field.label}
              aria-invalid={Boolean(errors[field.id])}
              aria-describedby={errors[field.id] ? `${field.id}-error` : field.help ? `${field.id}-help` : undefined}
            />
          ) : (
            <input
              type="number"
              inputMode="decimal"
              min={field.min}
              max={field.max}
              step={field.step}
              value={formData[field.id] ?? ""}
              onChange={(event) => onInputChange(field.id, event.target.value)}
              required={field.required !== false && !(field.required_unless && status === field.required_unless)}
              aria-label={field.label}
              aria-invalid={Boolean(errors[field.id])}
              aria-describedby={errors[field.id] ? `${field.id}-error` : field.help ? `${field.id}-help` : undefined}
            />
          )}
          {field.help && <small id={`${field.id}-help`}>{field.help}</small>}
          {errors[field.id] && <small className="field-error" id={`${field.id}-error`} role="alert">{errors[field.id]}</small>}
        </label>
      ))}
    </div>
  );
}

export default DynamicForm;
