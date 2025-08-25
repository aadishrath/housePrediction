import React, { Component } from 'react';

// Dropdown component to select prediction model
class ModelSelector extends Component {
  render() {
    const { models, selectedModel, onChange } = this.props;
    return (
      <div>
        <label>Select Model:</label>
        <select value={selectedModel} onChange={onChange}>
          {models.map(model => (
            <option key={model} value={model.replace(/\s+/g, '_').toLowerCase()}>
              {model}
            </option>
          ))}
        </select>
      </div>
    );
  }
}

export default ModelSelector;
