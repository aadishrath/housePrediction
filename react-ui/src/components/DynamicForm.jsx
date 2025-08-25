import React, { Component } from 'react';

// Form component that dynamically renders inputs based on selected features
class DynamicForm extends Component {
  handleChange = (e, feature) => {
    const value = parseFloat(e.target.value);
    this.props.onInputChange(feature, value);
  };

  render() {
    const { features, formData } = this.props;
    return (
      <form>
        {features.map(feature => (
          <div className="form-group" key={feature}>
            <label>{feature}</label>
            <input
              type={feature.toLowerCase().includes("lat") || feature.toLowerCase().includes("long") ? "number" : "text"}
              value={formData[feature] || ''}
              onChange={(e) => this.handleChange(e, feature)}
            />
          </div>
        ))}
      </form>
    );
  }
}

export default DynamicForm;
