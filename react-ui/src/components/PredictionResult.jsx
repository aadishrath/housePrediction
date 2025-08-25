import React, { Component } from 'react';

// Component to display predicted house price
class PredictionResult extends Component {
  render() {
    const { price, similarPrices } = this.props;

    const meanOfPrices = similarPrices.reduce((sum, p) => sum + p, 0) / similarPrices.length
;
    return (
      <div>
        <h2>Predicted Price: {price.toLocaleString('en-US', { style: 'currency', currency: 'USD' })}</h2>
        <h3>Mean Price of similar properties: {meanOfPrices.toLocaleString('en-US', { style: 'currency', currency: 'USD' })}</h3>
      </div>
    );
  }
}

export default PredictionResult;
