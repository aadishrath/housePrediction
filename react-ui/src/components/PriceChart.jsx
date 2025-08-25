import React, { Component } from 'react';
import { Line } from 'react-chartjs-2';
import { Chart as ChartJS, LineElement, PointElement, LineController, CategoryScale, LinearScale, Title, Tooltip, Legend } from 'chart.js';

// Register required components
ChartJS.register(LineElement, PointElement, LineController, CategoryScale, LinearScale, Title, Tooltip, Legend);


// Line chart comparing predicted price with similar house prices
class PriceChart extends Component {
  render() {
    const { predictedPrice, similarPrices } = this.props;
    const data = {
      labels: similarPrices.map((_, i) => `House ${i + 1}`),
      datasets: [
        {
          label: 'Similar House Prices',
          data: similarPrices,
          borderColor: 'blue',
          fill: false
        },
        {
          label: 'Your Prediction',
          data: Array(similarPrices.length).fill(predictedPrice),
          borderColor: 'red',
          borderDash: [5, 5],
          fill: false
        }
      ]
    };

    const options = {
      responsive: true,
      scales: {
        x: {
          type: 'category',
          title: {
            display: true,
            text: 'Similar Houses'
          }
        },
        y: {
          type: 'linear',
          title: {
            display: true,
            text: 'Price ($)'
          }
        }
      }
    };

    return <Line data={data} options={options} />;
  }
}

export default PriceChart;
