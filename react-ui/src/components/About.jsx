import React, { Component } from "react";

class About extends Component {
  render() {
    return (
      <div className="about-container">
        <h1>About House Price Predictor</h1>
        <p>
          Welcome to the House Price Predictor — a project born from curiosity, data, and a love for clean design. Whether you're a homeowner wondering what your property might be worth, or a data enthusiast exploring machine learning in action, this app is here to make predictions simple, fast, and surprisingly fun.
        </p>

        <h2>🔧 How It Works</h2>
        <p>
          It all starts with you. Choose a prediction model from the dropdown, enter your property details — square footage, bedrooms, location, and more — and let the app do the rest. Behind the scenes, your inputs are sent to a Flask-powered backend, which loads a trained model and returns a price estimate in seconds.
        </p>

        <h2>🧠 Machine Learning Models</h2>
        <p>
          The app supports a variety of models, from classic Linear Regression to powerful ensembles like Random Forest and XGBoost, and even Neural Networks. Each model has been trained on real housing data and fine-tuned to balance accuracy and speed. You get to pick the brain behind your prediction.
        </p>

        <h2>🎨 Frontend Experience</h2>
        <p>
          Built with React (yes, the class-based kind), the frontend is designed to be intuitive and responsive. The form adapts to your chosen model, sample inputs help guide you, and the chart gives you a visual comparison of your prediction against similar homes. It’s not just functional — it’s friendly.
        </p>

        <h2>📦 Backend Architecture</h2>
        <p>
          The Flask backend is the engine room. It handles everything from loading models and preprocessing inputs to generating predictions. With joblib for model storage and pandas for data wrangling, it’s built to be fast, scalable, and easy to extend. CORS support ensures smooth communication with the frontend.
        </p>

        <h2>🚀 Why This Project Matters</h2>
        <p>
          This isn’t just a demo — it’s a full-stack showcase of how machine learning can be woven into real-world applications. It brings together data science, API design, frontend development, and user experience in one cohesive package. Whether you're building your portfolio or just exploring, there's something here for you.
        </p>

        <h2>📁 Technologies Used</h2>
        <ul>
          <li>React (Class Components)</li>
          <li>Flask</li>
          <li>Scikit-learn, XGBoost, TensorFlow</li>
          <li>Chart.js</li>
          <li>CSS</li>
        </ul>

      </div>
    );
  }
}

export default About;
