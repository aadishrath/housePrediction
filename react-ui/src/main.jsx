import React from 'react';
import { createRoot } from 'react-dom/client';
import { BrowserRouter as Router, Routes, Route, Link } from "react-router-dom";
import './index.css';
import App from './App.jsx';
import About from "./components/About";
import homeIcon from './assets/home.png';

const root = createRoot(document.getElementById('root'));

root.render(
  <>
    <Router>
      <div className="top-bar">
        <Link to="/" className="nav-button">
          <img src={homeIcon} alt="Home" style={{ width: '20px', marginRight: '8px', verticalAlign: 'middle' }} />
          Home
        </Link>
        <Link to="/about" className="nav-button">About</Link>
      </div>

      <Routes>
        <Route path="/" element={<App />} />
        <Route path="/about" element={<About />} />
      </Routes>
    </Router>
  </>,
)
