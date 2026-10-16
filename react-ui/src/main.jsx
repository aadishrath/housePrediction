import React from "react";
import { createRoot } from "react-dom/client";
import { BrowserRouter, NavLink, Route, Routes } from "react-router-dom";
import "./index.css";
import "./App.css";
import App from "./App.jsx";
import About from "./components/About.jsx";

createRoot(document.getElementById("root")).render(
  <BrowserRouter>
    <header className="top-bar">
      <NavLink to="/" className="brand-link"><span className="brand-icon" aria-hidden="true">⌂</span> AdiReality</NavLink>
      <nav aria-label="Main navigation">
        <NavLink to="/" end className={({ isActive }) => `nav-button${isActive ? " active" : ""}`}>Estimate</NavLink>
        <NavLink to="/about" className={({ isActive }) => `nav-button${isActive ? " active" : ""}`}>About</NavLink>
      </nav>
    </header>
    <Routes>
      <Route path="/" element={<App />} />
      <Route path="/about" element={<About />} />
    </Routes>
  </BrowserRouter>,
);
