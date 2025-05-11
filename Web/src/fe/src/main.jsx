import React from "react";
import ReactDOM from "react-dom/client";
import { BrowserRouter } from "react-router-dom";
import App from "./App.jsx";
// Mantine styles
import "@mantine/core/styles.css";
import { VideoProcessingProvider } from "./context/VideoProcessingContext.jsx";

import "./index.css";
//chatbot styles

ReactDOM.createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <BrowserRouter>
      <VideoProcessingProvider>
        <App />
      </VideoProcessingProvider>
    </BrowserRouter>
  </React.StrictMode>
);