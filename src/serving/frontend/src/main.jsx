/* Entry point: mount the app and drop the static loading overlay from
 * index.html once the shell is interactive (predictions stream in afterward
 * with the lighter in-view loading indicators). */
import { createRoot } from "react-dom/client";
import { applyChartTheme } from "./lib/chartTheme.js";
import { App } from "./App.jsx";

applyChartTheme();
createRoot(document.getElementById("root")).render(<App />);

const overlay = document.getElementById("loading-overlay");
if (overlay) overlay.classList.add("hidden");
