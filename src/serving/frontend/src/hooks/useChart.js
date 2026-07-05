/* Chart.js lifecycle hook: build the chart when deps change, destroy on
 * cleanup (tab switches / theme flips must not leak canvas contexts). The
 * config builder receives the active theme palette so grid/text colors track
 * the OLED toggle — include `theme` in deps wherever the toggle should rebuild. */
import { useEffect, useRef } from "react";
import { applyChartTheme } from "../lib/chartTheme.js";

export function useChart(makeConfig, deps) {
    const canvasRef = useRef(null);
    useEffect(() => {
        if (!canvasRef.current || !window.Chart) return undefined;
        const t = applyChartTheme();
        const config = makeConfig(t);
        if (!config) return undefined;
        const chart = new window.Chart(canvasRef.current, config);
        return () => chart.destroy();
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, deps);
    return canvasRef;
}
