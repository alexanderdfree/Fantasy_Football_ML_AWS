/* useFilterFit — "priority plus" auto-fit for the filter bars.
 * A hidden copy of every filter control (plus the right-side menus block)
 * renders off-screen; this hook measures it against the real row's width and
 * reports how many controls fit on one line, in list order, never fewer than
 * one. Recomputes on any resize (window, container, or font reflow). */
import { useLayoutEffect, useRef, useState } from "react";

export function useFilterFit(filterCount) {
    const rowRef = useRef(null);
    const measureRef = useRef(null);
    const [fit, setFit] = useState(1);

    useLayoutEffect(() => {
        const recompute = () => {
            const row = rowRef.current;
            const meas = measureRef.current;
            if (!row || !meas) return;
            const avail = row.clientWidth;
            if (avail <= 0 || meas.children.length < 2) return;
            const kids = [...meas.children];
            const menusW = kids[kids.length - 1].offsetWidth;
            const gap = parseFloat(getComputedStyle(row).columnGap) || 16;
            let used = menusW;
            let n = 0;
            for (let i = 0; i < kids.length - 1; i++) {
                const w = kids[i].offsetWidth + gap;
                if (n >= 1 && used + w > avail) break;
                used += w;
                n += 1;
            }
            setFit(Math.max(1, Math.min(n, filterCount)));
        };
        recompute();
        const ro = new ResizeObserver(recompute);
        if (rowRef.current) ro.observe(rowRef.current);
        if (measureRef.current) ro.observe(measureRef.current);
        window.addEventListener("resize", recompute);
        return () => {
            ro.disconnect();
            window.removeEventListener("resize", recompute);
        };
    }, [filterCount]);

    return { rowRef, measureRef, fit };
}
