import type { ClassArea } from "./api";

// ponytail: Chart.js was 70 KB gzipped to draw one static donut. A donut is a
// circle with a dashed stroke, so this is that, in SVG the browser already has.
const R = 54;
const C = 2 * Math.PI * R;

export function renderDonut(host: HTMLElement, classes: ClassArea[]) {
  const shown = classes.filter((c) => c.percentage > 0);
  let offset = 0;

  const arcs = shown
    .map((c) => {
      const dash = (c.percentage / 100) * C;
      const arc = `<circle class="donut-arc" cx="70" cy="70" r="${R}" fill="none"
        stroke="${c.color}" stroke-width="22"
        stroke-dasharray="${dash} ${C - dash}" stroke-dashoffset="${-offset}">
        <title>${c.name}: ${c.area_ha.toLocaleString()} ha (${c.percentage}%)</title>
      </circle>`;
      offset += dash;
      return arc;
    })
    .join("");

  const legend = shown
    .map(
      (c) => `<li><span class="swatch" style="background:${c.color}"></span>
        <span class="legend-name">${c.name}</span>
        <span class="legend-pct">${c.percentage}%</span></li>`,
    )
    .join("");

  host.innerHTML = `
    <svg class="donut" viewBox="0 0 140 140" role="img" aria-label="Terrain class distribution">
      <g transform="rotate(-90 70 70)">${arcs}</g>
    </svg>
    <ul class="donut-legend">${legend}</ul>`;
}

export function renderClassCards(host: HTMLElement, classes: ClassArea[]) {
  host.innerHTML = classes
    .map(
      (c) => `
    <div class="class-area-card">
      <div class="class-card-header">
        <span class="class-name"><span class="class-badge" style="background:${c.color}"></span>${c.name}</span>
        <span class="class-pct">${c.percentage}%</span>
      </div>
      <div class="progress-bar-bg"><div class="progress-bar-fill" style="width:${c.percentage}%;background:${c.color}"></div></div>
      <div class="class-metrics">
        <span><strong>${c.area_ha.toLocaleString()}</strong> ha</span>
        <span><strong>${c.area_sqm.toLocaleString()}</strong> m²</span>
        <span><strong>${c.pixel_count.toLocaleString()}</strong> px</span>
      </div>
    </div>`,
    )
    .join("");
}
