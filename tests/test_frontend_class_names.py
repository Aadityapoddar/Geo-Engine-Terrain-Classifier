import subprocess
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]


def test_frontend_uses_requested_class_display_names():
    harness = r"""
const fs = require('fs');
const vm = require('vm');
const source = fs.readFileSync(process.argv[2], 'utf8');
const elements = {};
const container = { innerHTML: '', cards: [], appendChild(card) { this.cards.push(card); } };
const document = {
  getElementById(id) {
    if (id === 'class-areas-list') return container;
    if (id === 'chart-terrain-distribution') return { getContext() { return {}; } };
    return elements[id] ||= { textContent: '' };
  },
  createElement() { return { className: '', innerHTML: '' }; }
};
let chartConfig;
function Chart(_ctx, config) { chartConfig = config; this.destroy = () => {}; }
const context = { document, Chart, console };
vm.runInNewContext(source + ';globalThis.controller = ChartsController;', context);
context.controller.renderTerrainAnalytics(
  { total_area_ha: 10, processing_time_sec: 1 },
  [
    { name: 'Forest', percentage: 50, area_ha: 5, area_sqm: 50000, pixel_count: 500, color: '#0f0' },
    { name: 'Buildings', percentage: 30, area_ha: 3, area_sqm: 30000, pixel_count: 300, color: '#f00' },
    { name: 'Barren Land', percentage: 20, area_ha: 2, area_sqm: 20000, pixel_count: 200, color: '#aaa' }
  ]
);
const rendered = container.cards.map((card) => card.innerHTML).join(' ');
if (elements['stat-dominant'].textContent !== 'Vegetation (50%)') throw new Error('dominant label');
if (chartConfig.data.labels.join('|') !== 'Vegetation|Built Area|Open Land') throw new Error('chart labels');
for (const label of ['Vegetation', 'Built Area', 'Open Land']) {
  if (!rendered.includes(label)) throw new Error(`missing ${label}`);
}
"""
    result = subprocess.run(
        ["node", "-", str(REPO / "frontend/js/charts.js")],
        input=harness,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
