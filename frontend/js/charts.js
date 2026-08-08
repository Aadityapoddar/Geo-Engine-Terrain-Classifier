// Chart & Area Analytics Controller Module
// Swiss Design Palette: monochrome + accent on white background

const ChartsController = (function () {
  let donutChart = null;

  const FALLBACK_COLOR_MAP = {
    "Forest": "#10b981",
    "Water": "#3b82f6",
    "Buildings": "#f43f5e",
    "Barren Land": "#d97706",
    "Agriculture": "#F59E0B"
  };

  const CLASS_DISPLAY_NAMES = {
    "Forest": "Vegetation",
    "Buildings": "Built Area",
    "Barren Land": "Open Land"
  };

  function displayClassName(name) {
    return CLASS_DISPLAY_NAMES[name] || name;
  }

  function renderTerrainAnalytics(summaryData, classAreasList) {
    // 1. Update Top Level Summary Stat Cards
    document.getElementById("stat-total-ha").textContent = `${summaryData.total_area_ha.toLocaleString()} ha`;
    document.getElementById("stat-proc-time").textContent = `${summaryData.processing_time_sec}s`;

    // Find Dominant Class
    let dominantClass = "None";
    let maxPct = -1;
    if (classAreasList && classAreasList.length > 0) {
      classAreasList.forEach((cls) => {
        if (cls.percentage > maxPct) {
          maxPct = cls.percentage;
          dominantClass = `${displayClassName(cls.name)} (${cls.percentage}%)`;
        }
      });
    }
    document.getElementById("stat-dominant").textContent = dominantClass;

    // 2. Render Donut Chart using Chart.js
    renderDonutChart(classAreasList);

    // 3. Render Individual Class Area Cards in UI List
    renderIndividualClassCards(classAreasList);
  }

  function renderDonutChart(classAreasList) {
    const canvas = document.getElementById("chart-terrain-distribution");
    if (!canvas) return;

    const ctx = canvas.getContext("2d");

    const labels = classAreasList.map((c) => displayClassName(c.name));
    const dataValues = classAreasList.map((c) => c.area_ha);
    const backgroundColors = classAreasList.map((c) => c.color || FALLBACK_COLOR_MAP[c.name]);

    if (donutChart) {
      donutChart.destroy();
    }

    donutChart = new Chart(ctx, {
      type: "doughnut",
      data: {
        labels: labels,
        datasets: [
          {
            data: dataValues,
            backgroundColor: backgroundColors,
            borderColor: "#ffffff",
            borderWidth: 3,
            hoverOffset: 4
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            position: "right",
            labels: {
              color: "#000000",
              font: {
                family: "'Inter', 'Helvetica Neue', Helvetica, Arial, sans-serif",
                size: 11,
                weight: "700"
              },
              padding: 14,
              usePointStyle: true,
              pointStyle: "rect",
              boxWidth: 10,
              boxHeight: 10
            }
          },
          tooltip: {
            backgroundColor: "#000000",
            titleColor: "#ffffff",
            bodyColor: "#ffffff",
            titleFont: {
              family: "'Inter', sans-serif",
              weight: "700"
            },
            bodyFont: {
              family: "'Inter', sans-serif"
            },
            borderColor: "#000000",
            borderWidth: 0,
            cornerRadius: 0,
            padding: 10,
            callbacks: {
              label: function (context) {
                const cls = classAreasList[context.dataIndex];
                return ` ${displayClassName(cls.name)}: ${cls.area_ha.toLocaleString()} ha (${cls.percentage}%)`;
              }
            }
          }
        },
        cutout: "70%"
      }
    });
  }

  function renderIndividualClassCards(classAreasList) {
    const container = document.getElementById("class-areas-list");
    if (!container) return;

    container.innerHTML = "";

    classAreasList.forEach((cls) => {
      const card = document.createElement("div");
      const classKey = cls.name.toLowerCase();
      card.className = `class-area-card ${classKey}`;

      const classColor = cls.color || FALLBACK_COLOR_MAP[cls.name];

      card.innerHTML = `
        <div class="class-card-header">
          <span class="class-name">
            <span class="class-badge" style="background-color: ${classColor};"></span>
            ${displayClassName(cls.name)}
          </span>
          <span class="class-pct">${cls.percentage}%</span>
        </div>

        <div class="progress-bar-bg">
          <div class="progress-bar-fill" style="width: ${cls.percentage}%; background-color: ${classColor};"></div>
        </div>

        <div class="class-metrics">
          <span><strong>${cls.area_ha.toLocaleString()}</strong> ha</span>
          <span><strong>${cls.area_sqm.toLocaleString()}</strong> m²</span>
          <span><strong>${cls.pixel_count.toLocaleString()}</strong> px</span>
        </div>
      `;

      container.appendChild(card);
    });
  }

  return {
    renderTerrainAnalytics
  };
})();
