const form = document.getElementById("analyze-form");
const statusBox = document.getElementById("status");
const metaBox = document.getElementById("meta");

const mHashtag = document.getElementById("m-hashtag");
const mGemini = document.getElementById("m-gemini");
const mFallback = document.getElementById("m-fallback");
const mModels = document.getElementById("m-models");

const pieDiv = document.getElementById("pie");
const lineDiv = document.getElementById("line");
const tableDiv = document.getElementById("table");

// Parallax cursor
const cursor = document.getElementById("parallax-cursor");
window.addEventListener("mousemove", (e) => {
  const x = e.clientX, y = e.clientY;
  cursor.style.opacity = ".9";
  cursor.style.left = x + "px";
  cursor.style.top = y + "px";
});

// Helpers
function fmtPct(n){ return (Math.round(n * 100) / 100).toFixed(2); }

function renderMeta(meta) {
  mHashtag.textContent = meta.hashtag;
  mGemini.textContent = `Gemini: ${meta.generated_by.gemini}`;
  mFallback.textContent = `Fallback: ${meta.generated_by.fallback}`;
  mModels.textContent = `Gen: ${meta.model.generation} • Sentiment: ${meta.model.sentiment}`;
}

function renderPie(percent) {
  const data = [{
    values: [percent.positive, percent.neutral, percent.negative],
    labels: ['Positive', 'Neutral', 'Negative'],
    type: 'pie',
    textinfo: 'label+percent',
    hoverinfo: 'label+percent',
    hole: .35
  }];
  const layout = {
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(0,0,0,0)',
    font: {color: '#eaf2ff'},
    margin: {l: 4, r: 4, t: 0, b: 0},
    showlegend: false
  };
  Plotly.newPlot(pieDiv, data, layout, {displayModeBar:false, responsive:true});
}

function renderLine(rolling) {
  const data = [{
    x: [...Array(rolling.length).keys()].map(i => i+1),
    y: rolling,
    type: 'scatter',
    mode: 'lines+markers'
  }];
  const layout = {
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(0,0,0,0)',
    font: {color: '#eaf2ff'},
    margin: {l: 30, r: 10, t: 0, b: 24},
    yaxis: {range:[0,1], tickformat: '.0%'},
  };
  Plotly.newPlot(lineDiv, data, layout, {displayModeBar:false, responsive:true});
}

function renderTable(rows) {
  tableDiv.innerHTML = "";
  rows.forEach(r => {
    const row = document.createElement("div");
    row.className = "row";
    // text
    const c1 = document.createElement("div");
    c1.className = "cell";
    c1.textContent = r.text;
    // source chip
    const c2 = document.createElement("div");
    c2.className = "cell";
    const chip = document.createElement("span");
    chip.className = "chip " + (r.source === "gemini" ? "chip-gemini" : "chip-fallback");
    chip.textContent = r.source === "gemini" ? "Gemini" : "Fallback";
    c2.appendChild(chip);
    // sentiment badge
    const c3 = document.createElement("div");
    c3.className = "cell";
    const badge = document.createElement("span");
    const s = r.sentiment;
    badge.className = "badge " + (s === "POSITIVE" ? "pos" : s === "NEGATIVE" ? "neg" : "neu");
    badge.textContent = s + " " + (r.score.toFixed(2));
    c3.appendChild(badge);

    row.appendChild(c1);
    row.appendChild(c2);
    row.appendChild(c3);
    tableDiv.appendChild(row);
  });
}

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  const hashtag = document.getElementById("hashtag").value.trim();
  const count = parseInt(document.getElementById("count").value || "20", 10);

  if(!hashtag){
    alert("Please enter a hashtag (e.g., #gla)");
    return;
  }

  statusBox.classList.remove("hidden");
  metaBox.classList.add("hidden");

  try {
    const resp = await fetch("/api/analyze", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({hashtag, count})
    });
    if(!resp.ok){
      const err = await resp.json().catch(()=>({}));
      throw new Error(err.error || `HTTP ${resp.status}`);
    }
    const data = await resp.json();

    // META
    renderMeta(data.meta);
    metaBox.classList.remove("hidden");

    // CHARTS
    renderPie(data.aggregate.percent);
    renderLine(data.aggregate.rolling);

    // TABLE
    renderTable(data.rows);

  } catch (err) {
    console.error(err);
    alert("Failed: " + err.message);
  } finally {
    statusBox.classList.add("hidden");
  }
});
