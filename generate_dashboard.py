"""
generate_dashboard.py  —  builds confusion_matrix_dashboard_live.html
Run: py -3.12 generate_dashboard.py
Open: confusion_matrix_dashboard_live.html  (no server needed)
"""
import json

with open("eval_results.json", "r", encoding="utf-8") as f:
    data = json.load(f)

s = data["summary"]
results = data["results"]
cats = s["by_category"]
ts = data["timestamp"]
tp, fp, tn, fn = s["TP"], s["FP"], s["TN"], s["FN"]
m = s["metrics"]

def esc(t): return t.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;").replace('"',"&quot;")

rows = ""
for r in results:
    rc  = r.get("result_class", "N/A")
    cov = r.get("coverage_score", 0)
    rt  = round(r.get("response_time", 0), 2)
    q   = esc(r["question"][:60])
    color = {"TP":"#22c55e","FP":"#f59e0b","TN":"#3b82f6","FN":"#ef4444"}.get(rc, "#888")
    rows += (
        f"<tr><td>{esc(r['id'])}</td>"
        f"<td style='color:#71717a'>{esc(r['category'])}</td>"
        f"<td>{q}…</td>"
        f"<td><b style='color:{color}'>{rc}</b></td>"
        f"<td>{cov}%</td><td>{rt}s</td></tr>"
    )

cat_labels = json.dumps(list(cats.keys()))
cat_scores = json.dumps([cats[c]["avg_coverage"] for c in cats])

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>AMO AI Evaluation Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap" rel="stylesheet">
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{background:#09090b;color:#f4f4f5;font-family:Inter,sans-serif;padding:40px 24px}}
.wrap{{max-width:1200px;margin:0 auto}}
h1{{font-size:28px;font-weight:800;background:linear-gradient(90deg,#00e5a0,#4f8aff);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;margin-bottom:4px}}
p.sub{{color:#71717a;font-size:13px;margin-bottom:36px}}
.cards{{display:grid;grid-template-columns:repeat(4,1fr);gap:16px;margin-bottom:36px}}
.card{{background:#18181b;border:1px solid #27272a;border-radius:16px;padding:24px;text-align:center}}
.val{{font-size:32px;font-weight:800;color:#00e5a0}}
.lbl{{font-size:11px;color:#71717a;text-transform:uppercase;letter-spacing:1px;margin-top:6px}}
.grid2{{display:grid;grid-template-columns:1fr 1.4fr;gap:24px;margin-bottom:36px}}
.panel{{background:#18181b;border:1px solid #27272a;border-radius:16px;padding:24px}}
.ptitle{{font-size:12px;font-weight:700;color:#52525b;text-transform:uppercase;letter-spacing:2px;margin-bottom:20px}}
.matrix{{display:grid;grid-template-columns:1fr 1fr;gap:12px}}
.mc{{border-radius:12px;padding:24px;text-align:center;font-size:28px;font-weight:800}}
.tp{{background:rgba(34,197,94,.1);color:#22c55e;border:1px solid rgba(34,197,94,.2)}}
.fp{{background:rgba(245,158,11,.1);color:#f59e0b;border:1px solid rgba(245,158,11,.2)}}
.tn{{background:rgba(59,130,246,.1);color:#3b82f6;border:1px solid rgba(59,130,246,.2)}}
.fn{{background:rgba(239,68,68,.1);color:#ef4444;border:1px solid rgba(239,68,68,.2)}}
.mc span{{display:block;font-size:10px;font-weight:600;opacity:.6;margin-bottom:4px}}
table{{width:100%;border-collapse:collapse;background:#18181b;border:1px solid #27272a;
  border-radius:16px;overflow:hidden;margin-top:0}}
th{{background:#27272a;padding:12px 16px;text-align:left;font-size:10px;color:#71717a;
  text-transform:uppercase;letter-spacing:1px}}
td{{padding:11px 16px;font-size:12px;border-bottom:1px solid #1f1f22;vertical-align:top}}
tr:last-child td{{border:none}}
tr:hover td{{background:rgba(255,255,255,.02)}}
</style>
</head>
<body>
<div class="wrap">
  <h1>AMO AI Evaluation</h1>
  <p class="sub">Last run: {ts} &nbsp;|&nbsp; Total tests: {s["total"]}</p>

  <div class="cards">
    <div class="card"><div class="val">{m["f1_score"]}%</div><div class="lbl">F1 Score</div></div>
    <div class="card"><div class="val">{m["accuracy"]}%</div><div class="lbl">Accuracy</div></div>
    <div class="card"><div class="val" style="color:#4f8aff">{m["precision"]}%</div><div class="lbl">Precision</div></div>
    <div class="card"><div class="val" style="color:#f59e0b">{m["recall"]}%</div><div class="lbl">Recall</div></div>
  </div>

  <div class="grid2">
    <div class="panel">
      <div class="ptitle">Confusion Matrix</div>
      <div class="matrix">
        <div class="mc tp"><span>TP — Correct answers</span>{tp}</div>
        <div class="mc fp"><span>FP — Wrong language</span>{fp}</div>
        <div class="mc fn"><span>FN — Missed answers</span>{fn}</div>
        <div class="mc tn"><span>TN — OOS blocked</span>{tn}</div>
      </div>
    </div>
    <div class="panel">
      <div class="ptitle">Coverage by Category</div>
      <canvas id="cat" style="max-height:220px"></canvas>
    </div>
  </div>

  <div class="panel" style="overflow-x:auto">
    <div class="ptitle" style="margin-bottom:16px">Detailed Test Results</div>
    <table>
      <thead><tr>
        <th>ID</th><th>Category</th><th>Question</th>
        <th>Result</th><th>Coverage</th><th>Time</th>
      </tr></thead>
      <tbody>{rows}</tbody>
    </table>
  </div>
</div>

<script>
new Chart(document.getElementById("cat"), {{
  type: "bar",
  data: {{
    labels: {cat_labels},
    datasets: [{{
      label: "Avg Coverage %",
      data: {cat_scores},
      backgroundColor: "#4f8aff",
      borderRadius: 8
    }}]
  }},
  options: {{
    plugins: {{ legend: {{ display: false }} }},
    scales: {{
      y: {{ beginAtZero: true, max: 100, grid: {{ color: "#27272a" }}, ticks: {{ color: "#71717a" }} }},
      x: {{ grid: {{ display: false }}, ticks: {{ color: "#71717a", font: {{ size: 10 }} }} }}
    }}
  }}
}});
</script>
</body>
</html>
"""

with open("confusion_matrix_dashboard_live.html", "w", encoding="utf-8") as f:
    f.write(html)

print("Done! Open confusion_matrix_dashboard_live.html in your browser.")
