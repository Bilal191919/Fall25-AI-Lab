from flask import Flask, request, jsonify, render_template_string, send_file
import pandas as pd
import pickle, os, csv
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_absolute_error

app = Flask(__name__)
MODEL_FILE = "mpg_model.pkl"
HISTORY_FILE = "prediction_history.csv"

if os.path.exists(MODEL_FILE):
    os.remove(MODEL_FILE)

# ---------- FRONTEND ----------
HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
  <title>🚗 Car MPG Predictor</title>
  <style>
    :root {
      --bg:#f7f7f7;--text:#222;--card:#fff;--border:#ccc;
      --button:#333;--button-text:#fff;--hover:#4CAF50;--table-alt:#f0f0f0;
    }
    body.dark {
      --bg:#121212;--text:#f0f0f0;--card:#1e1e1e;--border:#444;
      --button:#4CAF50;--button-text:#fff;--hover:#5FD068;--table-alt:#1a1a1a;
    }
    body {font-family:'Segoe UI',sans-serif;background:var(--bg);color:var(--text);
          margin:0;padding:40px;transition:background .3s,color .3s;}
    .container {max-width:900px;margin:auto;background:var(--card);padding:30px 40px;
                border-radius:12px;box-shadow:0 2px 10px rgba(0,0,0,0.15);}
    h2{text-align:center;margin-bottom:30px;}
    select,input,button {
      margin:8px 0;padding:8px;width:230px;border:1px solid var(--border);
      border-radius:6px;background:var(--card);color:var(--text);
    }
    button {background:var(--button);color:var(--button-text);cursor:pointer;
            font-weight:bold;border:none;transition:all .2s ease;}
    button:hover {background:var(--hover);box-shadow:0 0 10px var(--hover);}
    .metric-box {background:var(--card);padding:10px;margin:15px 0;border-radius:8px;
                 border:1px solid var(--border);width:320px;}
    table {width:100%;border-collapse:collapse;margin-top:15px;border-radius:8px;overflow:hidden;}
    th,td {border:1px solid var(--border);padding:10px;text-align:center;}
    th {background-color:var(--button);color:var(--button-text);}
    tr:nth-child(even){background-color:var(--table-alt);}
    #themeToggle {float:right;background:none;border:none;font-size:18px;cursor:pointer;color:var(--text);}
    .section {margin-top:40px;}
    #resultCard {
      margin-top:15px;background:var(--hover);color:white;padding:15px;
      border-radius:10px;font-size:1.3rem;font-weight:bold;display:none;
      text-align:center;box-shadow:0 0 10px var(--hover);
    }
    #spinner {display:none;font-size:16px;margin-top:10px;color:var(--hover);}
  </style>
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
</head>
<body>
<div class="container">
  <button id="themeToggle" onclick="toggleTheme()">🌙</button>
  <h2>🚗 Car MPG Predictor</h2>

  <div class="section">
    <h3>📁 Train Model</h3>
    <input type="file" id="fileInput" accept=".csv">
    <button onclick="uploadFile()">Train Model</button>
    <div id="spinner">⏳ Training...</div>
    <p id="trainStatus"></p>
    <div class="metric-box">
      <strong>📊 Model Performance:</strong>
      <p id="accuracyMetrics">No model trained yet.</p>
    </div>
  </div>

  <div class="section">
    <h3>🔮 Predict MPG</h3>
    <p id="modelStatus"></p>
    <form id="predictForm">
      <label title="Select number of engine cylinders">Cylinders:</label><br>
      <select id="cylinders"></select><br>
      <label title="Select engine displacement in cubic inches">Displacement:</label><br>
      <select id="displacement"></select><br>
      <label title="Select the car model year">Year:</label><br>
      <select id="year"></select><br>
      <label title="Select the car brand/make">Make:</label><br>
      <select id="make"></select><br>
      <label title="Select fuel type used by the car">Fuel Type:</label><br>
      <select id="fuel_type"></select><br>
      <label title="Select transmission type">Transmission:</label><br>
      <select id="transmission"></select><br>
      <button type="button" onclick="predict()">Predict MPG</button>
    </form>
    <div id="resultCard"></div>
  </div>

  <div class="section">
    <h3>📊 Feature Importance</h3>
    <canvas id="featureChart" width="600" height="300"></canvas>
  </div>

  <div class="section">
    <h3>📜 Prediction History</h3>
    <button onclick="loadHistory()">Refresh History</button>
    <button onclick="downloadHistory()">⬇️ Download as CSV</button>
    <div id="historyTable"></div>
  </div>
</div>

<script>
const DEFAULT_MAKES=["Toyota","Honda","Ford","Chevrolet","Nissan","Other"];
const DEFAULT_FUELS=["gasoline","diesel","hybrid","electric","other"];
const DEFAULT_TRANS=["automatic","manual","cvt","other"];
let chart=null;

function toggleTheme(){
  document.body.classList.toggle("dark");
  const dark=document.body.classList.contains("dark");
  document.getElementById("themeToggle").textContent=dark?"☀️":"🌙";
  localStorage.setItem("theme",dark?"dark":"light");
}

window.onload=()=>{
  const t=localStorage.getItem("theme");
  if(t==="dark"){document.body.classList.add("dark");document.getElementById("themeToggle").textContent="☀️";}
  fillSelect("cylinders",1,50);
  fillSelect("displacement",1,1000);
  fillSelect("year",1980,2025);
  loadCategoricalOptions();loadMetrics();loadHistory();loadFeatureChart();
};

function fillSelect(id,start,end){
  const el=document.getElementById(id);el.innerHTML="";
  for(let i=start;i<=end;i++){const o=document.createElement("option");o.value=i;o.text=i;el.appendChild(o);}
}

async function loadCategoricalOptions(){
  try{
    const res=await fetch("/options");const data=await res.json();
    if(data.error){populateCategorical(DEFAULT_MAKES,DEFAULT_FUELS,DEFAULT_TRANS);
      document.getElementById("modelStatus").innerText="⚠️ Model not trained yet — using default lists.";
    }else{
      populateCategorical(data.make||DEFAULT_MAKES,data.fuel_type||DEFAULT_FUELS,data.transmission||DEFAULT_TRANS);
      document.getElementById("modelStatus").innerText="✅ Model options loaded.";
    }
  }catch(e){
    populateCategorical(DEFAULT_MAKES,DEFAULT_FUELS,DEFAULT_TRANS);
    document.getElementById("modelStatus").innerText="⚠️ Could not load model options — using default lists.";
  }
}

function populateCategorical(makes,fuels,trans){
  ["make","fuel_type","transmission"].forEach((id,i)=>{
    const el=document.getElementById(id);el.innerHTML="";
    const vals=[makes,fuels,trans][i];
    vals.forEach(v=>{const o=document.createElement("option");o.value=v;o.text=v;el.appendChild(o);});
  });
}

async function uploadFile(){
  const f=document.getElementById("fileInput");
  if(!f.files.length){alert("Please select a CSV file.");return;}
  const s=document.getElementById("spinner");s.style.display="inline-block";
  const form=new FormData();form.append("file",f.files[0]);
  const res=await (await fetch("/train",{method:"POST",body:form})).json();
  s.style.display="none";document.getElementById("trainStatus").innerText=res.message||res.error;
  await loadCategoricalOptions();await loadMetrics();await loadFeatureChart();
}

async function loadMetrics(){
  const res=await fetch("/metrics");const d=await res.json();
  const el=document.getElementById("accuracyMetrics");
  if(!d||!d.r2)el.innerHTML="No model trained yet.";
  else el.innerHTML=`R² Score:<b>${d.r2}</b><br>MAE:<b>${d.mae}</b>`;
}

async function predict(){
  const card=document.getElementById("resultCard");
  card.style.display="block";card.innerText="⏳ Predicting...";
  const p={
    cylinders:+document.getElementById("cylinders").value,
    displacement:+document.getElementById("displacement").value,
    year:+document.getElementById("year").value,
    make:document.getElementById("make").value,
    fuel_type:document.getElementById("fuel_type").value,
    transmission:document.getElementById("transmission").value
  };
  const r=await (await fetch("/predict",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(p)})).json();
  card.innerText=r.prediction!==undefined?`✅ Predicted MPG: ${r.prediction}`:`⚠️ ${r.error}`;
  loadHistory();
}

async function loadFeatureChart(){
  const r=await fetch("/feature_importance");const d=await r.json();
  if(d.error)return;
  const ctx=document.getElementById("featureChart").getContext("2d");
  if(chart)chart.destroy();
  chart=new Chart(ctx,{type:"bar",data:{labels:d.features,datasets:[{label:"Importance",data:d.importances,backgroundColor:"#4CAF50"}]},options:{scales:{y:{beginAtZero:true}}}});
}

async function loadHistory(){
  const r=await fetch("/history");const d=await r.json();
  const div=document.getElementById("historyTable");
  if(!d.length){div.innerHTML="<p>No prediction history yet.</p>";return;}
  let t="<table><tr>"+Object.keys(d[0]).map(k=>`<th>${k}</th>`).join("")+"</tr>";
  d.forEach(r=>t+="<tr>"+Object.values(r).map(v=>`<td>${v}</td>`).join("")+"</tr>");
  div.innerHTML=t+"</table>";
}

function downloadHistory(){window.location.href="/download_history";}
</script>
</body>
</html>
"""

# ---------- BACKEND ----------
@app.route("/")
def home(): return render_template_string(HTML_PAGE)

@app.route("/options")
def options():
    if not os.path.exists(MODEL_FILE): return jsonify({"error": "Model not trained yet."})
    m = pickle.load(open(MODEL_FILE, "rb"))
    return jsonify({c: list(le.classes_) for c, le in m["encoders"].items()})

@app.route("/metrics")
def metrics():
    if not os.path.exists(MODEL_FILE): return jsonify({"r2": None, "mae": None})
    m = pickle.load(open(MODEL_FILE, "rb"))
    return jsonify(m.get("metrics", {"r2": None, "mae": None}))

@app.route("/feature_importance")
def feature_importance():
    if not os.path.exists(MODEL_FILE): return jsonify({"error": "Model not trained yet"})
    m = pickle.load(open(MODEL_FILE, "rb"))
    return jsonify({"features": m["features"], "importances": list(m["model"].feature_importances_)})

@app.route("/train", methods=["POST"])
def train():
    file=request.files.get("file")
    if not file: return jsonify({"error":"Please upload a CSV file"}),400
    df=pd.read_csv(file); df.columns=[c.lower().strip() for c in df.columns]
    if "combination_mpg" not in df.columns:
        if {"city_mpg","highway_mpg"}.issubset(df.columns):
            df["combination_mpg"]=(df["city_mpg"]+df["highway_mpg"])/2
        else:
            return jsonify({"error":"Dataset must include MPG columns."}),400
    features=["cylinders","displacement","year"]
    cats=[f for f in ["make","fuel_type","transmission"] if f in df.columns]
    features+=cats; df=df.dropna(subset=features+["combination_mpg"])
    enc={}; 
    for c in cats:
        le=LabelEncoder(); df[c]=le.fit_transform(df[c]); enc[c]=le
    X,y=df[features],df["combination_mpg"]
    Xt,Xv,yt,yv=train_test_split(X,y,test_size=0.2,random_state=42)
    model=RandomForestRegressor().fit(Xt,yt)
    yp=model.predict(Xv)
    metrics={"r2":round(r2_score(yv,yp),3),"mae":round(mean_absolute_error(yv,yp),3)}
    pickle.dump({"model":model,"encoders":enc,"features":features,"metrics":metrics},open(MODEL_FILE,"wb"))
    return jsonify({"message":"✅ Model trained successfully!","metrics":metrics})

@app.route("/predict", methods=["POST"])
def predict():
    if not os.path.exists(MODEL_FILE): return jsonify({"error":"Model not trained yet."}),400
    data=request.get_json(); m=pickle.load(open(MODEL_FILE,"rb"))
    model,enc,features=m["model"],m["encoders"],m["features"]
    row=[]
    for f in features:
        v=data.get(f)
        if f in ["year","cylinders","displacement"]: v=float(v)
        elif f in enc:
            le=enc[f]
            if v not in le.classes_: v=le.classes_[0]
            v=le.transform([v])[0]
        row.append(v)
    p=round(float(model.predict([row])[0]),2)
    save_history(data,p)
    return jsonify({"prediction":p})

@app.route("/history")
def history():
    if not os.path.exists(HISTORY_FILE): return jsonify([])
    return jsonify(pd.read_csv(HISTORY_FILE).to_dict(orient="records"))

@app.route("/download_history")
def download_history():
    if not os.path.exists(HISTORY_FILE): return jsonify({"error":"No history found"}),404
    return send_file(HISTORY_FILE, as_attachment=True)

def save_history(inp,pred):
    row={**inp,"predicted_mpg":pred}; exist=os.path.exists(HISTORY_FILE)
    with open(HISTORY_FILE,"a",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(row.keys()))
        if not exist: w.writeheader()
        w.writerow(row)

if __name__=="__main__":
    app.run(debug=True)






















