"""Student performance prediction demo — Flask, HTML, CSS and JS in one file.

Start locally: python app.py
Production start command (Linux hosting): gunicorn app:app

IMPORTANT: pickle files can execute code when opened. Only use a trusted model file.
This particular model was trained with only 12 records and is for demonstration.
"""

import math
import os
import pickle
from pathlib import Path

import pandas as pd
from flask import Flask, jsonify, render_template_string, request


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "random_model.pkl"

# These names AND their order match the model's saved feature_names_in_.
FIELDS = [
    {
        "name": "Age", "label": "Student age", "hint": "Student's age in years",
        "placeholder": "e.g. 20", "min": 10, "max": 100, "step": "1",
        "unit": "years", "integer": True, "icon": "01",
    },
    {
        "name": "Gender", "label": "Gender code", "hint": "Enter the numeric code used when training",
        "placeholder": "e.g. 0", "min": 0, "max": 999, "step": "1",
        "unit": "code", "integer": True, "icon": "02",
    },
    {
        "name": "Department", "label": "Department code", "hint": "Enter the original encoded department number",
        "placeholder": "e.g. 1", "min": 0, "max": 999, "step": "1",
        "unit": "code", "integer": True, "icon": "03",
    },
    {
        "name": "Study_Hours_Per_Day", "label": "Daily study hours", "hint": "Average hours spent studying each day",
        "placeholder": "e.g. 4", "min": 0, "max": 24, "step": "0.1",
        "unit": "hrs/day", "integer": False, "icon": "04",
    },
    {
        "name": "Attendance_Percentage", "label": "Attendance", "hint": "Attendance percentage (0–100)",
        "placeholder": "e.g. 88", "min": 0, "max": 100, "step": "0.1",
        "unit": "%", "integer": False, "icon": "05",
    },
    {
        "name": "Assignments_Completed", "label": "Assignments completed", "hint": "Total number of assignments completed",
        "placeholder": "e.g. 8", "min": 0, "max": 1000, "step": "1",
        "unit": "total", "integer": True, "icon": "06",
    },
    {
        "name": "Midterm_Score", "label": "Midterm score", "hint": "Midterm exam score, out of 100",
        "placeholder": "e.g. 76", "min": 0, "max": 100, "step": "0.1",
        "unit": "/ 100", "integer": False, "icon": "07",
    },
    {
        "name": "Final_Score", "label": "Final exam score", "hint": "Final exam score, out of 100",
        "placeholder": "e.g. 82", "min": 0, "max": 100, "step": "0.1",
        "unit": "/ 100", "integer": False, "icon": "08",
    },
]

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024

if not MODEL_PATH.is_file():
    raise FileNotFoundError(
        f"Model file not found: {MODEL_PATH}. Place random_model.pkl alongside app.py."
    )

# Loading pickle is safe only when the model originates from a trusted source.
with MODEL_PATH.open("rb") as model_file:
    model = pickle.load(model_file)

expected_features = [field["name"] for field in FIELDS]
actual_features = list(getattr(model, "feature_names_in_", []))
if actual_features != expected_features:
    raise RuntimeError(
        "Model input features do not match this app. "
        f"Expected {expected_features}, found {actual_features}."
    )

model_classes = [str(label) for label in model.classes_]
number_of_trees = int(getattr(model, "n_estimators", 0))
number_of_training_samples = getattr(model, "_n_samples", None)


HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <meta name="theme-color" content="#0b1020">
  <meta name="description" content="A modern machine learning student performance prediction demo.">
  <title>EduPulse | Student Performance AI</title>
  <style>
    :root {
      color-scheme: dark;
      --bg:#090e1c; --panel:#111a2c; --panel-2:#151f32;
      --stroke:rgba(192,212,255,.13); --text:#eff4ff; --muted:#9baac5;
      --indigo:#817aff; --cyan:#56dcf4; --green:#52e6aa; --rose:#ff8aab;
      --radius:22px;
    }
    *{box-sizing:border-box}
    html{scroll-behavior:smooth}
    body{margin:0;min-height:100vh;background:var(--bg);color:var(--text);
      font:15px/1.55 Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
      -webkit-font-smoothing:antialiased;overflow-x:hidden}
    body::before{content:"";position:fixed;inset:0;pointer-events:none;z-index:-1;
      background:radial-gradient(ellipse 48% 40% at 6% 2%,rgba(76,72,205,.22),transparent 80%),
      radial-gradient(ellipse 40% 40% at 96% 89%,rgba(4,139,162,.15),transparent 80%)}
    .wrap{max-width:1240px;margin:auto;padding:24px 26px 46px}
    .nav{display:flex;align-items:center;justify-content:space-between;gap:15px;padding:8px 0 25px}
    .brand{display:flex;align-items:center;gap:12px;text-decoration:none;color:var(--text);font-size:21px;font-weight:800;letter-spacing:-.8px}
    .brand-mark{width:43px;height:43px;display:grid;place-items:center;border-radius:14px;
      background:linear-gradient(145deg,#8e7aff,#487de4);box-shadow:0 8px 24px rgba(102,106,252,.3)}
    .brand-mark svg{width:25px;height:25px;fill:none;stroke:white;stroke-width:1.8;stroke-linecap:round;stroke-linejoin:round}
    .brand span span{color:#8fd7fa}
    .nav-right{display:flex;align-items:center;gap:10px;flex-wrap:wrap}
    .tag,.status{border:1px solid var(--stroke);border-radius:999px;background:rgba(255,255,255,.035);
      color:#bcc9e1;font-size:12px;font-weight:600;padding:7px 13px}
    .status{display:flex;align-items:center;gap:8px}
    .dot{height:7px;width:7px;background:var(--green);border-radius:50%;box-shadow:0 0 0 4px rgba(82,230,170,.12)}
    .layout{display:grid;grid-template-columns:minmax(290px,.78fr) minmax(0,1.58fr);gap:22px;align-items:start}
    .side{position:sticky;top:20px}
    .hero{position:relative;overflow:hidden;padding:33px 29px;border:1px solid rgba(160,157,255,.24);
      border-radius:var(--radius);background:linear-gradient(145deg,rgba(62,66,143,.38),rgba(24,37,71,.92) 56%,rgba(15,49,67,.95));
      box-shadow:0 20px 80px rgba(0,0,0,.15)}
    .hero::after{content:"";position:absolute;width:240px;height:240px;border-radius:50%;right:-110px;top:-105px;
      border:46px solid rgba(128,116,255,.085);pointer-events:none}
    .eyebrow{display:inline-flex;align-items:center;gap:9px;color:#aeb7ff;font-size:11px;font-weight:800;
      text-transform:uppercase;letter-spacing:2px}
    .eyebrow i{width:22px;height:1px;background:#aea8ff;display:inline-block}
    h1{font-size:clamp(31px,3.2vw,46px);line-height:1.13;letter-spacing:-2px;margin:25px 0 17px;font-weight:800}
    h1 em{font-style:normal;background:linear-gradient(95deg,#af9dff,#6ee1f2);background-clip:text;-webkit-background-clip:text;color:transparent}
    .hero p{color:#bdc9de;margin:0;max-width:325px;font-size:14px;line-height:1.85}
    .orbit{height:210px;margin:23px auto 6px;position:relative;display:grid;place-items:center}
    .orbit .circle{position:absolute;border-radius:50%;border:1px solid rgba(173,190,250,.13)}
    .orbit .c1{width:198px;height:198px}.orbit .c2{width:147px;height:147px}.orbit .c3{width:97px;height:97px}
    .orbit .c1::before,.orbit .c2::before{content:"";width:9px;height:9px;position:absolute;background:#88b8ff;border-radius:50%;top:18px;left:23px;box-shadow:0 0 17px #8098ff}
    .orbit .c2::before{left:auto;right:5px;top:28px;background:#57e1d7}
    .orbit .core{width:86px;height:86px;border-radius:24px;display:grid;place-items:center;transform:rotate(-8deg);
      background:linear-gradient(145deg,rgba(117,102,255,.9),rgba(72,185,221,.78));box-shadow:0 0 75px rgba(83,140,255,.24)}
    .orbit .core svg{transform:rotate(8deg);width:42px;height:42px;stroke:white;stroke-width:1.6;fill:none;stroke-linecap:round;stroke-linejoin:round}
    .hero-foot{display:flex;gap:10px;align-items:center;color:#c4cee5;font-size:12px;border-top:1px solid var(--stroke);padding-top:20px}
    .hero-foot svg{width:17px;height:17px;fill:none;stroke:#80e2e0;stroke-width:2;flex-shrink:0}
    .info{margin-top:16px;background:rgba(18,29,50,.82);border:1px solid var(--stroke);border-radius:19px;padding:20px 22px}
    .info-title{font-size:12px;letter-spacing:.7px;text-transform:uppercase;color:#9aa9c5;font-weight:800;margin-bottom:16px}
    .mini-grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}
    .mini strong{font-size:26px;line-height:1;display:block;letter-spacing:-1px}
    .mini span{display:block;color:#9daecb;font-size:11px;margin-top:5px}
    .info small{display:block;color:#91a2be;line-height:1.7;margin-top:20px;border-top:1px solid var(--stroke);padding-top:14px}
    .main{background:rgba(17,26,44,.95);border:1px solid var(--stroke);border-radius:var(--radius);
      overflow:hidden;box-shadow:0 20px 80px rgba(0,0,0,.16)}
    .form-head{padding:27px 30px 23px;border-bottom:1px solid var(--stroke);display:flex;justify-content:space-between;gap:15px;align-items:center}
    .kicker{font-size:11px;font-weight:800;color:#8faff6;letter-spacing:2px;text-transform:uppercase;margin-bottom:5px}
    h2{font-size:23px;letter-spacing:-.6px;line-height:1.2;margin:0 0 7px;font-weight:750}
    .sub{color:#9fafc9;font-size:13px;margin:0}
    .section-index{font-size:12px;color:#9d9bff;font-weight:800;white-space:nowrap;padding:8px 11px;
      border-radius:9px;background:rgba(124,109,255,.11)}
    form{padding:26px 30px 30px}
    .section-caption{display:flex;align-items:center;gap:10px;margin:4px 0 17px;font-size:11px;font-weight:800;
      text-transform:uppercase;letter-spacing:1.65px;color:#8596b6}
    .section-caption::after{content:"";height:1px;flex:1;background:var(--stroke)}
    .inputs{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:19px 18px}
    .field{display:block;min-width:0}
    .field-label{display:flex;align-items:center;gap:9px;font-size:13px;font-weight:700;color:#e4ecfc;margin:0 0 8px}
    .field-label b{font-size:10px;display:grid;place-items:center;height:23px;width:23px;border-radius:7px;
      background:rgba(118,127,230,.12);color:#9baff3;font-weight:800}
    .input-wrap{position:relative}
    .input-wrap input{display:block;width:100%;padding:13px 65px 13px 15px;border-radius:12px;color:#f6f8ff;
      background:#0c1526;border:1px solid rgba(168,191,231,.20);outline:none;
      font-family:inherit;font-weight:600;font-size:15px;min-height:48px;transition:border-color .18s,box-shadow .18s}
    .input-wrap input::placeholder{color:#536683;font-size:13px;font-weight:400}
    .input-wrap input:hover{border-color:rgba(160,175,244,.37)}
    .input-wrap input:focus{border-color:#968eff;box-shadow:0 0 0 3px rgba(128,116,255,.16)}
    .input-wrap .unit{position:absolute;right:13px;top:50%;transform:translateY(-50%);color:#8a9ab6;font-size:11px;pointer-events:none}
    .field-hint{display:block;color:#8192b0;font-size:11px;margin-top:7px;line-height:1.4}
    .is-invalid input{border-color:#ed7794}
    .error{display:block;color:#ff9bb5;font-size:11px;margin-top:7px}
    .notice{display:flex;gap:12px;padding:14px 16px;border-radius:13px;background:rgba(76,142,255,.065);
      color:#aebedb;border:1px solid rgba(109,152,255,.14);margin:22px 0 0;font-size:12px;line-height:1.65}
    .notice svg{height:17px;width:17px;flex-shrink:0;stroke:#7ea8f9;fill:none;stroke-width:1.8;margin-top:1px}
    .error-summary{background:rgba(255,109,147,.08);color:#ffb4c5;border:1px solid rgba(255,109,147,.25);
      border-radius:12px;padding:12px 16px;font-size:13px;margin-bottom:18px}
    .actions{display:flex;align-items:center;gap:13px;margin-top:25px;flex-wrap:wrap}
    button{font:inherit;cursor:pointer}
    .primary{flex:1;min-width:205px;display:flex;align-items:center;justify-content:center;gap:11px;border:0;
      background:linear-gradient(110deg,#8479fa,#5d9bf0);color:#fff;border-radius:13px;padding:15px 20px;
      font-weight:800;font-size:14px;box-shadow:0 13px 30px rgba(95,108,244,.25);transition:transform .18s,filter .18s}
    .primary:hover{filter:brightness(1.11);transform:translateY(-1px)}
    .primary:disabled{opacity:.7;cursor:wait}
    .primary svg{width:17px;height:17px;stroke:currentColor;stroke-width:2;fill:none}
    .secondary{border:1px solid var(--stroke);background:transparent;color:#aebfda;border-radius:12px;
      padding:13px 16px;font-size:12px;font-weight:700;transition:background .16s}
    .secondary:hover{background:rgba(255,255,255,.06)}
    .result{border-top:1px solid var(--stroke);padding:27px 30px 30px;background:linear-gradient(140deg,rgba(40,62,86,.26),rgba(17,26,44,.6))}
    .result-row{display:flex;justify-content:space-between;gap:18px;align-items:center}
    .result-badge{display:inline-block;border:1px solid rgba(85,229,169,.23);background:rgba(85,229,169,.08);
      color:#79edba;border-radius:99px;padding:6px 13px;font-size:12px;font-weight:800;margin:5px 0 13px}
    .fail .result-badge{background:rgba(255,125,162,.085);color:#ffa3bd;border-color:rgba(255,125,162,.25)}
    .result h3{font-size:31px;line-height:1.13;margin:0 0 9px;letter-spacing:-1px}
    .result p{font-size:13px;color:#a2b2cd;margin:0;max-width:350px}
    .gauge{--score:50;--ring:var(--green);width:126px;height:126px;flex:none;border-radius:50%;display:grid;place-items:center;
      background:conic-gradient(var(--ring) calc(var(--score)*1%), rgba(174,190,218,.13) 0);
      box-shadow:0 0 35px rgba(82,230,170,.065)}
    .fail .gauge{--ring:var(--rose)}
    .gauge-inner{width:105px;height:105px;background:#111c2c;border-radius:50%;display:flex;flex-direction:column;align-items:center;justify-content:center}
    .gauge strong{font-size:27px;line-height:1.05;letter-spacing:-1px}
    .gauge span{font-size:10px;color:#97a7c3;margin-top:3px}
    .result-foot{padding-top:17px;margin-top:20px;border-top:1px solid var(--stroke);font-size:11px;color:#91a3bf}
    footer{display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:12px;
      font-size:11px;color:#7587a4;padding:25px 3px 0}
    footer strong{color:#a2acc7}
    @media(max-width:980px){.layout{grid-template-columns:1fr}.side{position:static}.hero{padding:26px;min-height:0}
      .orbit{height:130px;margin:15px auto 0}.orbit .c1{width:127px;height:127px}.orbit .c2{width:99px;height:99px}.orbit .c3{width:70px;height:70px}
      .orbit .core{width:62px;height:62px;border-radius:18px}.orbit .core svg{width:30px;height:30px}}
    @media(max-width:570px){.wrap{padding:12px 14px 30px}.nav{padding:10px 0 20px}
      .brand{font-size:19px}.nav-right .tag{display:none}.layout{gap:15px}.hero{padding:24px 21px}
      .hero h1{font-size:35px;margin:19px 0 12px}.form-head{padding:23px 20px}.form-head h2{font-size:21px}
      form{padding:22px 19px}.inputs{grid-template-columns:1fr;gap:17px}.actions{flex-direction:column}
      .actions button{width:100%}.result{padding:22px 18px}.result-row{align-items:flex-start}
      .gauge{width:98px;height:98px}.gauge-inner{width:82px;height:82px}.gauge strong{font-size:22px}
      .result h3{font-size:26px}.status{font-size:10px;padding:6px 9px}}
    @media(prefers-reduced-motion:reduce){*,*::before,*::after{scroll-behavior:auto!important;transition:none!important}}
  </style>
</head>
<body>
  <div class="wrap">
    <nav class="nav" aria-label="Top navigation">
      <a class="brand" href="/" aria-label="EduPulse home">
        <div class="brand-mark" aria-hidden="true"><svg viewBox="0 0 24 24"><path d="m2 9 10-5 10 5-10 5L2 9Z"/><path d="M6 11v6c4 3 8 3 12 0v-6"/><path d="M22 9v7"/></svg></div>
        <span>Edu<span>Pulse</span></span>
      </a>
      <div class="nav-right"><span class="tag">Student Analytics</span><span class="status"><span class="dot"></span>Model loaded</span></div>
    </nav>

    <div class="layout">
      <aside class="side">
        <section class="hero">
          <div class="eyebrow"><i></i> Machine learning demo</div>
          <h1>Insights that<br><em>shape learning.</em></h1>
          <p>Explore how your saved Random Forest model classifies student performance using academic and attendance inputs.</p>
          <div class="orbit" aria-hidden="true">
            <div class="circle c1"></div><div class="circle c2"></div><div class="circle c3"></div>
            <div class="core"><svg viewBox="0 0 48 48"><rect x="7" y="6" width="34" height="36" rx="8"/>
              <path d="M15 31V24m9 7V16m9 15v-10"/></svg></div>
          </div>
          <div class="hero-foot"><svg viewBox="0 0 24 24"><path d="M12 2 4 5v7c0 5 3 8 8 10 5-2 8-5 8-10V5L12 2Z"/><path d="m8.5 12 2.5 2.5 5-5"/></svg>
            Local inference · No external prediction API</div>
        </section>
        <section class="info">
          <div class="info-title">Model at a glance</div>
          <div class="mini-grid">
            <div class="mini"><strong>{{ tree_count }}</strong><span>Decision trees</span></div>
            <div class="mini"><strong>{{ fields|length|string|e }}</strong><span>Input features</span></div>
            <div class="mini"><strong>2</strong><span>Output classes</span></div>
            <div class="mini"><strong>RF</strong><span>Classification model</span></div>
          </div>
          <small>Experimental model{% if training_count %} · {{ training_count }} training samples{% endif %}. Results are for demonstration and are not validated academic decisions.</small>
        </section>
      </aside>

      <main class="main">
        <div class="form-head">
          <div><div class="kicker">Prediction workspace</div>
            <h2>Student performance</h2>
            <p class="sub">Enter a student's details to estimate the outcome.</p></div>
          <span class="section-index">8 inputs</span>
        </div>
        <form id="predictionForm" method="post" action="/" novalidate>
          {% if errors %}<div class="error-summary" role="alert">Please review the highlighted fields before predicting.</div>{% endif %}
          <div class="section-caption">Academic & personal details</div>
          <div class="inputs">
            {% for field in fields %}
            <label class="field {% if field.name in errors %}is-invalid{% endif %}">
              <span class="field-label"><b>{{ field.icon }}</b>{{ field.label }}</span>
              <span class="input-wrap">
                <input type="number" name="{{ field.name }}" id="{{ field.name }}" required
                  min="{{ field.min }}" max="{{ field.max }}" step="{{ field.step }}"
                  placeholder="{{ field.placeholder }}" value="{{ values.get(field.name,'') }}"
                  aria-describedby="hint-{{ field.name }}" {% if field.name in errors %}aria-invalid="true"{% endif %}>
                <span class="unit">{{ field.unit }}</span>
              </span>
              <span class="field-hint" id="hint-{{ field.name }}">{{ field.hint }}</span>
              {% if field.name in errors %}<span class="error">{{ errors[field.name] }}</span>{% endif %}
            </label>
            {% endfor %}
          </div>
          <div class="notice">
            <svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="10"/><path d="M12 11v6m0-10h.01"/></svg>
            <div><strong>Encoding note:</strong> Gender and Department were saved as numeric features. Use the <strong>same codes used during training</strong>; their label-to-number mapping was not stored in the model file.</div>
          </div>
          <div class="actions">
            <button class="primary" type="submit" id="predictButton">
              <svg viewBox="0 0 24 24"><path d="m5 12 4 4L19 6"/></svg>
              <span>Generate prediction</span>
            </button>
            <button class="secondary" type="button" id="sampleButton">Fill demo values</button>
            <button class="secondary" type="button" id="resetButton">Reset</button>
          </div>
        </form>

        {% if result %}
        <section id="prediction-result" class="result {{ 'fail' if result.label|lower == 'fail' else 'pass' }}" aria-live="polite">
          <div class="kicker">Prediction result</div>
          <div class="result-row">
            <div>
              <div class="result-badge">{{ 'Positive outcome' if result.label|lower == 'pass' else 'Attention recommended' }}</div>
              <h3>Predicted: {{ result.label }}</h3>
              <p>{{ 'The model predicts a passing result based on the provided inputs.' if result.label|lower == 'pass' else 'The model predicts a failing result based on the provided inputs.' }}</p>
            </div>
            {% if result.vote_percent is not none %}
            <div class="gauge" style="--score:{{ result.vote_percent }}" aria-label="Model vote share {{ result.vote_percent }} percent">
              <div class="gauge-inner"><strong>{{ result.vote_percent }}%</strong><span>tree vote*</span></div>
            </div>
            {% endif %}
          </div>
          <div class="result-foot">*Vote share supports this prediction; it is <strong>not</strong> a validated probability of success. This small training dataset cannot establish real-world accuracy.</div>
        </section>
        {% endif %}
      </main>
    </div>

    <footer><span>© EduPulse · Built with Flask + scikit-learn</span><span><strong>Educational prototype</strong> · Interpret results cautiously</span></footer>
  </div>
  <script>
    const form = document.getElementById('predictionForm');
    const sampleButton = document.getElementById('sampleButton');
    const resetButton = document.getElementById('resetButton');
    const predictButton = document.getElementById('predictButton');
    const sample = {
      Age: 20, Gender: 0, Department: 1, Study_Hours_Per_Day: 4,
      Attendance_Percentage: 88, Assignments_Completed: 8,
      Midterm_Score: 76, Final_Score: 82
    };
    sampleButton.addEventListener('click', () => {
      for (const [name, value] of Object.entries(sample)) {
        const element = form.elements.namedItem(name);
        if (element) element.value = value;
      }
    });
    resetButton.addEventListener('click', () => {
      form.reset();
      for (const input of form.querySelectorAll('input[type=number]')) input.value = '';
      window.location.href = '/';
    });
    form.addEventListener('submit', (event) => {
      if (!form.checkValidity()) {
        event.preventDefault(); form.reportValidity(); return;
      }
      predictButton.disabled = true;
      predictButton.querySelector('span').textContent = 'Analyzing student data...';
    });
    if (document.getElementById('prediction-result')) {
      document.getElementById('prediction-result').scrollIntoView({behavior:'smooth',block:'nearest'});
    }
  </script>
</body>
</html>"""


def validate_inputs(form_data):
    """Return valid model inputs, any per-field errors, and original form values."""
    data = {}
    errors = {}
    values = {}
    for field in FIELDS:
        name = field["name"]
        raw = (form_data.get(name) or "").strip()
        values[name] = raw
        if raw == "":
            errors[name] = "This field is required."
            continue
        try:
            parsed = float(raw)
        except (TypeError, ValueError):
            errors[name] = "Enter a valid number."
            continue
        if not math.isfinite(parsed):
            errors[name] = "Enter a finite number."
        elif not field["min"] <= parsed <= field["max"]:
            errors[name] = f"Enter a value from {field['min']} to {field['max']}."
        elif field["integer"] and not parsed.is_integer():
            errors[name] = "Enter a whole number."
        else:
            data[name] = int(parsed) if field["integer"] else parsed
    return data, errors, values


@app.route("/", methods=["GET", "POST"])
def index():
    errors = {}
    values = {}
    result = None

    if request.method == "POST":
        data, errors, values = validate_inputs(request.form)
        if not errors:
            try:
                # A DataFrame maintains the exact names and order seen during training.
                features = pd.DataFrame([[data[key] for key in expected_features]], columns=expected_features)
                label = str(model.predict(features)[0])
                vote = None
                if hasattr(model, "predict_proba"):
                    scores = model.predict_proba(features)[0]
                    vote = round(float(scores[model_classes.index(label)]) * 100, 1)
                result = {"label": label, "vote_percent": vote}
            except Exception:
                app.logger.exception("Model prediction error")
                errors["__prediction__"] = "Unable to generate the prediction. Check the model configuration."

    return render_template_string(
        HTML, fields=FIELDS, errors=errors, values=values, result=result,
        tree_count=number_of_trees, training_count=number_of_training_samples,
    )


@app.get("/health")
def health():
    return jsonify(status="ok", model="loaded", n_features=len(expected_features))


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=False)
