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
