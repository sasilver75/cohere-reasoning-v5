import os
import numpy as np
import pandas as pd
from flask import Flask, render_template_string, request
from flask.json import provider

app = Flask(__name__)

# Load the CSV file
csv_path = "datasets/derived/interesting_problems_completed.csv"
if not os.path.exists(csv_path):
    print(f"Error: CSV file not found at {csv_path}")
    exit(1)

try:
    df = pd.read_csv(csv_path)
except Exception as e:
    print(f"Error reading CSV file: {e}")
    exit(1)

@app.route("/")
def index():
    page = request.args.get("page", 1, type=int)
    if page < 1 or page > len(df):
        page = 1

    row = df.iloc[page - 1]
    completion_data = {
        "row_id": int(row.get("row_id", 0)),
        "solution_id": int(row.get("solution_id", 0)),
        "completion_id": int(row.get("completion_id", 0)),
        "problem": str(row.get("problem", "N/A")),
        "solution": str(row.get("solution", "N/A")),
        "candidate_solution": str(row.get("candidate_solution", "N/A")),
        "candidate_verification_reasoning": str(row.get("candidate_verification_reasoning", "N/A")),
        "candidate_verification_result": bool(row.get("candidate_verification_result", False)),
        "prefix": str(row.get("prefix", "N/A")),
        "completion": str(row.get("completion", "N/A")),
        "completion_verification_result": bool(row.get("completion_verification_result", False)),
        "completion_verification_reasoning": str(row.get("completion_verification_reasoning", "N/A")),
    }

    return render_template_string(
        """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Math Problem Completion Viewer</title>
        <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
        <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
        <script>
            MathJax = {
                tex: {
                    inlineMath: [['$', '$'], ['\\(', '\\)']],
                    displayMath: [['$$', '$$'], ['\\[', '\\]']],
                    processEscapes: true,
                    processEnvironments: true
                },
                options: {
                    skipHtmlTags: ['script', 'noscript', 'style', 'textarea', 'pre']
                }
            };
        </script>
        <style>
            body { 
                font-family: Arial, sans-serif; 
                line-height: 1.6; 
                padding: 20px;
                max-width: 1800px;
                margin: 0 auto;
            }
            .header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 20px;
            }
            .completion { 
                border: 1px solid #ddd; 
                padding: 20px; 
                margin-bottom: 20px;
                display: flex;
                flex-wrap: wrap;
                justify-content: space-between;
            }
            .section {
                width: 48%;
                margin-bottom: 20px;
            }
            h2, h3 { color: #333; }
            .content-box { 
                background-color: #f4f4f4; 
                padding: 10px; 
                border-radius: 4px;
                margin-bottom: 15px;
                word-wrap: break-word;
                overflow-wrap: break-word;
                white-space: pre-wrap;
            }
            .verification-box {
                padding: 10px;
                border-radius: 4px;
                margin-bottom: 15px;
                font-weight: bold;
            }
            .verification-true {
                background-color: lightgreen;
            }
            .verification-false {
                background-color: lightcoral;
            }
            .navigation { 
                display: flex; 
                gap: 10px;
            }
            .navigation a { 
                text-decoration: none; 
                color: #333; 
                padding: 10px; 
                border: 1px solid #ddd;
            }
        </style>
    </head>
    <body>
        <div class="header">
            <h1>Math Problem Completion Viewer ({{ page }}/{{ total_pages }})</h1>
            <div class="navigation">
                {% if page > 1 %}
                    <a href="{{ url_for('index', page=page-1) }}">Previous</a>
                {% endif %}
                {% if page < total_pages %}
                    <a href="{{ url_for('index', page=page+1) }}">Next</a>
                {% endif %}
            </div>
        </div>
        <h2>Row ID: {{ completion_data.row_id }} | Solution ID: {{ completion_data.solution_id }} | Completion ID: {{ completion_data.completion_id }}</h2>
        <div class="completion">
            <div class="section">
                <h2>Problem:</h2>
                <div class="content-box">{{ completion_data.problem }}</div>
                
                <h2>Ground Truth Solution:</h2>
                <div class="content-box">{{ completion_data.solution }}</div>

                <h2>Candidate Solution:</h2>
                <div class="content-box">{{ completion_data.candidate_solution }}</div>

                <h2>Candidate Verification Result:</h2>
                <div class="verification-box verification-{{ completion_data.candidate_verification_result|lower }}">
                    {{ completion_data.candidate_verification_result }}
                </div>

                <h2>Candidate Verification Reasoning:</h2>
                <div class="content-box">{{ completion_data.candidate_verification_reasoning }}</div>
            </div>
            <div class="section">
                <h2>Prefix:</h2>
                <div class="content-box">{{ completion_data.prefix }}</div>

                <h2>Completion:</h2>
                <div class="content-box">{{ completion_data.completion }}</div>
                
                <h2>Completion Verification Result:</h2>
                <div class="verification-box verification-{{ completion_data.completion_verification_result|lower }}">
                    {{ completion_data.completion_verification_result }}
                </div>
                
                <h2>Completion Verification Reasoning:</h2>
                <div class="content-box">{{ completion_data.completion_verification_reasoning }}</div>
            </div>
        </div>
    </body>
    </html>
    """,
        completion_data=completion_data,
        page=page,
        total_pages=len(df),
    )

# Custom JSON encoder to handle numpy types
class NumpyEncoder(provider.DefaultJSONProvider):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

app.json = NumpyEncoder(app)

if __name__ == "__main__":
    print(f"Starting server. CSV file path: {csv_path}")
    app.run(debug=True, host="localhost", port=5000) 