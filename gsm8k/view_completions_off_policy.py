import os
import pandas as pd
from flask import Flask, render_template_string, request
from flask.json import provider
import matplotlib.pyplot as plt
import io
import base64
import numpy as np

app = Flask(__name__)

# Load the CSV file
csv_path = f"gsm8k/datasets/gsm8k_completions_off_policy.csv"
if not os.path.exists(csv_path):
    print(f"Error: CSV file not found at {csv_path}")
    exit(1)

try:
    df = pd.read_csv(csv_path)
except Exception as e:
    print(f"Error reading CSV file: {e}")
    exit(1)

def create_verification_plot():
    # Calculate rates by model
    stats = df.groupby('completion_model').agg({
        'problem_id': 'count',  # total problems
        'perturbed_stub_lm_solution_verified': 'sum'  # sum of True values
    }).assign(
        verification_rate=lambda x: x['perturbed_stub_lm_solution_verified'] / x['problem_id']
    )

    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Set the width of each bar and positions of the bars
    x = np.arange(len(stats.index))
    
    # Create single bar for verification rate
    rects = ax.bar(x, stats['verification_rate'], 
                   label='Correct Answer Rate', color='#2196F3')

    # Customize the plot
    ax.set_title('Correct Answer Rate by Model')
    ax.set_xlabel('Model')
    ax.set_ylabel('Rate')
    ax.set_xticks(x)
    ax.set_xticklabels(stats.index, rotation=45, ha='right')
    ax.set_ylim(0, 1)  # Set y-axis limits from 0 to 1
    ax.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Add percentage labels on top of each bar
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.1%}',
                       xy=(rect.get_x() + rect.get_width()/2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom')

    autolabel(rects)
    plt.tight_layout()
    
    # Convert plot to base64 string
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight', dpi=120)
    img.seek(0)
    plt.close()
    
    return base64.b64encode(img.getvalue()).decode()

@app.route("/")
def stats():
    # Calculate statistics for each model
    model_stats = df.groupby(['completion_model', 'completion_model_provider']).agg({
        'problem_id': 'count',
        'perturbed_stub_lm_solution_verified': 'sum'  # sum of True values
    })
    
    # Calculate the rate manually
    model_stats['verified_rate'] = model_stats['perturbed_stub_lm_solution_verified'] / model_stats['problem_id']
    
    # Rename columns for clarity
    model_stats.columns = ['total_problems', 'verified_count', 'verified_rate']
    
    # Create the plot
    plot_url = create_verification_plot()
    
    return render_template_string("""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>GSM8K Completions Overview</title>
            <style>
                body { 
                    font-family: Arial, sans-serif; 
                    line-height: 1.6; 
                    padding: 20px;
                    max-width: 1200px;
                    margin: 0 auto;
                }
                .header {
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    margin-bottom: 20px;
                }
                .header-buttons {
                    display: flex;
                    gap: 10px;
                }
                .header-button {
                    text-decoration: none;
                    color: white;
                    background-color: #007bff;
                    padding: 10px 20px;
                    border-radius: 4px;
                    transition: background-color 0.2s;
                }
                .header-button:hover {
                    background-color: #0056b3;
                }
                .stats-container {
                    background-color: #f4f4f4;
                    padding: 20px;
                    border-radius: 8px;
                    margin-bottom: 20px;
                }
                .model-stats-table {
                    width: 100%;
                    border-collapse: collapse;
                    margin-top: 20px;
                    background-color: white;
                }
                .model-stats-table th,
                .model-stats-table td {
                    padding: 12px;
                    text-align: left;
                    border-bottom: 1px solid #ddd;
                }
                .model-stats-table th {
                    background-color: #f8f9fa;
                }
                .model-stats-table tr:hover {
                    background-color: #f5f5f5;
                }
                .chart-container {
                    margin: 30px auto;
                    text-align: center;
                }
                .model-link {
                    text-decoration: none;
                    color: #007bff;
                }
                .model-link:hover {
                    text-decoration: underline;
                }
                .title-section {
                    flex: 1;
                }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>GSM8K Completions Overview</h1>
                <div class="header-buttons">
                    <a href="{{ url_for('view_problems') }}" class="header-button">View Problems</a>
                </div>
            </div>
            
            <div class="stats-container">
                <h2>Model Performance Summary</h2>
                <table class="model-stats-table">
                    <thead>
                        <tr>
                            <th>Model</th>
                            <th>Provider</th>
                            <th>Total Problems</th>
                            <th>Verified Solutions</th>
                            <th>Verification Rate</th>
                            <th>View Results</th>
                        </tr>
                    </thead>
                    <tbody>
                        {% for (model, provider), stats in model_stats.iterrows() %}
                        <tr>
                            <td>{{ model }}</td>
                            <td>{{ provider }}</td>
                            <td>{{ stats.total_problems }}</td>
                            <td>{{ stats.verified_count }}</td>
                            <td>{{ "%.1f%%"|format(stats.verified_rate * 100) }}</td>
                            <td>
                                <a href="{{ url_for('view_completions', model=model) }}" class="model-link">
                                    View Completions
                                </a>
                            </td>
                        </tr>
                        {% endfor %}
                    </tbody>
                </table>
            </div>

            <div class="chart-container">
                <img src="data:image/png;base64,{{ plot_url }}" alt="Verification Rates">
            </div>
        </body>
        </html>
    """, model_stats=model_stats, plot_url=plot_url)

@app.route("/completions")
def view_completions():
    model = request.args.get("model")
    if not model:
        return "Model parameter is required", 400
    
    # Filter dataframe for selected model
    model_df = df[df['completion_model'] == model].copy()
    
    # Get problem_id from query parameters if provided
    problem_id = request.args.get("problem_id", type=int)
    if problem_id:
        try:
            page = model_df[model_df['problem_id'] == problem_id].index[0] + 1
        except IndexError:
            page = 1
    else:
        page = request.args.get("page", 1, type=int)
    
    if page < 1 or page > len(model_df):
        page = 1

    # Get current row
    row = model_df.iloc[page - 1]
    
    completion_data = {
        "model": model,
        "provider": str(row.get("provider", "N/A")),
        "problem_id": int(row.get("problem_id", 0)),
        "problem": str(row.get("problem", "N/A")),
        "answer": str(row.get("answer", "N/A")),
        "stub": str(row.get("stub", "N/A")),
        "perturbed_stub_lm": str(row.get("perturbed_stub_lm", "N/A")),
        "perturbed_stub_lm_completion": str(row.get("perturbed_stub_lm_completion", "N/A")),
        "verified": bool(row.get("perturbed_stub_lm_solution_verified", False)),
        "correction_detected": bool(row.get("perturbed_stub_lm_solution_correction_detected", False))
    }

    return render_template_string("""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>GSM8K Completions Viewer</title>
            <style>
                body { 
                    font-family: Arial, sans-serif; 
                    line-height: 1.6; 
                    padding: 20px;
                    max-width: 1200px;
                    margin: 0 auto;
                }
                .header {
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    margin-bottom: 20px;
                }
                .content-section {
                    background-color: #f4f4f4;
                    padding: 20px;
                    border-radius: 4px;
                    margin-bottom: 20px;
                }
                .content-box {
                    background-color: white;
                    padding: 15px;
                    border-radius: 4px;
                    margin-top: 10px;
                    white-space: pre-wrap;
                    border: 2px solid #007bff;
                }
                .content-box-secondary {
                    background-color: #fafafa;
                    padding: 15px;
                    border-radius: 4px;
                    margin-top: 10px;
                    white-space: pre-wrap;
                    border: 1px solid #e0e0e0;
                }
                .verification-box {
                    padding: 10px;
                    border-radius: 4px;
                    margin-top: 10px;
                    font-weight: bold;
                }
                .verification-true {
                    background-color: lightgreen;
                }
                .verification-false {
                    background-color: lightcoral;
                }
                .nav-button {
                    text-decoration: none;
                    color: white;
                    background-color: #007bff;
                    padding: 10px 20px;
                    border-radius: 4px;
                    transition: background-color 0.2s;
                }
                .nav-button.overview {
                    background-color: #ff9800;
                }
                .nav-button:hover:not(.disabled) {
                    background-color: #0056b3;
                }
                .nav-button.overview:hover {
                    background-color: #f57c00;
                }
                .nav-button.disabled {
                    background-color: #6c757d;
                    cursor: not-allowed;
                    pointer-events: none;
                    opacity: 0.65;
                }
            </style>
        </head>
        <body>
            <div class="header">
                <div class="title-section">
                    <h1>GSM8K Completion ({{ page }}/{{ total_pages }})</h1>
                    <h3 style="margin-top: 0; color: #666;">Model: {{ completion_data.model }}</h3>
                </div>
                <div class="navigation">
                    <a href="{{ url_for('stats') }}" class="nav-button overview">Back to Overview</a>
                    <a href="{{ url_for('view_completions', model=model, page=page-1) }}" 
                       class="nav-button {% if page <= 1 %}disabled{% endif %}">Previous</a>
                    <a href="{{ url_for('view_completions', model=model, page=page+1) }}" 
                       class="nav-button {% if page >= total_pages %}disabled{% endif %}">Next</a>
                </div>
            </div>

            <div class="content-section">
                <h2>Problem ID: {{ completion_data.problem_id }}</h2>
                
                <h3>Problem:</h3>
                <div class="content-box">{{ completion_data.problem }}</div>
                
                <h3>Answer:</h3>
                <div class="content-box-secondary">{{ completion_data.answer }}</div>
                
                <h3>Original Stub:</h3>
                <div class="content-box-secondary">{{ completion_data.stub }}</div>
                
                <h3>Perturbed Stub:</h3>
                <div class="content-box">{{ completion_data.perturbed_stub_lm }}</div>
                
                <h3>Completion:</h3>
                <div class="content-box">{{ completion_data.perturbed_stub_lm_completion }}</div>
                
                <h3>Verification Result:</h3>
                <div class="verification-box verification-{{ completion_data.verified|lower }}">
                    Verified: {{ completion_data.verified }}
                </div>
            </div>
        </body>
        </html>
    """, completion_data=completion_data, page=page, total_pages=len(model_df), model=model)

@app.route("/problems")
def view_problems():
    page = request.args.get("page", 1, type=int)
    per_page = 10  # Number of problems per page
    
    # Get total number of unique problems
    unique_problems = df.drop_duplicates('problem_id')
    total_problems = len(unique_problems)
    total_pages = (total_problems + per_page - 1) // per_page
    
    # Ensure page is within valid range
    page = max(1, min(page, total_pages))
    
    # Get problems for current page
    start_idx = (page - 1) * per_page
    end_idx = start_idx + per_page
    current_problems = unique_problems.iloc[start_idx:end_idx]
    
    return render_template_string("""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>GSM8K Problems Overview</title>
            <style>
                body { 
                    font-family: Arial, sans-serif; 
                    line-height: 1.6; 
                    padding: 20px;
                    max-width: 1200px;
                    margin: 0 auto;
                }
                .header {
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    margin-bottom: 20px;
                }
                .problem-container {
                    background-color: #f4f4f4;
                    padding: 20px;
                    border-radius: 8px;
                    margin-bottom: 20px;
                }
                .problem-box {
                    background-color: white;
                    padding: 15px;
                    border-radius: 4px;
                    margin-bottom: 15px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }
                .section-title {
                    font-weight: bold;
                    color: #666;
                    margin-bottom: 5px;
                }
                .navigation {
                    display: flex;
                    justify-content: center;
                    gap: 10px;
                    margin-top: 20px;
                }
                .nav-button {
                    text-decoration: none;
                    color: white;
                    background-color: #007bff;
                    padding: 10px 20px;
                    border-radius: 4px;
                    transition: background-color 0.2s;
                }
                .nav-button.overview {
                    background-color: #ff9800;
                }
                .nav-button:hover:not(.disabled) {
                    background-color: #0056b3;
                }
                .nav-button.disabled {
                    background-color: #6c757d;
                    cursor: not-allowed;
                    pointer-events: none;
                    opacity: 0.65;
                }
                .pagination-info {
                    text-align: center;
                    margin-bottom: 20px;
                    color: #666;
                }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>GSM8K Problems Overview</h1>
                <a href="{{ url_for('stats') }}" class="nav-button overview">Back to Overview</a>
            </div>
            
            <div class="pagination-info">
                Page {{ page }} of {{ total_pages }} ({{ total_problems }} total problems)
            </div>

            {% for _, problem in current_problems.iterrows() %}
            <div class="problem-container">
                <h3>Problem ID: {{ problem.problem_id }}</h3>
                
                <div class="problem-box">
                    <div class="section-title">Problem:</div>
                    {{ problem.problem }}
                </div>
                
                <div class="problem-box">
                    <div class="section-title">Original Stub:</div>
                    {{ problem.stub }}
                </div>
                
                <div class="problem-box">
                    <div class="section-title">Perturbed Stub:</div>
                    {{ problem.perturbed_stub_lm }}
                </div>
            </div>
            {% endfor %}
            
            <div class="navigation">
                <a href="{{ url_for('view_problems', page=page-1) }}" 
                   class="nav-button {% if page <= 1 %}disabled{% endif %}">Previous</a>
                <a href="{{ url_for('view_problems', page=page+1) }}" 
                   class="nav-button {% if page >= total_pages %}disabled{% endif %}">Next</a>
            </div>
        </body>
        </html>
    """, page=page, total_pages=total_pages, total_problems=total_problems, 
        current_problems=current_problems)

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
