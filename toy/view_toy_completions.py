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
csv_path = "toy/toy_evaluate.csv"
if not os.path.exists(csv_path):
    print(f"Error: CSV file not found at {csv_path}")
    exit(1)

try:
    df = pd.read_csv(csv_path)
except Exception as e:
    print(f"Error reading CSV file: {e}")
    exit(1)

def create_recovery_plot():
    # Calculate rates by model
    stats = df.groupby('model').agg({
        'problem_id': 'count',  # total problems
        'verified': lambda x: x.sum(),  # number of verified solutions
        'correction_detected': lambda x: x.sum()  # number of corrections detected
    }).assign(
        recovery_rate=lambda x: x['verified'] / x['problem_id'],
        correction_rate=lambda x: x['correction_detected'] / x['problem_id']
    )

    # Set up the plot with a more constrained width
    fig, ax = plt.subplots(figsize=(10, 6))  # Reduced from 12 to 10
    
    # Set the width of each bar and positions of the bars
    width = 0.35
    x = np.arange(len(stats.index))
    
    # Create bars
    rects1 = ax.bar(x - width/2, stats['recovery_rate'], width, label='Recovery Rate', color='#2196F3')
    rects2 = ax.bar(x + width/2, stats['correction_rate'], width, label='Correction Rate', color='#FF9800')

    # Customize the plot
    ax.set_title('Recovery and Correction Rates by Model')
    ax.set_xlabel('Model')
    ax.set_ylabel('Rate')
    ax.set_xticks(x)
    ax.set_xticklabels(stats.index, rotation=45, ha='right')
    # Move legend to the top of the plot, outside the chart area
    ax.legend(bbox_to_anchor=(0.75, 1.2), loc='upper left', ncol=1)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add percentage labels on top of each bar
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.1%}',
                       xy=(rect.get_x() + rect.get_width()/2, height),
                       xytext=(0, 3),  # 3 points vertical offset
                       textcoords="offset points",
                       ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)

    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    # Convert plot to base64 string
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight', dpi=120)
    img.seek(0)
    plt.close()
    
    return base64.b64encode(img.getvalue()).decode()

def create_problem_difficulty_plot():
    # Pivot the data to get problems on x-axis and models as different lines
    pivot_data = df.pivot(index='problem_id', columns='model', values='verified')
    
    # Set up the plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Create a scatter plot for each model with slight vertical offsets to avoid overlap
    models = pivot_data.columns
    for i, model in enumerate(models):
        offset = i * 0.1  # Small vertical offset for each model
        # Convert boolean to 0/1 and add offset
        y_values = pivot_data[model].astype(int) + offset
        ax.scatter(pivot_data.index, y_values, label=model, alpha=0.6, marker='o')

    # Customize the plot
    ax.set_title('Problem Success Pattern by Model')
    ax.set_xlabel('Problem ID')
    ax.set_ylabel('Success (0=False, 1=True)')
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['Failed', 'Succeeded'])
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Move legend to the right side of the plot
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Adjust layout to prevent label cutoff
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
    model_stats = df.groupby(['model', 'provider']).agg({
        'problem_id': 'count',
        'verified': ['sum', 'mean'],
        'correction_detected': ['sum', 'mean']
    }).round(3)
    
    # Flatten column names
    model_stats.columns = ['total_problems', 'verified_count', 'verified_rate', 
                          'corrections_count', 'corrections_rate']
    
    # Create the recovery rate plot
    plot_url = create_recovery_plot()
    difficulty_plot_url = create_problem_difficulty_plot()
    
    return render_template_string("""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Toy Evaluation Results</title>
            <style>
                body { 
                    font-family: Arial, sans-serif; 
                    line-height: 1.6; 
                    padding: 20px;
                    max-width: 1200px;
                    margin: 0 auto;
                }
                .header {
                    margin-bottom: 20px;
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
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Toy Evaluation Results Overview</h1>
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
                            <th>Corrections Detected</th>
                            <th>Correction Rate</th>
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
                            <td>{{ stats.corrections_count }}</td>
                            <td>{{ "%.1f%%"|format(stats.corrections_rate * 100) }}</td>
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
                <img src="data:image/png;base64,{{ plot_url }}" alt="Recovery Rates by Model">
            </div>

            <div class="chart-container">
                <img src="data:image/png;base64,{{ difficulty_plot_url }}" alt="Problem Difficulty Pattern">
            </div>
        </body>
        </html>
    """, model_stats=model_stats, plot_url=plot_url, difficulty_plot_url=difficulty_plot_url)

@app.route("/completions")
def view_completions():
    model = request.args.get("model")
    if not model:
        return "Model parameter is required", 400
    
    # Filter dataframe for selected model
    model_df = df[df['model'] == model].copy()
    
    # Get page number from query parameters
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
        "solution": str(row.get("solution", "N/A")),
        "prefix": str(row.get("prefix", "N/A")),
        "completion": str(row.get("completion", "N/A")),
        "verified": bool(row.get("verified", False)),
        "correction_detected": bool(row.get("correction_detected", False))
    }

    return render_template_string("""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Model Completions Viewer</title>
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
                .navigation {
                    display: flex;
                    gap: 10px;  /* Add spacing between buttons */
                    align-items: center;
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
                    background-color: #ff9800;  /* Orange color for overview button */
                }
                .nav-button.overview:hover {
                    background-color: #f57c00;  /* Darker orange on hover */
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
                .title-section {
                    flex: 1;
                }
                .provider-info {
                    margin: 0;
                    color: #666;
                    font-size: 1em;
                }
            </style>
        </head>
        <body>
            <div class="header">
                <div class="title-section">
                    <h1>{{ completion_data.model }} Completions ({{ page }}/{{ total_pages }})</h1>
                    <h3 class="provider-info">Provider: {{ completion_data.provider }}</h3>
                </div>
                <div class="navigation">
                    <a href="{{ url_for('stats') }}" class="nav-button overview">Back to Overview</a>
                    <a href="{{ url_for('view_completions', model=completion_data.model, page=page-1) }}" 
                       class="nav-button {% if page <= 1 %}disabled{% endif %}">Previous</a>
                    <a href="{{ url_for('view_completions', model=completion_data.model, page=page+1) }}" 
                       class="nav-button {% if page >= total_pages %}disabled{% endif %}">Next</a>
                </div>
            </div>

            <div class="content-section">
                <h2>Problem ID: {{ completion_data.problem_id }}</h2>
                
                <h3>Problem:</h3>
                <div class="content-box">{{ completion_data.problem }}</div>
                
                <h3>Ground Truth Solution:</h3>
                <div class="content-box">{{ completion_data.solution }}</div>
                
                <h3>Prefix:</h3>
                <div class="content-box">{{ completion_data.prefix }}</div>
                
                <h3>Completion:</h3>
                <div class="content-box">{{ completion_data.completion }}</div>
                
                <h3>Verification Result:</h3>
                <div class="verification-box verification-{{ completion_data.verified|lower }}">
                    Verified: {{ completion_data.verified }}
                </div>
                
                <h3>Correction Detection:</h3>
                <div class="verification-box verification-{{ completion_data.correction_detected|lower }}">
                    Correction Detected: {{ completion_data.correction_detected }}
                </div>
            </div>
        </body>
        </html>
    """, completion_data=completion_data, page=page, total_pages=len(model_df))

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