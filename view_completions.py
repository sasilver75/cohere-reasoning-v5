import os
import numpy as np
import pandas as pd
from flask import Flask, render_template_string, request
from flask.json import provider
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

# Load the CSV file
EXPERIMENT_NAME = "test-cohere"
csv_path = f"datasets/derived/{EXPERIMENT_NAME}/interesting_problems_completed.csv"
if not os.path.exists(csv_path):
    print(f"Error: CSV file not found at {csv_path}")
    exit(1)

try:
    df = pd.read_csv(csv_path)
except Exception as e:
    print(f"Error reading CSV file: {e}")
    exit(1)

def create_plot(correct_completions, incorrect_completions):
    plt.figure(figsize=(10, 6))
    bars = plt.bar(['Incorrect Completions', 'Correct Completions'], 
                  [incorrect_completions, correct_completions],
                  color=['lightcoral', 'lightgreen'])
    
    # Add value labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom')
    
    plt.title('Completion Results Distribution')
    plt.ylabel('Count')
    
    # Convert plot to base64 string
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    plt.close()
    
    return base64.b64encode(img.getvalue()).decode()

@app.route("/")
def stats():
    # Calculate overall statistics
    total_attempts = len(df)
    correct_completions = len(df[df['completion_verification_result'] == True])
    incorrect_completions = len(df[df['completion_verification_result'] == False])
    
    # Calculate per-problem recovery rates
    per_problem_stats = df.groupby('row_id').agg({
        'completion_verification_result': ['count', 'sum', 'mean'],  # sum gives us total True values
        'problem': 'first'  # Get the problem text for reference
    }).reset_index()
    
    # Rename columns for clarity
    per_problem_stats.columns = ['row_id', 'attempts', 'recoveries', 'recovery_rate', 'problem']
    # Sort by recovery rate descending
    per_problem_stats = per_problem_stats.sort_values('recovery_rate', ascending=False)
    # Convert recovery rate to percentage
    per_problem_stats['recovery_rate'] = per_problem_stats['recovery_rate'] * 100
    
    # Create the plot
    plot_url = create_plot(correct_completions, incorrect_completions)
    
    return render_template_string(
        """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Completion Statistics</title>
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
                .stats-container {
                    background-color: #f4f4f4;
                    padding: 20px;
                    border-radius: 8px;
                    margin-bottom: 20px;
                }
                .stat-box {
                    display: inline-block;
                    padding: 15px;
                    margin: 10px;
                    background-color: white;
                    border-radius: 4px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }
                .chart-container {
                    margin-top: 30px;
                    text-align: center;
                }
                .chart-container img {
                    max-width: 100%;
                    height: auto;
                }
                .navigation {
                    margin-top: 20px;
                    text-align: center;
                }
                .navigation a {
                    text-decoration: none;
                    color: white;
                    background-color: #007bff;
                    padding: 10px 20px;
                    border-radius: 4px;
                }
                .navigation a:hover {
                    background-color: #0056b3;
                }
                .problem-stats-container {
                    margin-top: 30px;
                    background-color: #f4f4f4;
                    padding: 20px;
                    border-radius: 8px;
                }
                .problem-stats-table {
                    width: 100%;
                    border-collapse: collapse;
                    margin-top: 10px;
                    background-color: white;
                }
                .problem-stats-table th,
                .problem-stats-table td {
                    padding: 12px;
                    text-align: left;
                    border-bottom: 1px solid #ddd;
                }
                .problem-stats-table th {
                    background-color: #f8f9fa;
                    font-weight: bold;
                    cursor: pointer;  /* Show pointer cursor on sortable headers */
                }
                .problem-stats-table th:hover {
                    background-color: #e9ecef;
                }
                .sort-arrow::after {
                    content: '↕';
                    margin-left: 5px;
                    opacity: 0.5;
                }
                .sort-arrow.asc::after {
                    content: '↑';
                    opacity: 1;
                }
                .sort-arrow.desc::after {
                    content: '↓';
                    opacity: 1;
                }
                .problem-stats-table tr:hover {
                    background-color: #f5f5f5;
                }
                .problem-text {
                    max-width: 500px;
                    overflow: hidden;
                    text-overflow: ellipsis;
                    white-space: nowrap;
                }
                .jump-form {
                    display: inline-flex;
                    align-items: center;
                    gap: 10px;
                    margin-left: 20px;
                }
                .jump-form input {
                    width: 80px;
                    padding: 5px;
                    border: 1px solid #ddd;
                    border-radius: 4px;
                }
                .jump-form button {
                    padding: 5px 10px;
                    background-color: #007bff;
                    color: white;
                    border: none;
                    border-radius: 4px;
                    cursor: pointer;
                }
                .jump-form button:hover {
                    background-color: #0056b3;
                }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Completion Statistics Overview</h1>
                <div style="display: flex; align-items: center; gap: 20px;">
                    <form class="jump-form" action="{{ url_for('view_completions') }}" method="get">
                        <label for="row_id">Jump to Row ID:</label>
                        <input type="number" id="row_id" name="row_id" min="1">
                        <button type="submit">Go</button>
                    </form>
                    <div class="navigation">
                        <a href="{{ url_for('view_completions', page=1) }}">View Individual Completions</a>
                    </div>
                </div>
            </div>
            
            <div class="stats-container">
                <div class="stat-box">
                    <h3>Total Attempts</h3>
                    <p>{{ total_attempts }}</p>
                </div>
                <div class="stat-box">
                    <h3>Incorrect Completions</h3>
                    <p>{{ incorrect_completions }} ({{ "%.1f"|format(incorrect_completions/total_attempts*100) }}%)</p>
                </div>
                <div class="stat-box">
                    <h3>Correct Completions</h3>
                    <p>{{ correct_completions }} ({{ "%.1f"|format(correct_completions/total_attempts*100) }}%)</p>
                </div>
            </div>

            <div class="chart-container">
                <img src="data:image/png;base64,{{ plot_url }}" alt="Completion Results Distribution">
            </div>

            <div class="problem-stats-container">
                <h2>Per-Problem Recovery Rates <span style="font-size: 0.8em; font-weight: normal; color: #666;">(click columns to sort)</span></h2>
                <table class="problem-stats-table">
                    <thead>
                        <tr>
                            <th onclick="sortTable(0)" class="sort-arrow">Row ID</th>
                            <th onclick="sortTable(1)">Problem</th>
                            <th onclick="sortTable(2)">Total Attempts</th>
                            <th onclick="sortTable(3)">Recoveries</th>
                            <th onclick="sortTable(4)" class="sort-arrow desc">Recovery Rate</th>
                        </tr>
                    </thead>
                    <tbody>
                        {% for _, row in problem_stats.iterrows() %}
                        <tr>
                            <td>{{ row['row_id'] }}</td>
                            <td class="problem-text" title="{{ row['problem'] }}">{{ row['problem'] }}</td>
                            <td>{{ row['attempts'] }}</td>
                            <td>{{ row['recoveries'] }}</td>
                            <td>{{ "%.1f"|format(row['recovery_rate']) }}</td>
                        </tr>
                        {% endfor %}
                    </tbody>
                </table>
            </div>

            <script>
                function sortTable(columnIndex) {
                    const table = document.querySelector('.problem-stats-table');
                    const tbody = table.querySelector('tbody');
                    const rows = Array.from(tbody.querySelectorAll('tr'));
                    const th = table.querySelectorAll('th')[columnIndex];
                    
                    // Remove sort arrows from all headers except current
                    table.querySelectorAll('th').forEach(header => {
                        if (header !== th) {
                            header.classList.remove('sort-arrow', 'asc', 'desc');
                        }
                    });
                    
                    // Toggle sort direction
                    const isAscending = th.classList.contains('asc');
                    th.classList.toggle('sort-arrow');
                    th.classList.toggle('asc', !isAscending);
                    th.classList.toggle('desc', isAscending);

                    // Sort rows
                    rows.sort((rowA, rowB) => {
                        const cellA = rowA.cells[columnIndex].textContent;
                        const cellB = rowB.cells[columnIndex].textContent;
                        
                        // Parse numbers for numeric columns
                        if (columnIndex === 0) {  // Row ID
                            return (parseInt(cellA) - parseInt(cellB)) * (isAscending ? 1 : -1);
                        } else if (columnIndex === 2) {  // Total Attempts
                            return (parseInt(cellA) - parseInt(cellB)) * (isAscending ? 1 : -1);
                        } else if (columnIndex === 3) {  // Recoveries
                            return (parseInt(cellA) - parseInt(cellB)) * (isAscending ? 1 : -1);
                        } else if (columnIndex === 4) {  // Recovery Rate
                            return (parseFloat(cellA) - parseFloat(cellB)) * (isAscending ? 1 : -1);
                        }
                        // Text comparison for Problem column
                        return cellA.localeCompare(cellB) * (isAscending ? 1 : -1);
                    });
                    
                    // Reorder the table
                    rows.forEach(row => tbody.appendChild(row));
                }
            </script>
        </body>
        </html>
        """,
        total_attempts=total_attempts,
        correct_completions=correct_completions,
        incorrect_completions=incorrect_completions,
        plot_url=plot_url,
        problem_stats=per_problem_stats
    )

@app.route("/completions")
def view_completions():
    # Get row_id from query parameters if provided
    row_id = request.args.get("row_id", type=int)
    if row_id:
        # Find the index of the row_id in the dataframe
        try:
            page = df[df['row_id'] == row_id].index[0] + 1
        except IndexError:
            page = 1
    else:
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
            .jump-form {
                display: inline-flex;
                align-items: center;
                gap: 10px;
                margin-left: 20px;
            }
            .jump-form input {
                width: 80px;
                padding: 5px;
                border: 1px solid #ddd;
                border-radius: 4px;
            }
            .jump-form button {
                padding: 5px 10px;
                background-color: #007bff;
                color: white;
                border: none;
                border-radius: 4px;
                cursor: pointer;
            }
            .jump-form button:hover {
                background-color: #0056b3;
            }
        </style>
    </head>
    <body>
        <div class="header">
            <h1>Math Problem Completion Viewer ({{ page }}/{{ total_pages }})</h1>
            <div style="display: flex; align-items: center; gap: 20px;">
                <form class="jump-form" action="{{ url_for('view_completions') }}" method="get">
                    <label for="row_id">Jump to Row ID:</label>
                    <input type="number" id="row_id" name="row_id" min="1">
                    <button type="submit">Go</button>
                </form>
                <div class="navigation">
                    <a href="{{ url_for('stats') }}">Back to Statistics</a>
                    {% if page > 1 %}
                        <a href="{{ url_for('view_completions', page=page-1) }}">Previous</a>
                    {% endif %}
                    {% if page < total_pages %}
                        <a href="{{ url_for('view_completions', page=page+1) }}">Next</a>
                    {% endif %}
                </div>
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