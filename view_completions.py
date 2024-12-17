import os
import numpy as np
import pandas as pd
from flask import Flask, render_template_string, request
from flask.json import provider
import matplotlib.pyplot as plt
import io
import base64
from utils import plot_recovery_figures

app = Flask(__name__)

# Load the CSV file
EXPERIMENT_NAME = "experiment-MATH-qwen2.5_70b-100-12_16_2024-new-verification"
csv_path = f"datasets/experiments/{EXPERIMENT_NAME}/interesting_problems_completed.csv"
if not os.path.exists(csv_path):
    print(f"Error: CSV file not found at {csv_path}")
    exit(1)

try:
    df = pd.read_csv(csv_path)
except Exception as e:
    print(f"Error reading CSV file: {e}")
    exit(1)

def create_plots(correct_completions, incorrect_completions, per_problem_stats):
    # Create the plots using matplotlib
    plt.figure(figsize=(24, 8))
    
    # Use the plot_recovery_figures function
    plot_recovery_figures(df)
    
    # Convert plot to base64 string
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight', dpi=120)
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
        'problem': 'first',  # Get the problem text for reference
        'row_id_success_rate': 'first'  # Get the problem difficulty
    }).reset_index()
    
    # Rename columns for clarity
    per_problem_stats.columns = ['row_id', 'attempts', 'recoveries', 'recovery_rate', 'problem', 'difficulty']
    # Sort by recovery rate descending
    per_problem_stats = per_problem_stats.sort_values('recovery_rate', ascending=False)
    # Convert recovery rate to percentage
    per_problem_stats['recovery_rate'] = per_problem_stats['recovery_rate'] * 100
    # Convert difficulty to percentage
    per_problem_stats['difficulty'] = per_problem_stats['difficulty'] * 100
    
    # Calculate min and max difficulty for normalization
    min_difficulty = per_problem_stats['difficulty'].min()
    max_difficulty = per_problem_stats['difficulty'].max()
    
    # Add normalized difficulty column (0 to 100)
    per_problem_stats['normalized_difficulty'] = (
        (per_problem_stats['difficulty'] - min_difficulty) / 
        (max_difficulty - min_difficulty) * 100
    )
    
    # Create the plots
    plot_url = create_plots(correct_completions, incorrect_completions, per_problem_stats)
    
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
                    margin: 30px auto;
                    text-align: center;
                    max-width: 90%;
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
                    display: inline-block;
                    min-width: 200px;
                    text-align: center;
                    margin: 5px;
                    line-height: 1.5;
                }
                .navigation a:hover {
                    background-color: #0056b3;
                    transition: background-color 0.2s;
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
                <h3 style="margin-top: 0; color: #666;">Experiment: {{ experiment_name }}</h3>
                <div style="display: flex; align-items: center; gap: 20px;">
                    <form class="jump-form" action="{{ url_for('view_completions') }}" method="get">
                        <label for="row_id">Jump to Row ID:</label>
                        <input type="number" id="row_id" name="row_id" min="1">
                        <button type="submit">Go</button>
                    </form>
                    <div class="navigation">
                        <a href="{{ url_for('view_completions', page=1) }}">View All Completions</a>
                        <a href="{{ url_for('view_completions', page=1, filter='correct') }}">View Correct Completions</a>
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
                <div class="column-definitions" style="color: #666; margin-bottom: 15px; font-size: 0.9em;">
                    <p><strong>Problem Difficulty:</strong> Percentage of straight-shot solutions that were successful during initial problem evaluation.</p>
                    <p><strong>Recoveries:</strong> The fraction of incorrect solutions that, when completed, resulted in a correct answer.</p>
                    <p><strong>Recovery Rate:</strong> The fraction above expressed as a percentage.</p>
                </div>
                <table class="problem-stats-table">
                    <thead>
                        <tr>
                            <th onclick="sortTable(0)" class="sort-arrow">Row ID</th>
                            <th onclick="sortTable(1)">Problem</th>
                            <th onclick="sortTable(2)" title="Original success rate for this problem">Problem Difficulty</th>
                            <th onclick="sortTable(3)">Recoveries</th>
                            <th onclick="sortTable(4)" class="sort-arrow desc">Recovery Rate</th>
                        </tr>
                    </thead>
                    <tbody>
                        {% for _, row in problem_stats.iterrows() %}
                        <tr>
                            <td><a href="{{ url_for('view_completions', row_id=row['row_id']) }}" 
                                  style="text-decoration: none; color: #007bff;">{{ row['row_id'] }}</a></td>
                            <td class="problem-text" title="{{ row['problem'] }}">{{ row['problem'] }}</td>
                            <td style="background-color: hsl({{ (row['normalized_difficulty'] * 1.2) }}, 80%, 85%);">
                                {{ "%.1f%%"|format(row['difficulty']) }}</td>
                            <td>{{ row['recoveries'] }} / {{ row['attempts'] }}</td>
                            <td style="background-color: hsl({{ (row['recovery_rate'] * 1.2) }}, 80%, 85%);">
                                {{ "%.1f%%"|format(row['recovery_rate']) }}</td>
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
        problem_stats=per_problem_stats,
        experiment_name=EXPERIMENT_NAME
    )

@app.route("/completions")
def view_completions():
    # Get filter parameter
    completion_filter = request.args.get("filter")
    
    # Filter dataframe if needed
    filtered_df = df[df['completion_verification_result'] == True] if completion_filter == 'correct' else df
    
    # Get row_id from query parameters if provided
    row_id = request.args.get("row_id", type=int)
    if row_id:
        try:
            page = filtered_df[filtered_df['row_id'] == row_id].index[0] + 1
        except IndexError:
            page = 1
    else:
        page = request.args.get("page", 1, type=int)
    
    if page < 1 or page > len(filtered_df):
        page = 1

    row = filtered_df.iloc[page - 1]
    
    # Calculate recovery rate for this problem
    problem_df = filtered_df[filtered_df['row_id'] == row['row_id']]
    problem_stats = problem_df['completion_verification_result']
    problem_recovery_rate = problem_stats.mean() * 100  # Convert to percentage
    
    # Calculate total solutions and completions for this problem
    total_solutions = len(problem_df['solution_id'].unique())
    solution_ids = sorted(problem_df['solution_id'].unique())
    current_solution_num = solution_ids.index(row['solution_id'])
    
    completions_for_solution = problem_df[problem_df['solution_id'] == row['solution_id']]
    total_completions = len(completions_for_solution['completion_id'].unique())
    completion_ids = sorted(completions_for_solution['completion_id'].unique())
    current_completion_num = completion_ids.index(row['completion_id'])
    
    # Calculate problem-wide recovery rate
    problem_recovery_rate = problem_stats.mean() * 100
    
    # Calculate solution-specific recovery rate
    solution_stats = completions_for_solution['completion_verification_result']
    solution_recovery_rate = solution_stats.mean() * 100
    
    completion_data = {
        "row_id": int(row.get("row_id", 0)),
        "solution_id": int(row.get("solution_id", 0)),
        "total_solutions": total_solutions - 1,
        "solution_number": current_solution_num,
        "completion_id": int(row.get("completion_id", 0)),
        "total_completions": total_completions - 1,
        "completion_number": current_completion_num,
        "problem_difficulty": float(row.get("row_id_success_rate", 0)) * 100,
        "problem_recovery_rate": problem_recovery_rate,
        "solution_recovery_rate": solution_recovery_rate,
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
            <style>
                body { 
                    font-family: Arial, sans-serif; 
                    line-height: 1.6; 
                    padding: 20px;
                    max-width: 1800px;
                    margin: 0 auto;
                    background-color: #1a1a1a;
                    color: #fff;
                }
                .header {
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    margin-bottom: 20px;
                }
                .completion { 
                    display: flex;
                    justify-content: space-between;
                    margin-top: 20px;
                }
                .section {
                    flex: 0 0 calc(50% - 15px);  /* Fixed width, no growing/shrinking */
                    background-color: #242424;
                    padding: 20px;
                    border-radius: 4px;
                    box-sizing: border-box;
                }
                .content-box { 
                    background-color: #2a2a2a;
                    padding: 10px; 
                    border-radius: 4px;
                    margin-bottom: 15px;
                    word-wrap: break-word;
                    overflow-wrap: break-word;
                    white-space: pre-wrap;
                }
                /* Handle LaTeX content */
                .MathJax_Display {
                    max-width: 100% !important;
                    font-size: 0.9em !important;
                }
                .MathJax {
                    max-width: 100% !important;
                }
                /* Force inline math to wrap */
                .MathJax_Display > .MathJax {
                    display: inline-block !important;
                }
                h2, h3 { 
                    color: #fff;
                    margin-top: 20px;
                    margin-bottom: 10px;
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
                @media (max-width: 768px) {
                    .completion {
                        flex-direction: column;
                    }
                }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Math Problem Completion Viewer ({{ page }}/{{ total_pages }})</h1>
                <h3 style="margin-top: 0; color: #666;">Experiment: {{ experiment_name }}</h3>
                <div style="display: flex; align-items: center; gap: 20px;">
                    <form class="jump-form" action="{{ url_for('view_completions') }}" method="get">
                        <label for="row_id">Jump to Row ID:</label>
                        <input type="number" id="row_id" name="row_id" min="1">
                        <button type="submit">Go</button>
                    </form>
                    <div class="navigation">
                        {% if page > 1 %}
                            <a href="{{ url_for('view_completions', page=page-1, filter=request.args.get('filter')) }}">Previous</a>
                        {% endif %}
                        {% if page < total_pages %}
                            <a href="{{ url_for('view_completions', page=page+1, filter=request.args.get('filter')) }}">Next</a>
                        {% endif %}
                        <a href="{{ url_for('stats') }}">Back to Statistics</a>
                    </div>
                </div>
            </div>
            <div style="background-color: {{ 'lightgreen' if completion_data.completion_verification_result else 'lightcoral' }}; 
                        padding: 15px; 
                        border-radius: 4px;">
                <h2 style="margin: 0 0 10px 0;">
                    Row ID: {{ completion_data.row_id }} | 
                    Solution ID: {{ completion_data.solution_number }}/{{ completion_data.total_solutions }} | 
                    Completion ID: {{ completion_data.completion_number }}/{{ completion_data.total_completions }}
                </h2>
                <h3 style="margin: 0;">
                    Problem Difficulty: {{ "%.1f%%"|format(completion_data.problem_difficulty) }} | 
                    Problem Recovery Rate: {{ "%.1f%%"|format(completion_data.problem_recovery_rate) }} | 
                    Solution Recovery Rate: {{ "%.1f%%"|format(completion_data.solution_recovery_rate) }}
                </h3>
            </div>
            <div class="completion">
                <div class="section">
                    <h2>Problem:</h2>
                    <div class="content-box">{{ completion_data.problem|safe }}</div>
                    
                    <h2>Ground Truth Solution:</h2>
                    <div class="content-box">{{ completion_data.solution|safe }}</div>

                    <h2>Candidate Solution:</h2>
                    <div class="content-box">{{ completion_data.candidate_solution|safe }}</div>

                    <h2>Candidate Verification Result:</h2>
                    <div class="verification-box verification-{{ completion_data.candidate_verification_result|lower }}">
                        {{ completion_data.candidate_verification_result }}
                    </div>

                    <h2>Candidate Verification Reasoning:</h2>
                    <div class="content-box">{{ completion_data.candidate_verification_reasoning|safe }}</div>
                </div>
                <div class="section">
                    <h2>Prefix:</h2>
                    <div class="content-box">{{ completion_data.prefix|safe }}</div>

                    <h2>Completion:</h2>
                    <div class="content-box">{{ completion_data.completion|safe }}</div>
                    
                    <h2>Completion Verification Result:</h2>
                    <div class="verification-box verification-{{ completion_data.completion_verification_result|lower }}">
                        {{ completion_data.completion_verification_result }}
                    </div>
                    
                    <h2>Completion Verification Reasoning:</h2>
                    <div class="content-box">{{ completion_data.completion_verification_reasoning|safe }}</div>
                </div>
            </div>
        </body>
        </html>
        """,
        completion_data=completion_data,
        page=page,
        total_pages=len(filtered_df),
        experiment_name=EXPERIMENT_NAME
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