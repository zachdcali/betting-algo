#!/usr/bin/env python3
"""
Demo report generator - creates the HTML report from existing JSON results
This allows users to see the report format without running the full scraper
"""

import json
import os
from pathlib import Path

def generate_demo_report():
    """Generate HTML report from frozen JSON results"""
    
    # Get paths
    base_dir = Path(__file__).parent.parent  # /src/scraping/
    json_path = base_dir / "analysis_output" / "analysis_results.json"
    html_path = base_dir / "analysis_output" / "utr_analysis_report.html" 
    
    if not json_path.exists():
        print(f"Error: {json_path} not found")
        print("This demo requires the analysis_results.json file to be present")
        return False
    
    # Load results
    with open(json_path, 'r') as f:
        results = json.load(f)
    
    # Generate HTML from template
    html_content = generate_html_template(results)
    
    # Write HTML file
    with open(html_path, 'w') as f:
        f.write(html_content)
    
    print(f"✅ Demo report generated: {html_path}")
    print(f"📊 Based on {results['results']['total_matches']:,} matches")
    print(f"🎯 Overall accuracy: {results['results']['overall_accuracy']:.2%}")
    print(f"\nOpen the HTML file in your browser to view the full report.")
    
    return True

def generate_html_template(results):
    """Generate HTML content from results JSON"""
    
    data = results['results']
    
    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>UTR Predictive Analysis Report (Demo)</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1, h2 {{ color: #2c3e50; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        .plot {{ margin: 20px 0; text-align: center; }}
        .plot img {{ max-width: 100%; border: 1px solid #ddd; }}
        .stats {{ background-color: #f9f9f9; padding: 15px; border-radius: 5px; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background-color: #f2f2f2; }}
        .demo-note {{ background-color: #e8f4fd; padding: 15px; border-radius: 5px; border-left: 4px solid #2196F3; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="demo-note">
            <h3>📊 Demo Report</h3>
            <p>This is a demonstration of the UTR analysis report generated from frozen results. 
            The actual scraping pipeline would collect fresh data and generate updated visualizations.</p>
        </div>
        
        <h1>UTR Predictive Analysis Report</h1>
        <p>Generated on: {results['timestamp']}</p>
        
        <div class="stats">
            <h2>Key Statistics</h2>
            <ul>
                <li>Overall UTR prediction accuracy (higher-UTR rule, ties=0.5): {data['overall_accuracy']:.2%}</li>
                <li>Total matches analyzed: {data['total_matches']:,} (ties: {data['ties']:,})</li>
                <li>Mean UTR in dataset: {data['utr_distribution']['mean']:.2f} (range: {data['utr_distribution']['min']:.2f}-{data['utr_distribution']['max']:.2f})</li>
                <li>Brier score (test): {data['calibration']['brier']:.3f}</li>
                <li>ECE (test): {data['calibration']['ece']:.3f} | MCE (test): {data['calibration']['mce']:.3f}</li>
            </ul>
        </div>
        
        <div class="stats">
            <h2>Methodology Notes</h2>
            <p><strong>Higher-UTR Rule:</strong> Predictions are based on the simple rule that the player with higher UTR wins. 
            Ties (equal UTR) are counted as 0.5 credit to both outcomes, reflecting the inherent uncertainty when players have identical ratings.</p>
            <p><strong>Validation:</strong> All model metrics (Brier, ECE, MCE, AUC) are computed on a holdout test set to ensure unbiased evaluation.</p>
        </div>
        
        <h2>Accuracy by UTR Level</h2>
        <div class="stats">
            <table>
                <tr>
                    <th>UTR Level</th>
                    <th>Accuracy</th>
                    <th>Sample Size</th>
                    <th>95% CI Lower</th>
                    <th>95% CI Upper</th>
                </tr>"""
    
    # Add accuracy by level table
    for level, stats in data['accuracy_by_level'].items():
        html += f"""
                <tr>
                    <td>{level}</td>
                    <td>{stats['acc']:.2%}</td>
                    <td>{stats['n']:,}</td>
                    <td>{stats['lo']:.2%}</td>
                    <td>{stats['hi']:.2%}</td>
                </tr>"""
    
    html += """
            </table>
        </div>
        
        <h2>Regression Results</h2>
        <table>
            <tr>
                <th>Level</th>
                <th>Matches</th>
                <th>Coefficient</th>
                <th>Intercept</th>
                <th>Accuracy</th>
                <th>Brier</th>
                <th>AUC</th>
            </tr>"""
    
    # Add regression results
    for result in data['regression_results']:
        html += f"""
            <tr>
                <td>{result['Level']}</td>
                <td>{result['Matches']:,}</td>
                <td>{result['Coefficient']:.4f}</td>
                <td>{result['Intercept']:.6f}</td>
                <td>{result['Accuracy']:.2%}</td>
                <td>{result['Brier']:.3f}</td>
                <td>{result['AUC']:.3f}</td>
            </tr>"""
    
    html += """
        </table>
        
        <div class="demo-note">
            <h3>📈 Visualization Plots</h3>
            <p>In the full pipeline, this report would include interactive visualizations:</p>
            <ul>"""
    
    # List the plots that would be generated
    for plot in results['plots_generated']:
        html += f"<li>{plot.replace('_', ' ').replace('.png', '').title()}</li>"
    
    html += """
            </ul>
            <p><strong>To generate the full report with plots:</strong><br>
            1. Set up UTR credentials: <code>export UTR_EMAIL=... UTR_PASSWORD=...</code><br>
            2. Run the analysis: <code>python utr_analysis_report.py</code></p>
        </div>
    </div>
</body>
</html>"""
    
    return html

if __name__ == "__main__":
    success = generate_demo_report()
    if not success:
        exit(1)