from pyspark.sql import SparkSession
import json
import matplotlib.pyplot as plt
import os

# Create Spark Session
spark = SparkSession.builder.appName("MemoryAnalysis").getOrCreate()

# Function to load results
def load_results(folder):
    results_path = os.path.join(folder, 'results.json')
    with open(results_path, 'r') as f:
        return json.load(f)

# Load results from both folders
without_zone_results = load_results('without_zone')
with_zone_results = load_results('with_zone')

# Metrics to analyze
metrics = [
    'memory', 
    'vocab', 
    'string_store', 
    'overall_memory_saved'
]

# Create DataFrame for display
data = [
    {
        'Metric': metric.replace('_', ' ').title(), 
        'Without Zone': without_zone_results[metric], 
        'With Zone': with_zone_results[metric]
    } for metric in metrics
]

df = spark.createDataFrame(data)

# Display table
display(df)

# Create individual plots for each metric
for metric in metrics:
    plt.figure(figsize=(8,6))
    
    without_zone_val = without_zone_results[metric]
    with_zone_val = with_zone_results[metric]
    
    plt.bar(['Without Zone', 'With Zone'], 
            [without_zone_val, with_zone_val], 
            color=['blue', 'red'], 
            alpha=0.7)
    
    plt.title(f'{metric.replace("_", " ").title()} Comparison')
    plt.ylabel('Value')
    
    # Add value labels on top of each bar
    for i, v in enumerate([without_zone_val, with_zone_val]):
        plt.text(i, v, f'{v}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()

# Summary of results
print("\nSummary Results:")
for metric in metrics:
    without_zone_val = without_zone_results[metric]
    with_zone_val = with_zone_results[metric]
    reduction = (without_zone_val - with_zone_val) / without_zone_val * 100 if without_zone_val != 0 else 0
    print(f"{metric.replace('_', ' ').title()}: ")
    print(f"  Without Zone: {without_zone_val}")
    print(f"  With Zone: {with_zone_val}")
    print(f"  Reduction: {reduction:.2f}%\n")
