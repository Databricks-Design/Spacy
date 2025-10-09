from pyspark.sql import SparkSession
import json
import matplotlib.pyplot as plt

# Create Spark Session
spark = SparkSession.builder.appName("MemoryAnalysis").getOrCreate()

# Read the results JSON
with open('results.json', 'r') as f:
    results = json.load(f)

# Prepare data for table and plotting
metrics = [
    'memory', 
    'vocab', 
    'string_store', 
    'overall_memory_saved'
]

# Create DataFrame for display
data = [
    {
        'Metric': 'Memory (MB)', 
        'Without Zone': results['without_zone'][metric], 
        'With Zone': results['with_zone'][metric]
    } for metric in metrics
]

df = spark.createDataFrame(data)

# Display table
display(df)

# Create bar plot for comparison
plt.figure(figsize=(12,6))

x = range(len(metrics))
width = 0.35

plt.bar([i - width/2 for i in x], 
        [results['without_zone'][m] for m in metrics], 
        width, label='Without Zone', color='blue', alpha=0.7)

plt.bar([i + width/2 for i in x], 
        [results['with_zone'][m] for m in metrics], 
        width, label='With Zone', color='red', alpha=0.7)

plt.xlabel('Metrics')
plt.ylabel('Value')
plt.title('Memory Metrics Comparison')
plt.xticks(x, [m.replace('_', ' ').title() for m in metrics])
plt.legend()

plt.tight_layout()
plt.show()

# Summary of results
print("\nSummary Results:")
for metric in metrics:
    without_zone_val = results['without_zone'][metric]
    with_zone_val = results['with_zone'][metric]
    reduction = (without_zone_val - with_zone_val) / without_zone_val * 100
    print(f"{metric.replace('_', ' ').title()}: ")
    print(f"  Without Zone: {without_zone_val}")
    print(f"  With Zone: {with_zone_val}")
    print(f"  Reduction: {reduction:.2f}%\n")
