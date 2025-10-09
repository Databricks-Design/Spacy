import json
import os
from pyspark.sql import SparkSession

# Initialize Spark
spark = SparkSession.builder.getOrCreate()

# Load results
OUTPUT_DIR = "./output"
without_json_path = os.path.join(OUTPUT_DIR, "without_zone", "test_results.json")
with_json_path = os.path.join(OUTPUT_DIR, "with_zone", "test_results.json")

with open(without_json_path, 'r') as f:
    results_without = json.load(f)

with open(with_json_path, 'r') as f:
    results_with = json.load(f)

# Convert to DataFrame - Without Zone
print("="*100)
print("🔵 WITHOUT MEMORY ZONE")
print("="*100)
df_without = spark.createDataFrame([
    {'batch': b, 'memory_mb': m, 'vocab_size': v, 'string_store_size': s}
    for b, m, v, s in zip(
        results_without['batches'],
        results_without['memory_mb'],
        results_without['vocab_size'],
        results_without['string_store_size']
    )
])
display(df_without)

# Convert to DataFrame - With Zone
print("\n" + "="*100)
print("🟢 WITH MEMORY ZONE")
print("="*100)
df_with = spark.createDataFrame([
    {'batch': b, 'memory_mb': m, 'vocab_size': v, 'string_store_size': s}
    for b, m, v, s in zip(
        results_with['batches'],
        results_with['memory_mb'],
        results_with['vocab_size'],
        results_with['string_store_size']
    )
])
display(df_with)

import json
import os
import matplotlib.pyplot as plt

# Load results
OUTPUT_DIR = "./output"
without_json_path = os.path.join(OUTPUT_DIR, "without_zone", "test_results.json")
with_json_path = os.path.join(OUTPUT_DIR, "with_zone", "test_results.json")

with open(without_json_path, 'r') as f:
    results_without = json.load(f)

with open(with_json_path, 'r') as f:
    results_with = json.load(f)

batches = results_without['batches']

# ============================================================================
# GRAPH 1: Memory Consumption
# ============================================================================
plt.figure(figsize=(20, 6))
plt.plot(batches, results_without['memory_mb'], 'b-', linewidth=2.5, label='Without Memory Zone', alpha=0.8)
plt.plot(batches, results_with['memory_mb'], 'g-', linewidth=2.5, label='With Memory Zone', alpha=0.8)
plt.fill_between(batches, results_without['memory_mb'], results_with['memory_mb'], alpha=0.2, color='yellow')

plt.xlabel('Batch Number', fontsize=14, fontweight='bold')
plt.ylabel('Memory Usage (MB)', fontsize=14, fontweight='bold')
plt.title('Memory Consumption Over Batches', fontsize=16, fontweight='bold', pad=20)
plt.legend(fontsize=12, loc='upper left')
plt.grid(True, alpha=0.3, linestyle='--')
plt.xticks(fontsize=11)
plt.yticks(fontsize=11)

# Add final values annotation
final_without = results_without['memory_mb'][-1]
final_with = results_with['memory_mb'][-1]
savings = final_without - final_with
plt.annotate(f'Final: {final_without:.0f} MB', 
             xy=(batches[-1], final_without), xytext=(-80, 20),
             textcoords='offset points', fontsize=10, 
             bbox=dict(boxstyle='round,pad=0.5', facecolor='blue', alpha=0.2),
             arrowprops=dict(arrowstyle='->', color='blue'))
plt.annotate(f'Final: {final_with:.0f} MB\nSaved: {savings:.0f} MB', 
             xy=(batches[-1], final_with), xytext=(-80, -40),
             textcoords='offset points', fontsize=10,
             bbox=dict(boxstyle='round,pad=0.5', facecolor='green', alpha=0.2),
             arrowprops=dict(arrowstyle='->', color='green'))

plt.tight_layout()
plt.show()

# ============================================================================
# GRAPH 2: Vocabulary Size
# ============================================================================
plt.figure(figsize=(20, 6))
plt.plot(batches, results_without['vocab_size'], 'r-', linewidth=2.5, label='Without Memory Zone', alpha=0.8)
plt.plot(batches, results_with['vocab_size'], 'orange', linewidth=2.5, label='With Memory Zone', alpha=0.8)

plt.xlabel('Batch Number', fontsize=14, fontweight='bold')
plt.ylabel('Vocabulary Size', fontsize=14, fontweight='bold')
plt.title('Vocabulary Growth Over Batches', fontsize=16, fontweight='bold', pad=20)
plt.legend(fontsize=12, loc='upper left')
plt.grid(True, alpha=0.3, linestyle='--')
plt.xticks(fontsize=11)
plt.yticks(fontsize=11)

# Add final values annotation
final_vocab_without = results_without['vocab_size'][-1]
final_vocab_with = results_with['vocab_size'][-1]
vocab_reduction = final_vocab_without - final_vocab_with
reduction_pct = (vocab_reduction / final_vocab_without * 100)

plt.annotate(f'Final: {final_vocab_without:,}', 
             xy=(batches[-1], final_vocab_without), xytext=(-80, 20),
             textcoords='offset points', fontsize=10,
             bbox=dict(boxstyle='round,pad=0.5', facecolor='red', alpha=0.2),
             arrowprops=dict(arrowstyle='->', color='red'))
plt.annotate(f'Final: {final_vocab_with:,}\nReduction: {vocab_reduction:,} ({reduction_pct:.1f}%)', 
             xy=(batches[-1], final_vocab_with), xytext=(-100, -50),
             textcoords='offset points', fontsize=10,
             bbox=dict(boxstyle='round,pad=0.5', facecolor='orange', alpha=0.2),
             arrowprops=dict(arrowstyle='->', color='orange'))

plt.tight_layout()
plt.show()

# ============================================================================
# GRAPH 3: String Store Size
# ============================================================================
plt.figure(figsize=(20, 6))
plt.plot(batches, results_without['string_store_size'], 'm-', linewidth=2.5, label='Without Memory Zone', alpha=0.8)
plt.plot(batches, results_with['string_store_size'], 'purple', linewidth=2.5, label='With Memory Zone', alpha=0.8)

plt.xlabel('Batch Number', fontsize=14, fontweight='bold')
plt.ylabel('String Store Size', fontsize=14, fontweight='bold')
plt.title('String Store Growth Over Batches', fontsize=16, fontweight='bold', pad=20)
plt.legend(fontsize=12, loc='upper left')
plt.grid(True, alpha=0.3, linestyle='--')
plt.xticks(fontsize=11)
plt.yticks(fontsize=11)

# Add final values annotation
final_str_without = results_without['string_store_size'][-1]
final_str_with = results_with['string_store_size'][-1]
str_reduction = final_str_without - final_str_with
str_reduction_pct = (str_reduction / final_str_without * 100)

plt.annotate(f'Final: {final_str_without:,}', 
             xy=(batches[-1], final_str_without), xytext=(-80, 20),
             textcoords='offset points', fontsize=10,
             bbox=dict(boxstyle='round,pad=0.5', facecolor='magenta', alpha=0.2),
             arrowprops=dict(arrowstyle='->', color='magenta'))
plt.annotate(f'Final: {final_str_with:,}\nReduction: {str_reduction:,} ({str_reduction_pct:.1f}%)', 
             xy=(batches[-1], final_str_with), xytext=(-100, -50),
             textcoords='offset points', fontsize=10,
             bbox=dict(boxstyle='round,pad=0.5', facecolor='purple', alpha=0.2),
             arrowprops=dict(arrowstyle='->', color='purple'))

plt.tight_layout()
plt.show()

# ============================================================================
# GRAPH 4: Memory Savings Per Batch
# ============================================================================
memory_savings = [results_without['memory_mb'][i] - results_with['memory_mb'][i] 
                  for i in range(len(batches))]

plt.figure(figsize=(20, 6))
plt.plot(batches, memory_savings, 'darkgreen', linewidth=2.5, label='Memory Saved per Batch', alpha=0.8)
plt.fill_between(batches, 0, memory_savings, alpha=0.3, color='lightgreen')
plt.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)

plt.xlabel('Batch Number', fontsize=14, fontweight='bold')
plt.ylabel('Memory Saved (MB)', fontsize=14, fontweight='bold')
plt.title('Memory Savings Per Batch (Without Zone - With Zone)', fontsize=16, fontweight='bold', pad=20)
plt.legend(fontsize=12, loc='upper left')
plt.grid(True, alpha=0.3, linestyle='--')
plt.xticks(fontsize=11)
plt.yticks(fontsize=11)

avg_savings = sum(memory_savings) / len(memory_savings)
plt.axhline(y=avg_savings, color='red', linestyle='--', linewidth=2, label=f'Average: {avg_savings:.1f} MB', alpha=0.7)
plt.legend(fontsize=12, loc='upper left')

plt.tight_layout()
plt.show()

# ============================================================================
# GRAPH 5: Final Comparison Bar Chart
# ============================================================================
fig, ax = plt.subplots(figsize=(14, 8))

categories = ['Memory Growth\n(MB)', 'Final Vocab\nSize', 'Final String Store\nSize']
without_values = [
    results_without['memory_growth'],
    results_without['vocab_size'][-1] / 1000,  # Scale down for visibility
    results_without['string_store_size'][-1] / 1000  # Scale down for visibility
]
with_values = [
    results_with['memory_growth'],
    results_with['vocab_size'][-1] / 1000,
    results_with['string_store_size'][-1] / 1000
]
savings_values = [
    results_without['memory_growth'] - results_with['memory_growth'],
    (results_without['vocab_size'][-1] - results_with['vocab_size'][-1]) / 1000,
    (results_without['string_store_size'][-1] - results_with['string_store_size'][-1]) / 1000
]

x = range(len(categories))
width = 0.25

bars1 = ax.bar([i - width for i in x], without_values, width, label='Without Memory Zone', 
               color='#E74C3C', alpha=0.8, edgecolor='black', linewidth=1.2)
bars2 = ax.bar([i for i in x], with_values, width, label='With Memory Zone', 
               color='#2ECC71', alpha=0.8, edgecolor='black', linewidth=1.2)
bars3 = ax.bar([i + width for i in x], savings_values, width, label='Savings', 
               color='#F39C12', alpha=0.8, edgecolor='black', linewidth=1.2)

ax.set_ylabel('Value (Vocab & String Store in thousands)', fontsize=14, fontweight='bold')
ax.set_title('Final Resource Usage Comparison', fontsize=16, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(categories, fontsize=12, fontweight='bold')
ax.legend(fontsize=12, loc='upper right')
ax.grid(True, alpha=0.3, axis='y', linestyle='--')

# Add value labels on bars
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.0f}',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 5),
                   textcoords="offset points",
                   ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.show()

print("✅ All graphs generated successfully!")
