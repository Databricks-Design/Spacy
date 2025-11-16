# ============================================
# MEMORY CONSUMPTION COMPARISON: Real vs Synthetic Data (Both With Memory Zones)
# ============================================

import json
import matplotlib.pyplot as plt
import os

# ===== CONFIGURE PATHS =====
REAL_DATA_FOLDER = ""
SYNTHETIC_DATA_FOLDER = ""

# ===== LOAD RESULTS =====
real_json_path = os.path.join(REAL_DATA_FOLDER, 'results.json')
synthetic_json_path = os.path.join(SYNTHETIC_DATA_FOLDER, 'results.json')

with open(real_json_path, 'r') as f:
    real_results = json.load(f)

with open(synthetic_json_path, 'r') as f:
    synthetic_results = json.load(f)

# ===== CREATE COMPARISON PLOT =====
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Memory Consumption Comparison: Real Data vs Synthetic Data (With Memory Zones)', 
             fontsize=16, fontweight='bold')

# Plot 1: Memory Consumption Over Batches
ax1 = axes[0, 0]
ax1.plot(real_results['batches'], real_results['memory_mb'], 
         label='Real Data (With Zone)', color='tab:green', marker='o', markersize=3, linewidth=2)
ax1.plot(synthetic_results['batches'], synthetic_results['memory_mb'], 
         label='Synthetic Data (With Zone)', color='tab:purple', marker='s', markersize=3, linewidth=2)
ax1.set_title('Memory Consumption Over Batches', fontsize=12, fontweight='bold')
ax1.set_xlabel('Batch Number')
ax1.set_ylabel('Total Memory Usage (MB)', color='tab:blue')
ax1.tick_params(axis='y', labelcolor='tab:blue')
ax1.legend(loc='upper left')
ax1.grid(True, axis='both', linestyle='--', alpha=0.6)

# Plot 2: Vocabulary Size Over Batches
ax2 = axes[0, 1]
ax2.plot(real_results['batches'], real_results['vocab_size'], 
         label='Real Data (With Zone)', color='tab:green', marker='o', markersize=3, linewidth=2)
ax2.plot(synthetic_results['batches'], synthetic_results['vocab_size'], 
         label='Synthetic Data (With Zone)', color='tab:purple', marker='s', markersize=3, linewidth=2)
ax2.set_title('Vocabulary Size Over Batches', fontsize=12, fontweight='bold')
ax2.set_xlabel('Batch Number')
ax2.set_ylabel('Vocabulary Size', color='tab:red')
ax2.tick_params(axis='y', labelcolor='tab:red')
ax2.legend(loc='upper left')
ax2.grid(True, axis='both', linestyle='--', alpha=0.6)

# Plot 3: String Store Size Over Batches
ax3 = axes[1, 0]
ax3.plot(real_results['batches'], real_results['string_store_size'], 
         label='Real Data (With Zone)', color='tab:green', marker='o', markersize=3, linewidth=2)
ax3.plot(synthetic_results['batches'], synthetic_results['string_store_size'], 
         label='Synthetic Data (With Zone)', color='tab:purple', marker='s', markersize=3, linewidth=2)
ax3.set_title('String Store Size Over Batches', fontsize=12, fontweight='bold')
ax3.set_xlabel('Batch Number')
ax3.set_ylabel('String Store Size', color='tab:pink')
ax3.tick_params(axis='y', labelcolor='tab:pink')
ax3.legend(loc='upper left')
ax3.grid(True, axis='both', linestyle='--', alpha=0.6)

# Plot 4: Final Memory Growth Comparison (Bar Chart)
ax4 = axes[1, 1]
test_names = ['Real Data\n(With Zone)', 'Synthetic Data\n(With Zone)']
memory_growths = [real_results['memory_growth'], synthetic_results['memory_growth']]
colors = ['tab:green', 'tab:purple']

bars = ax4.bar(test_names, memory_growths, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
ax4.set_title('Final Memory Growth Comparison', fontsize=12, fontweight='bold')
ax4.set_ylabel('Total Memory Growth (MB)')
ax4.grid(True, axis='y', linestyle='--', alpha=0.6)

# Annotate bars with values
for bar in bars:
    height = bar.get_height()
    ax4.annotate(f'{height:.2f} MB',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 5),
                textcoords="offset points",
                ha='center', va='bottom',
                fontsize=11, fontweight='bold')

# Add summary statistics as text
summary_text = f"""
SUMMARY STATISTICS:

Real Data (With Zone):
  • Total Batches: {len(real_results['batches'])}
  • Memory Growth: {real_results['memory_growth']:.2f} MB
  • Final Vocab Size: {real_results['vocab_size'][-1]:,}
  • Final String Store: {real_results['string_store_size'][-1]:,}
  • Elapsed Time: {real_results['elapsed_time']:.2f}s

Synthetic Data (With Zone):
  • Total Batches: {len(synthetic_results['batches'])}
  • Memory Growth: {synthetic_results['memory_growth']:.2f} MB
  • Final Vocab Size: {synthetic_results['vocab_size'][-1]:,}
  • Final String Store: {synthetic_results['string_store_size'][-1]:,}
  • Elapsed Time: {synthetic_results['elapsed_time']:.2f}s
"""

plt.figtext(0.02, 0.02, summary_text, fontsize=9, family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

plt.tight_layout(rect=[0, 0.15, 1, 0.96])
plt.show()

print("✓ Plot displayed successfully!")
