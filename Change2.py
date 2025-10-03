# Import Packages
import os
import json
import numpy as np
import pandas as pd
from unidecode import unidecode
from packages.spacy_model import SpacyModel
from triton_python_backend_utils import Tensor
from tests.mocks import mockInferenceRequest
import psutil
import time
import gc
import argparse

# --- HELPER CLASS FOR MEMORY PROFILING ---
class MemoryProfiler:
    def __init__(self):
        self.process = psutil.Process(os.getpid())
    def get_memory_mb(self) -> float:
        return self.process.memory_info().rss / 1024 / 1024

# --- MAIN SCRIPT LOGIC ---

# 1. Standard Model Initialization
ner = SpacyModel()
init_args = {'model_name': 'us__spacy.ner'}
ner.initialize(init_args)
profiler = MemoryProfiler()

# 2. Setup Command-Line Argument Parser
parser = argparse.ArgumentParser(description="Run a specific NER memory test scenario.")
parser.add_argument(
    '--test', 
    type=str, 
    required=True, 
    choices=['baseline', 'fix_only', 'full_pipeline'],
    help="The name of the test to run."
)
args = parser.parse_args()

# 3. Load Full Dataset
full_df = pd.read_csv("./case_examples_ner.csv", keep_default_na=False) # Make sure path is correct
full_df['description'] = full_df['description'].apply(unidecode)
full_df['memo'] = full_df['memo'].apply(unidecode)

# 4. Define All Possible Test Scenarios
test_scenarios = {
    'baseline': {'name': 'Baseline (No memory_zone)', 'use_memory_zone': False, 'aggregate_results': True},
    'fix_only': {'name': 'memory_zone Only (No Aggregation)', 'use_memory_zone': True, 'aggregate_results': False},
    'full_pipeline': {'name': 'Full Pipeline (memory_zone + Aggregation)', 'use_memory_zone': True, 'aggregate_results': True}
}
scenario = test_scenarios[args.test]
output_filename = f"{args.test}_results.json"

print(f"\n{'='*70}\n🔬 RUNNING TEST: {scenario['name']}\n{'='*70}")

# 5. Run the chosen scenario
memory_data, vocab_data, timestamps = [], [], []
all_final_outputs = []
initial_memory = profiler.get_memory_mb()
batch_size = 100

# The loop now processes the entire DataFrame, not a fixed number of iterations
for i in range(0, len(full_df), batch_size):
    df_batch = full_df.iloc[i:i+batch_size]
    if len(df_batch) == 0: break

    # Prepare batch
    descriptions = df_batch['description'].to_list()
    memos = df_batch['memo'].to_list()
    requests = [mockInferenceRequest(inputs=[
        Tensor(data=np.array(descriptions, dtype='|S0').reshape(len(descriptions), 1), name='description'),
        Tensor(data=np.array(memos, dtype='|S0').reshape(len(memos), 1), name='memo')
    ])]

    # --- Detailed Memory Tracking ---
    mem_before_model = profiler.get_memory_mb()
    raw_results = ner.execute(requests, use_memory_zone=scenario['use_memory_zone'])
    mem_after_model = profiler.get_memory_mb()
    model_cost = mem_after_model - mem_before_model
    
    aggregation_cost = 0
    if scenario['aggregate_results']:
        outputs = []
        # Your original post-processing logic
        for raw_result in raw_results:
            labels, extracted_texts, entity_ids = raw_result.output_tensors()
            labels, extracted_texts, entity_ids = labels.as_numpy().tolist(), extracted_texts.as_numpy().tolist(), entity_ids.as_numpy().tolist()
            for label_list, extracted_text_list, entity_id_list in zip(labels, extracted_texts, entity_ids):
                these_outputs = []
                decoded_labels = [x.decode('utf-8') for x in label_list]
                decoded_extracted_texts = [x for x in extracted_text_list]
                decoded_entity_ids = [x.decode('utf-8') for x in entity_id_list]
                for label, extracted_text, entity_id in zip(decoded_labels, decoded_extracted_texts, decoded_entity_ids):
                    if label != '' and label is not None:
                        these_outputs.append({'entity_type': label, 'extracted_entity': extracted_text, 'standardized_entity': entity_id})
                outputs.append(these_outputs)
        all_final_outputs.extend(outputs)
        mem_after_aggregation = profiler.get_memory_mb()
        aggregation_cost = mem_after_aggregation - mem_after_model

    # Vocab Tracking
    vocab_stats = ner.get_vocab_stats()

    batch_num = (i // batch_size) + 1
    if batch_num % 10 == 0:
        growth = profiler.get_memory_mb() - initial_memory
        log_msg = f"  Batch {batch_num}: Mem Growth={growth:+.1f}MB | Vocab={vocab_stats['string_store_size']}"
        if scenario['name'] in ['Baseline (No memory_zone)', 'Full Pipeline (memory_zone + Aggregation)']:
             log_msg += f" | Batch Cost (Model: {model_cost:+.2f}MB, Agg: {aggregation_cost:+.2f}MB)"
        print(log_msg)
        memory_data.append(profiler.get_memory_mb())
        vocab_data.append(vocab_stats['string_store_size'])
        timestamps.append(batch_num)

gc.collect()
final_memory = profiler.get_memory_mb()

# 6. Save all results, including final output data, to a JSON file
results_data = {
    'test_name': scenario['name'],
    'memory_mb': memory_data,
    'vocab_size': vocab_data,
    'iterations': timestamps,
    'memory_growth': final_memory - initial_memory,
    'output_data': all_final_outputs 
}
with open(output_filename, 'w') as f:
    json.dump(results_data, f)

print(f"\n✅ Test complete. Results for '{scenario['name']}' saved to '{output_filename}'")




import json
import os
import glob
import matplotlib.pyplot as plt

def generate_report_and_compare():
    """
    Finds result files, generates a comparative plot, and compares
    the outputs of the 'baseline' and 'full_pipeline' tests.
    """
    print("📊 Generating report and comparing outputs...")
    
    # --- Part 1: Plotting (from before) ---
    result_files = glob.glob('*_results.json')
    if not result_files:
        print("❌ No result files found. Please run the tests in `example.py` first.")
        return

    all_plot_data = []
    for file_path in result_files:
        with open(file_path, 'r') as f:
            all_plot_data.append(json.load(f))
    
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 7))
    for result in sorted(all_plot_data, key=lambda x: x['test_name']):
        if result.get('memory_mb'):
            ax.plot(result['iterations'], result['memory_mb'], label=f"{result['test_name']} (Growth: {result['memory_growth']:.1f}MB)", marker='o')
    ax.set_title('Combined Pipeline Memory Analysis Report', fontsize=16)
    ax.set_xlabel('Batch Number')
    ax.set_ylabel('Total Memory Usage (MB)')
    ax.legend(title="Test Scenarios")
    ax.grid(True)
    report_filename = 'comparison_report.png'
    plt.savefig(report_filename)
    plt.close()
    print(f"✅ Combined plot saved as '{report_filename}'")

    # --- Part 2: Output Comparison ---
    print(f"\n{'='*50}\n🔍 COMPARING OUTPUTS\n{'='*50}")
    baseline_data, pipeline_data = None, None
    try:
        with open('baseline_results.json', 'r') as f:
            baseline_data = json.load(f)
        with open('full_pipeline_results.json', 'r') as f:
            pipeline_data = json.load(f)
    except FileNotFoundError:
        print("❌ Could not compare: 'baseline_results.json' or 'full_pipeline_results.json' not found.")
        return

    baseline_output = baseline_data.get('output_data')
    pipeline_output = pipeline_data.get('output_data')

    if baseline_output == pipeline_output:
        print("✅ SUCCESS: Outputs from 'baseline' and 'full_pipeline' tests are identical.")
    else:
        print("❌ FAILURE: Outputs are different. The fix may have impacted results.")

if __name__ == "__main__":
    generate_report_and_compare()
