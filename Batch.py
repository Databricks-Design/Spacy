#!/usr/bin/env python3

import os
import json
import numpy as np
import pandas as pd
import time
import gc
import psutil
import matplotlib.pyplot as plt

# Assuming your project structure for these imports
from unidecode import unidecode
from packages.spacy_model import SpacyModel
from triton_python_backend_utils import Tensor
from tests.mocks import mockInferenceRequest

# --- Utilities for Memory Profiling and Reporting ---

class MemoryProfiler:
    """Memory monitoring and data collection utilities"""
    def __init__(self):
        self.process = psutil.Process(os.getpid())
    def get_memory_mb(self) -> float:
        return self.process.memory_info().rss / 1024 / 1024
    def get_vocab_stats(self, nlp) -> dict:
        """Get vocabulary and string store statistics"""
        return {
            'vocab_size': len(nlp.vocab),
            'string_store_size': len(nlp.vocab.strings),
        }

def generate_report(results: list, output_path: str):
    """Generates and saves a visual report of memory and store consumption."""
    print("\nGENERATING INVESTIGATION REPORT...")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('NER Pipeline Memory Investigation', fontsize=16, fontweight='bold')

    # --- Plot 1: Consumption Over Time ---
    ax1 = axes[0]
    ax1.set_title('Consumption Over Batches')
    ax1.set_xlabel('Batch Number')
    ax1.set_ylabel('Total Memory Usage (MB)', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.grid(True, axis='y', linestyle='--', alpha=0.6)

    ax2 = ax1.twinx()
    ax2.set_ylabel('Vocab / String Store Size', color='tab:red')
    ax2.tick_params(axis='y', labelcolor='tab:red')

    for result in results:
        ax1.plot(result['batches'], result['memory_mb'], label=f"{result['test_name']} (Memory)", color='tab:blue' if 'Without' in result['test_name'] else 'tab:cyan', marker='.')
        ax2.plot(result['batches'], result['vocab_size'], label=f"{result['test_name']} (Vocab)", color='tab:red' if 'Without' in result['test_name'] else 'tab:orange', linestyle='--')
        ax2.plot(result['batches'], result['string_store_size'], label=f"{result['test_name']} (Strings)", color='tab:pink' if 'Without' in result['test_name'] else 'tab:purple', linestyle=':')

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='center right')

    # --- Plot 2: Final Memory Growth Comparison ---
    ax3 = axes[1]
    ax3.set_title('Final Memory Growth Comparison')
    test_names = [r['test_name'] for r in results]
    memory_growths = [r['memory_growth'] for r in results]
    colors = ['tab:red', 'tab:green']
    
    bars = ax3.bar(test_names, memory_growths, color=colors)
    ax3.set_ylabel('Total Memory Growth (MB)')
    
    for bar in bars:
        height = bar.get_height()
        ax3.annotate(f'{height:.1f} MB',
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3),
                     textcoords="offset points",
                     ha='center', va='bottom')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    report_filename = os.path.join(output_path, 'memory_consumption_report.png')
    plt.savefig(report_filename, dpi=300)
    plt.close()
    print(f"Visual report saved to: {report_filename}")


def run_test(ner_model: SpacyModel, df: pd.DataFrame, batch_size: int, test_name: str, output_folder: str, batches_per_file: int = 10):
    """Runs a test scenario, processes data in batches, and profiles memory."""
    print(f"\n--- Running Test: {test_name} ---")
    
    # Create output folder for this test
    os.makedirs(output_folder, exist_ok=True)
    
    profiler = MemoryProfiler()
    initial_memory = profiler.get_memory_mb()
    
    memory_data, vocab_data, string_store_data, batch_numbers = [], [], [], []
    
    # Track file splitting
    file_counter = 1
    batches_in_current_file = 0
    first_batch_in_file = True
    current_csv_path = os.path.join(output_folder, f'output_part_{file_counter:03d}.csv')
    
    num_batches = (len(df) + batch_size - 1) // batch_size
    for i in range(0, len(df), batch_size):
        batch_df = df.iloc[i:i+batch_size]
        
        dsp = batch_df['dsp'].to_list()
        mmo = batch_df['mmo'].to_list()
        dsp_vec = np.array(dsp, dtype='|S0').reshape(len(dsp), 1)
        mmo_vec = np.array(mmo, dtype='|S0').reshape(len(mmo), 1)
        
        requests = [Request(inputs=[
            Tensor(data=dsp_vec, name='dsp'),
            Tensor(data=mmo_vec, name='mmo')
        ])]
        
        raw_results = ner_model.execute(requests)
        
        outputs = []
        for raw_result in raw_results:
            labels, extracted_texts, entity_ids = raw_result.output_tensors()
            labels = labels.as_numpy().tolist()
            extracted_texts = extracted_texts.as_numpy().tolist()
            entity_ids = entity_ids.as_numpy().tolist()
            for label_list, extracted_text_list, entity_id_list in zip(labels, extracted_texts, entity_ids):
                these_outputs = []
                decoded_labels = [x.decode('utf-8') for x in label_list]
                decoded_entity_ids = [x.decode('utf-8') for x in entity_id_list]
                for label, extracted_text, entity_id in zip(decoded_labels, extracted_text_list, decoded_entity_ids):
                    if label:
                        these_outputs.append({
                            'entity_type': label,
                            'extracted_entity': extracted_text,
                            'standardized_entity': entity_id
                        })
                outputs.append(these_outputs)
        
        batch_df['outputs_ner'] = outputs
        
        # Write to current CSV file
        if first_batch_in_file:
            batch_df.to_csv(current_csv_path, index=False, mode='w')
            first_batch_in_file = False
        else:
            batch_df.to_csv(current_csv_path, index=False, mode='a', header=False)
        
        batches_in_current_file += 1
        
        # Check if we need to start a new file
        if batches_in_current_file >= batches_per_file:
            print(f"  ✓ Completed file: {current_csv_path} ({batches_in_current_file} batches, {batches_in_current_file * batch_size} records)")
            file_counter += 1
            batches_in_current_file = 0
            first_batch_in_file = True
            current_csv_path = os.path.join(output_folder, f'output_part_{file_counter:03d}.csv')
        
        batch_num = (i // batch_size) + 1
        memory_mb = profiler.get_memory_mb()
        vocab_stats = profiler.get_vocab_stats(ner_model.nlp)
        
        memory_data.append(memory_mb)
        vocab_data.append(vocab_stats['vocab_size'])
        string_store_data.append(vocab_stats['string_store_size'])
        batch_numbers.append(batch_num)
        
        print(
            f"  Batch {batch_num}/{num_batches}: "
            f"Memory = {memory_mb:.2f} MB | "
            f"Vocab Size = {vocab_stats['vocab_size']} | "
            f"String Store = {vocab_stats['string_store_size']}"
        )
        
        del batch_df, dsp, mmo, requests, raw_results, outputs
        gc.collect()
        time.sleep(0.05)
    
    # Log final file if it has data
    if batches_in_current_file > 0:
        print(f"  ✓ Completed file: {current_csv_path} ({batches_in_current_file} batches, {batches_in_current_file * batch_size} records)")
        
    final_memory = profiler.get_memory_mb()
    print(f"Test complete. Output saved to folder: {output_folder}")
    print(f"Total files created: {file_counter}")
    
    return {
        'test_name': test_name,
        'memory_mb': memory_data,
        'vocab_size': vocab_data,
        'string_store_size': string_store_data,
        'batches': batch_numbers,
        'memory_growth': final_memory - initial_memory,
    }

# --- Main Execution Block ---

def main():
    # --------------------------------------------------------------------
    # SCRIPT CONFIGURATION
    # --------------------------------------------------------------------
    # 1. Set the batch size for processing records.
    BATCH_SIZE = 10000
    
    # 2. Set how many batches per CSV file (to avoid file too large error)
    BATCHES_PER_FILE = 10  # 10 batches × 10,000 records = 100,000 records per file

    INPUT_CSV = "./case_examples.csv"
    
    OUTPUT_DIR = "./output"
    # --------------------------------------------------------------------
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Initialize Model
    ner = SpacyModel()
    init_args = {'model_name': 'modelr'}
    ner.initialize(init_args)
    
    # Load and Prepare Data
    df = pd.read_csv(INPUT_CSV, keep_default_na=False)
    df['dsp'] = df['dsp'].apply(unidecode)
    df['mmo'] = df['mmo'].apply(unidecode)
    print(f"Loaded {len(df)} records to be processed in batches of {BATCH_SIZE}.")
    print(f"Each CSV file will contain {BATCHES_PER_FILE} batches ({BATCHES_PER_FILE * BATCH_SIZE} records).")
    
    # Run Investigation
    all_results = []
    
    # TEST 1: Without Memory Zone
    ner.use_memory_zone = False
    without_zone_folder = os.path.join(OUTPUT_DIR, "without_zone")
    results_without_zone = run_test(
        ner, df, BATCH_SIZE, "Without Memory Zone", 
        without_zone_folder,
        BATCHES_PER_FILE
    )
    all_results.append(results_without_zone)
    
    time.sleep(2)
    gc.collect()
    
    # TEST 2: With Memory Zone
    ner.use_memory_zone = True
    with_zone_folder = os.path.join(OUTPUT_DIR, "with_zone")
    results_with_zone = run_test(
        ner, df, BATCH_SIZE, "With Memory Zone",
        with_zone_folder,
        BATCHES_PER_FILE
    )
    all_results.append(results_with_zone)
    
    # Generate Final Report
    generate_report(all_results, OUTPUT_DIR)
    
    print("\nInvestigation Finished.")

if __name__ == "__main__":
    main()
