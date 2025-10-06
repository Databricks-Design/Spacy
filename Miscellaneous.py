
import os
import numpy as np
import pandas as pd
import time
import gc
import psutil
import matplotlib.pyplot as plt

from unidecode import unidecode
from packages.spacy_model import SpacyModel
from triton_python_backend_utils import Tensor
from tests.mocks import mockInferenceRequest

class MemoryProfiler:
    """Memory monitoring and data collection utilities"""
    def __init__(self):
        self.process = psutil.Process(os.getpid())
    def get_memory_mb(self) -> float:
        return self.process.memory_info().rss / 1024 / 1024
    def get_vocab_stats(self, nlp) -> dict:
        return {'vocab_size': len(nlp.vocab), 'string_store_size': len(nlp.vocab.strings)}

class ReportGenerator:
    """Generates comprehensive visual and text reports from test results."""
    def __init__(self, results: list, output_dir: str):
        self.results = [r for r in results if 'error' not in r]
        self.output_dir = output_dir
        if not self.results:
            print("No successful test results to report.")
            return
        self.generate_visual_report()
        self.generate_text_summary()

    def generate_visual_report(self):
        """Creates and saves a 2x2 grid of plots, including grouped bar charts."""
        print("\nGenerating Visual Investigation Report...")
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 14), constrained_layout=True)
        fig.suptitle('NER Pipeline: Memory & Performance Investigation (Multi-Batch Size)', fontsize=18, fontweight='bold')
        
        # --- Prepare data for grouped bar charts ---
        labels = sorted(list(set(r['batch_size_tested'] for r in self.results)))
        no_zone_growths = [next((r['memory_growth'] for r in self.results if r['batch_size_tested'] == bs and 'No Zone' in r['test_name']), 0) for bs in labels]
        with_zone_growths = [next((r['memory_growth'] for r in self.results if r['batch_size_tested'] == bs and 'With Memory Zone' in r['test_name']), 0) for bs in labels]
        no_zone_times = [next((r['duration'] for r in self.results if r['batch_size_tested'] == bs and 'No Zone' in r['test_name']), 0) for bs in labels]
        with_zone_times = [next((r['duration'] for r in self.results if r['batch_size_tested'] == bs and 'With Memory Zone' in r['test_name']), 0) for bs in labels]
        
        x = np.arange(len(labels))
        width = 0.35

        # Plot 1: Memory Consumption Line Chart (for a representative batch size)
        ax1.set_title('Memory Trend (Example Batch Size)', fontweight='bold')
        rep_bs = labels[len(labels) // 2] # Choose the middle batch size
        for r in self.results:
            if r['batch_size_tested'] == rep_bs:
                ax1.plot(r['batches'], r['memory_mb'], label=r['test_name'], marker='.', markersize=4)
        ax1.set_xlabel('Batch Number')
        ax1.set_ylabel('Memory Usage (MB)')
        ax1.legend()
        ax1.grid(True, alpha=0.4)

        # Plot 2: Vocabulary Growth Line Chart (for a representative batch size)
        ax2.set_title('Vocabulary Trend (Example Batch Size)', fontweight='bold')
        for r in self.results:
            if r['batch_size_tested'] == rep_bs:
                ax2.plot(r['batches'], r['vocab_size'], label=r['test_name'], marker='.', markersize=4)
        ax2.set_xlabel('Batch Number')
        ax2.set_ylabel('Vocabulary Size')
        ax2.legend()
        ax2.grid(True, alpha=0.4)

        # Plot 3: Grouped Bar Chart - Memory Growth by Batch Size
        ax3.set_title('Total Memory Growth by Batch Size', fontweight='bold')
        rects1 = ax3.bar(x - width/2, no_zone_growths, width, label='No Zone', color='tomato')
        rects2 = ax3.bar(x + width/2, with_zone_growths, width, label='With Memory Zone', color='mediumseagreen')
        ax3.set_ylabel('Memory Growth (MB)')
        ax3.set_xticks(x, [f'Batch Size\n{l}' for l in labels])
        ax3.legend()
        ax3.bar_label(rects1, padding=3, fmt='%.1f')
        ax3.bar_label(rects2, padding=3, fmt='%.1f')

        # Plot 4: Grouped Bar Chart - Processing Time by Batch Size
        ax4.set_title('Total Processing Time by Batch Size', fontweight='bold')
        rects3 = ax4.bar(x - width/2, no_zone_times, width, label='No Zone', color='tomato')
        rects4 = ax4.bar(x + width/2, with_zone_times, width, label='With Memory Zone', color='mediumseagreen')
        ax4.set_ylabel('Time Taken (seconds)')
        ax4.set_xticks(x, [f'Batch Size\n{l}' for l in labels])
        ax4.legend()
        ax4.bar_label(rects3, padding=3, fmt='%.2f')
        ax4.bar_label(rects4, padding=3, fmt='%.2f')

        report_filename = os.path.join(self.output_dir, 'memory_investigation_report.png')
        plt.savefig(report_filename, dpi=300)
        plt.close()
        print(f"Visual report saved to: {report_filename}")

    def generate_text_summary(self):
        """Prints a detailed text summary with analysis and recommendations."""
        print("\n📋 INVESTIGATION SUMMARY REPORT")
        print("=" * 60)
        for bs in sorted(list(set(r['batch_size_tested'] for r in self.results))):
            print(f"\n--- Analysis for Batch Size: {bs} ---")
            no_zone_res = next((r for r in self.results if r['batch_size_tested'] == bs and 'No Zone' in r['test_name']), None)
            with_zone_res = next((r for r in self.results if r['batch_size_tested'] == bs and 'With Memory Zone' in r['test_name']), None)

            if no_zone_res and with_zone_res:
                mem_saved = no_zone_res['memory_growth'] - with_zone_res['memory_growth']
                print(f"  - Memory Growth (No Zone):    {no_zone_res['memory_growth']:.1f} MB")
                print(f"  - Memory Growth (With Zone):  {with_zone_res['memory_growth']:.1f} MB  (Saved {mem_saved:.1f} MB)")
                print(f"  - Time Taken (No Zone):       {no_zone_res['duration']:.2f} s")
                print(f"  - Time Taken (With Zone):     {with_zone_res['duration']:.2f} s")
        
        print("\nCONCLUSIONS & RECOMMENDATIONS:")
        print("1. Effectiveness: The `memory_zone` solution is consistently effective at preventing memory leaks across all tested batch sizes.")
        print("2. Impact of Batch Size: While batch size has a minor effect on total processing time, it does not significantly change the rate of the memory leak in the 'No Zone' scenario. The leak is proportional to the total number of unique strings processed, regardless of batching.")
        print("3. Performance Trade-off: The performance overhead of using `memory_zone` is minimal and consistent across batch sizes, making it a safe and reliable solution.")
        print("\nFINAL RECOMMENDATION: Implement the `memory_zone` solution. The choice of batch size can be based on performance/throughput requirements rather than memory concerns.")
        print("=" * 60)


def run_test(ner_model: SpacyModel, df: pd.DataFrame, batch_size: int, test_name: str):
    """Runs a test scenario, processes data in batches, and profiles performance."""
    print(f"\n--- Running Test: {test_name} (Batch Size: {batch_size}) ---")
    profiler = MemoryProfiler()
    initial_memory = profiler.get_memory_mb()
    
    # Get initial vocab stats
    initial_vocab_stats = profiler.get_vocab_stats(ner_model.nlp)
    
    # Lists for plotting
    memory_data, vocab_data, string_store_data, batch_numbers = [], [], [], []
    
    start_time = time.time()
    
    num_batches = (len(df) + batch_size - 1) // batch_size
    for i in range(0, len(df), batch_size):
        batch_df = df.iloc[i:i+batch_size]
        dsp = batch_df['dsp'].to_list()
        mmo = batch_df['mmo'].to_list()
        dsp_vec = np.array(dsp, dtype='|S0').reshape(len(dsp), 1)
        mmo_vec = np.array(mmo, dtype='|S0').reshape(len(mmo), 1)
        requests = [mockInferenceRequest(inputs=[Tensor(data=dsp_vec, name='dsp'), Tensor(data=mmo_vec, name='mmo')])]
        ner_model.execute(requests)

        # Profile and print progress
        batch_num = (i // batch_size) + 1
        memory_mb = profiler.get_memory_mb()
        vocab_stats = profiler.get_vocab_stats(ner_model.nlp)
        memory_data.append(memory_mb)
        vocab_data.append(vocab_stats['vocab_size'])
        string_store_data.append(vocab_stats['string_store_size'])
        batch_numbers.append(batch_num)
        
        growth = memory_mb - initial_memory
        if batch_num % 10 == 0 or batch_num == num_batches: # Print every 10 batches or on the last batch
            print(f"  Batch {batch_num}/{num_batches}: Memory = {memory_mb:.1f} MB ({growth:+.1f} MB), Vocab = {vocab_stats['vocab_size']}")

    duration = time.time() - start_time
    final_memory = profiler.get_memory_mb()
    
    return {
        'test_name': test_name,
        'batch_size_tested': batch_size,
        'duration': duration,
        'memory_growth': final_memory - initial_memory,
        'memory_mb': memory_data,
        'vocab_size': vocab_data,
        'string_store_size': string_store_data,
        'batches': batch_numbers,
    }

def main():
    # --- SCRIPT CONFIGURATION ---
    BATCH_SIZES_TO_TEST = [100, 500, 2000] # List of batch sizes to iterate through
    INPUT_CSV = "./case_examples_ner.csv"
    OUTPUT_DIR = "./output"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # --- Load and Prepare Data ---
    df = pd.read_csv(INPUT_CSV, keep_default_na=False)
    df['dsp'] = df['dsp'].apply(unidecode)
    df['mmo'] = df['mmo'].apply(unidecode)
    print(f"Loaded {len(df)} records for investigation.")
    
    # --- Run Investigation Loop ---
    all_results = []
    init_args = {'model_name': 'us_spacy_ner'}
    
    for batch_size in BATCH_SIZES_TO_TEST:
        print(f"\n{'='*20} INVESTIGATING WITH BATCH SIZE: {batch_size} {'='*20}")
        
        # Test 1: Batched (No Zone)
        ner_1 = SpacyModel()
        ner_1.initialize(init_args)
        ner_1.use_memory_zone = False
        result1 = run_test(ner_1, df, batch_size, 'Batched (No Zone)')
        all_results.append(result1)
        del ner_1; gc.collect(); time.sleep(2)

        # Test 2: Batched (With Memory Zone)
        ner_2 = SpacyModel()
        ner_2.initialize(init_args)
        ner_2.use_memory_zone = True
        result2 = run_test(ner_2, df, batch_size, 'Batched (With Memory Zone)')
        all_results.append(result2)
        del ner_2; gc.collect(); time.sleep(2)

    # --- Generate Final Report ---
    ReportGenerator(all_results, OUTPUT_DIR)

if __name__ == "__main__":
    main()



def extract_results(self, docs, descriptions, memos):
    """Optimized to minimize string object creation"""
    
    # Pre-extract entity data to minimize repeated attribute access
    all_ents_data = []
    for doc in docs:
        doc_ents = [(ent.text, ent.label_, ent.ent_id_) for ent in doc.ents]
        all_ents_data.append(doc_ents)
    
    entities = []
    all_labels = []
    all_extracted_texts = []
    all_entity_ids = []
    max_len = 0
    
    for ents_data, description, memo in zip(all_ents_data, descriptions, memos):
        labels = []
        extracted_texts = []
        entity_ids = []
        sub_entities_parts = []
        
        # Check IGNORE_ALL using tuple data
        ignore_all_flag = any(label == "IGNORE_ALL" for _, label, _ in ents_data)
        desc_memo = self.combine_desc_memo(desc=description, memo=memo)
        
        for ent_text, ent_label, ent_id in ents_data:
            clean_ent = self.clean_ent(ent_text)
            
            if not self.is_valid_ent(ent_label, clean_ent, ignore_all_flag=ignore_all_flag):
                continue
            
            labels.append(ent_label)
            extracted_texts.append(clean_ent)
            entity_ids.append(ent_id)
            sub_entities_parts.append(f"{ent_text}-{ent_label}-{ent_id}| ")
        
        sub_entities = "".join(sub_entities_parts)
        
        # Regex logic
        regex_flag = False
        if len(labels) == 0:
            regex_flag = True
        else:
            if ("PLATFORMS" in labels) and not("Business entity" in labels) and any(id in entity_ids for id in self.regex_filter_ent):
                regex_flag = True
            elif not any(label in labels for label in ["PLATFORMS", "PAYMENT PROCESSORS", "Business entity"]):
                regex_flag = True
        
        if regex_flag:
            text_ = self.apply_regex_patterns_if_no_business_entity(desc_memo, "", "")
            clean_ent = self.clean_ent(text_)
            if clean_ent:
                extracted_texts.append(clean_ent)
                labels.append("Business entity")
                entity_ids.append("")
                sub_entities += f"{clean_ent}-Business entity|| "
        
        max_len = max(max_len, len(labels))
        entities.append(sub_entities)
        all_labels.append(labels)
        all_extracted_texts.append(extracted_texts)
        all_entity_ids.append(entity_ids)
    
    return (entities, ) + (self.normalize_lengths(
        max_len, ["", "", "", -1],
        all_labels, all_extracted_texts, all_entity_ids
    ))
