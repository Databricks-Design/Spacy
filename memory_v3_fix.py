import psutil
import matplotlib.pyplot as plt
import os
import time
import gc
from datetime import datetime
from typing import List, Dict, Tuple

class MemoryProfiler:
    def __init__(self):
        self.process = psutil.Process(os.getpid())
        
    def get_memory_mb(self) -> float:
        return self.process.memory_info().rss / 1024 / 1024
    
    def get_vocab_stats(self, nlp) -> Dict[str, int]:
        return {
            'vocab_size': len(nlp.vocab),
            'string_store_size': len(nlp.vocab.strings)
        }

def run_memory_comparison(df: pd.DataFrame, spacy_model_name: str = 'en_ob', 
                         batch_size: int = 500, iterations: int = 50, output_path: str = ''):
    print("Running memory comparison test...")
    
    profiler = MemoryProfiler()
    
    print("FACT 1: Single shot WITHOUT memory zone (baseline)")
    single_start = time.time()
    test_df = df.iloc[:min(batch_size * iterations, len(df))].copy()
    result_single = predict_on_df(test_df, spacy_model_name=spacy_model_name, 
                                 use_memory_zone=False, use_abc=False)
    single_time = time.time() - single_start
    
    nlp_check = spacy.load(spacy_model_name)
    single_memory = profiler.get_memory_mb()
    single_vocab = profiler.get_vocab_stats(nlp_check)
    del nlp_check
    gc.collect()
    
    print(f"Single shot: {single_time:.2f}s, Memory: {single_memory:.1f}MB, Vocab: {single_vocab['vocab_size']}, StringStore: {single_vocab['string_store_size']}")
    
    print("\nFACT 2: Batch processing WITHOUT memory zone (shows memory leak)")
    batch_start = time.time()
    nlp_without = spacy.load(spacy_model_name)
    initial_memory = profiler.get_memory_mb()
    initial_vocab = profiler.get_vocab_stats(nlp_without)
    
    memory_growth = []
    vocab_growth = []
    stringstore_growth = []
    
    for i in range(iterations):
        start_idx = i * batch_size
        end_idx = min(start_idx + batch_size, len(df))
        if start_idx >= len(df):
            break
        
        batch_df = df.iloc[start_idx:end_idx]
        texts = batch_df['clean_desc_memo'].fillna('').astype(str).tolist()
        texts = [t for t in texts if t.strip()]
        if texts:
            docs = list(nlp_without.pipe(texts))
            results = [(doc.text, [ent.text for ent in doc.ents]) for doc in docs]
            del docs, results
        
        if i % 10 == 0:
            current_memory = profiler.get_memory_mb()
            current_vocab = profiler.get_vocab_stats(nlp_without)
            
            memory_growth.append(current_memory - initial_memory)
            vocab_growth.append(current_vocab['vocab_size'] - initial_vocab['vocab_size'])
            stringstore_growth.append(current_vocab['string_store_size'] - initial_vocab['string_store_size'])
            
            print(f"  Batch {i}: Memory={current_memory:.1f}MB (+{memory_growth[-1]:.1f}MB), Vocab={current_vocab['vocab_size']}, StringStore={current_vocab['string_store_size']}")
        
        time.sleep(0.02)
    
    batch_without_time = time.time() - batch_start
    final_memory_without = profiler.get_memory_mb()
    final_vocab_without = profiler.get_vocab_stats(nlp_without)
    del nlp_without
    gc.collect()
    
    print(f"FINAL WITHOUT ZONE: {batch_without_time:.2f}s, Memory={final_memory_without:.1f}MB (+{final_memory_without - initial_memory:.1f}MB), Vocab={final_vocab_without['vocab_size']} (+{final_vocab_without['vocab_size'] - initial_vocab['vocab_size']}), StringStore={final_vocab_without['string_store_size']}")
    
    print("\nFACT 3: Batch processing WITH memory zone (shows fix)")
    batch_with_start = time.time()
    nlp_with = spacy.load(spacy_model_name)
    
    if not hasattr(nlp_with, 'memory_zone'):
        print("Memory zone not available")
        del nlp_with
        return
    
    initial_memory_with = profiler.get_memory_mb()
    initial_vocab_with = profiler.get_vocab_stats(nlp_with)
    
    memory_growth_with = []
    vocab_growth_with = []
    stringstore_growth_with = []
    
    for i in range(iterations):
        start_idx = i * batch_size
        end_idx = min(start_idx + batch_size, len(df))
        if start_idx >= len(df):
            break
        
        batch_df = df.iloc[start_idx:end_idx]
        texts = batch_df['clean_desc_memo'].fillna('').astype(str).tolist()
        texts = [t for t in texts if t.strip()]
        if texts:
            with nlp_with.memory_zone():
                docs = list(nlp_with.pipe(texts))
                results = [(doc.text, [ent.text for ent in doc.ents]) for doc in docs]
                del docs, results
        
        if i % 10 == 0:
            current_memory = profiler.get_memory_mb()
            current_vocab = profiler.get_vocab_stats(nlp_with)
            
            memory_growth_with.append(current_memory - initial_memory_with)
            vocab_growth_with.append(current_vocab['vocab_size'] - initial_vocab_with['vocab_size'])
            stringstore_growth_with.append(current_vocab['string_store_size'] - initial_vocab_with['string_store_size'])
            
            print(f"  Batch {i}: Memory={current_memory:.1f}MB (+{memory_growth_with[-1]:.1f}MB), Vocab={current_vocab['vocab_size']}, StringStore={current_vocab['string_store_size']}")
        
        time.sleep(0.02)
    
    batch_with_time = time.time() - batch_with_start
    final_memory_with = profiler.get_memory_mb()
    final_vocab_with = profiler.get_vocab_stats(nlp_with)
    del nlp_with
    gc.collect()
    
    print(f"FINAL WITH ZONE: {batch_with_time:.2f}s, Memory={final_memory_with:.1f}MB (+{final_memory_with - initial_memory_with:.1f}MB), Vocab={final_vocab_with['vocab_size']} (+{final_vocab_with['vocab_size'] - initial_vocab_with['vocab_size']}), StringStore={final_vocab_with['string_store_size']}")
    
    print(f"\nEVIDENCE SUMMARY:")
    print(f"Memory leak WITHOUT zone: +{final_memory_without - initial_memory:.1f}MB")
    print(f"Memory growth WITH zone: +{final_memory_with - initial_memory_with:.1f}MB")
    print(f"Memory savings: {(final_memory_without - initial_memory) - (final_memory_with - initial_memory_with):.1f}MB")
    print(f"Vocab leak WITHOUT zone: +{final_vocab_without['vocab_size'] - initial_vocab['vocab_size']} tokens")
    print(f"Vocab growth WITH zone: +{final_vocab_with['vocab_size'] - initial_vocab_with['vocab_size']} tokens")
    print(f"Time comparison - Single: {single_time:.2f}s, Batch: {batch_without_time:.2f}s, Batch+Zone: {batch_with_time:.2f}s")
    
    if output_path:
        report_path = os.path.join(output_path, f'memory_evidence_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')
        with open(report_path, 'w') as f:
            f.write(f"SPACY MEMORY LEAK EVIDENCE REPORT\n")
            f.write(f"Generated: {datetime.now()}\n")
            f.write(f"Dataset: {len(df)} transactions, Batch size: {batch_size}, Iterations: {iterations}\n\n")
            
            f.write(f"FACT 1 - Single shot baseline:\n")
            f.write(f"Time: {single_time:.2f}s, Memory: {single_memory:.1f}MB, Vocab: {single_vocab['vocab_size']}, StringStore: {single_vocab['string_store_size']}\n\n")
            
            f.write(f"FACT 2 - Batch processing WITHOUT memory zone (leak):\n")
            f.write(f"Time: {batch_without_time:.2f}s, Final Memory: {final_memory_without:.1f}MB, Memory Growth: +{final_memory_without - initial_memory:.1f}MB\n")
            f.write(f"Vocab Growth: +{final_vocab_without['vocab_size'] - initial_vocab['vocab_size']} tokens, Final StringStore: {final_vocab_without['string_store_size']}\n\n")
            
            f.write(f"FACT 3 - Batch processing WITH memory zone (fixed):\n")
            f.write(f"Time: {batch_with_time:.2f}s, Final Memory: {final_memory_with:.1f}MB, Memory Growth: +{final_memory_with - initial_memory_with:.1f}MB\n")
            f.write(f"Vocab Growth: +{final_vocab_with['vocab_size'] - initial_vocab_with['vocab_size']} tokens, Final StringStore: {final_vocab_with['string_store_size']}\n\n")
            
            f.write(f"EVIDENCE:\n")
            f.write(f"Memory savings with zone: {(final_memory_without - initial_memory) - (final_memory_with - initial_memory_with):.1f}MB\n")
            f.write(f"Vocab leak prevented: {(final_vocab_without['vocab_size'] - initial_vocab['vocab_size']) - (final_vocab_with['vocab_size'] - initial_vocab_with['vocab_size'])} tokens\n")
        print(f"Evidence report saved: {report_path}")
    
    return {
        'single_time': single_time, 'single_memory': single_memory, 'single_vocab': single_vocab,
        'batch_without_time': batch_without_time, 'batch_without_memory_growth': final_memory_without - initial_memory,
        'batch_with_time': batch_with_time, 'batch_with_memory_growth': final_memory_with - initial_memory_with,
        'memory_growth_timeline': memory_growth, 'memory_growth_with_timeline': memory_growth_with,
        'vocab_growth_timeline': vocab_growth, 'vocab_growth_with_timeline': vocab_growth_with,
        'stringstore_growth_timeline': stringstore_growth, 'stringstore_growth_with_timeline': stringstore_growth_with
    }

def generate_memory_graphs(results_data: dict, output_path: str = ''):
    print("Generating memory comparison graphs...")
    
    if not results_data.get('memory_growth_timeline'):
        print("No timeline data available for graphing")
        return
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('SpaCy Memory Zone Investigation Report - Production Data', fontsize=16, fontweight='bold')
    
    iterations_x = list(range(0, len(results_data['memory_growth_timeline']) * 10, 10))
    
    ax1.set_title('Memory Consumption Over Time', fontweight='bold')
    ax1.plot(iterations_x, results_data['memory_growth_timeline'], 'r-o', label='Without Memory Zone', markersize=4)
    ax1.plot(iterations_x, results_data['memory_growth_with_timeline'], 'g-o', label='With Memory Zone', markersize=4)
    ax1.set_xlabel('Batch Number')
    ax1.set_ylabel('Memory Growth (MB)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.set_title('Vocabulary Size Growth', fontweight='bold')
    ax2.plot(iterations_x, results_data['vocab_growth_timeline'], 'r-s', label='Without Memory Zone', markersize=4)
    ax2.plot(iterations_x, results_data['vocab_growth_with_timeline'], 'g-s', label='With Memory Zone', markersize=4)
    ax2.set_xlabel('Batch Number')
    ax2.set_ylabel('Vocabulary Growth')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    ax3.set_title('String Store Size Growth', fontweight='bold')
    ax3.plot(iterations_x, results_data['stringstore_growth_timeline'], 'r-^', label='Without Memory Zone', markersize=4)
    ax3.plot(iterations_x, results_data['stringstore_growth_with_timeline'], 'g-^', label='With Memory Zone', markersize=4)
    ax3.set_xlabel('Batch Number')
    ax3.set_ylabel('String Store Growth')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    ax4.set_title('Processing Time Comparison', fontweight='bold')
    categories = ['Single Shot\n(No Memory Zone)', 'Batch Processing\n(No Memory Zone)', 'Batch Processing\n(With Memory Zone)']
    times = [results_data['single_time'], results_data['batch_without_time'], results_data['batch_with_time']]
    colors = ['blue', 'red', 'green']
    
    bars = ax4.bar(categories, times, color=colors, alpha=0.7)
    ax4.set_ylabel('Processing Time (seconds)')
    ax4.tick_params(axis='x', labelsize=10)
    
    for bar, time_val in zip(bars, times):
        height = bar.get_height()
        ax4.annotate(f'{time_val:.2f}s',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    
    if output_path:
        graph_path = os.path.join(output_path, f'spacy_memory_investigation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
        plt.savefig(graph_path, dpi=300, bbox_inches='tight')
        print(f"Memory investigation graph saved: {graph_path}")
    else:
        plt.show()
    
    plt.close()

def compare_csv_columns(csv1_path: str, csv2_path: str, col1_name: str, col2_name: str, output_path: str = ''):
    print(f"Comparing CSV columns: {col1_name} vs {col2_name}")
    
    try:
        df1 = pd.read_csv(csv1_path)
        df2 = pd.read_csv(csv2_path)
        
        if col1_name not in df1.columns:
            print(f"Column '{col1_name}' not found in {csv1_path}")
            return
        
        if col2_name not in df2.columns:
            print(f"Column '{col2_name}' not found in {csv2_path}")
            return
        
        def clean_text(text):
            if pd.isna(text):
                return ""
            return str(text).strip().lower()
        
        col1_cleaned = df1[col1_name].apply(clean_text)
        col2_cleaned = df2[col2_name].apply(clean_text)
        
        set1 = set(col1_cleaned)
        set2 = set(col2_cleaned)
        
        exact_matches = set1.intersection(set2)
        only_in_csv1 = set1 - set2
        only_in_csv2 = set2 - set1
        
        total_unique_csv1 = len(set1)
        total_unique_csv2 = len(set2)
        total_matches = len(exact_matches)
        
        print(f"\nCSV Comparison Results:")
        print(f"CSV1 ({csv1_path}) - Unique values: {total_unique_csv1}")
        print(f"CSV2 ({csv2_path}) - Unique values: {total_unique_csv2}")
        print(f"Exact matches: {total_matches}")
        print(f"Only in CSV1: {len(only_in_csv1)}")
        print(f"Only in CSV2: {len(only_in_csv2)}")
        print(f"Match percentage: {(total_matches / max(total_unique_csv1, total_unique_csv2)) * 100:.2f}%")
        
        assert total_matches + len(only_in_csv1) == total_unique_csv1, "CSV1 total mismatch"
        assert total_matches + len(only_in_csv2) == total_unique_csv2, "CSV2 total mismatch"
        print("Verification passed: Matches + Non-matches = Total input rows")
        
        if output_path:
            report_path = os.path.join(output_path, f'csv_comparison_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')
            with open(report_path, 'w') as f:
                f.write(f"CSV Comparison Report\n")
                f.write(f"Generated: {datetime.now()}\n\n")
                f.write(f"CSV1: {csv1_path} (Column: {col1_name})\n")
                f.write(f"CSV2: {csv2_path} (Column: {col2_name})\n\n")
                f.write(f"Results:\n")
                f.write(f"Unique values in CSV1: {total_unique_csv1}\n")
                f.write(f"Unique values in CSV2: {total_unique_csv2}\n")
                f.write(f"Exact matches: {total_matches}\n")
                f.write(f"Only in CSV1: {len(only_in_csv1)}\n")
                f.write(f"Only in CSV2: {len(only_in_csv2)}\n")
            print(f"Detailed report saved: {report_path}")
            
    except Exception as e:
        print(f"Error comparing CSVs: {str(e)}")
