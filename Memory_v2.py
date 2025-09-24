import psutil
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
                         batch_size: int = 500, iterations: int = 10, output_path: str = ''):
    print("Running memory comparison test...")
    
    profiler = MemoryProfiler()
    
    print("Single shot processing...")
    single_start = time.time()
    test_df = df.iloc[:batch_size * iterations].copy() if len(df) >= batch_size * iterations else df.copy()
    result_single = predict_on_df(test_df, spacy_model_name=spacy_model_name, use_memory_zone=True, use_abc=False)
    single_time = time.time() - single_start
    
    nlp_check = spacy.load(spacy_model_name)
    single_memory = profiler.get_memory_mb()
    single_vocab_stats = profiler.get_vocab_stats(nlp_check)
    del nlp_check
    gc.collect()
    
    print(f"Single shot: {single_time:.2f}s, Memory: {single_memory:.1f}MB, Vocab: {single_vocab_stats['vocab_size']}, StringStore: {single_vocab_stats['string_store_size']}")
    
    print("\nBatch processing WITHOUT memory zone...")
    batch_start = time.time()
    nlp_without = spacy.load(spacy_model_name)
    initial_memory = profiler.get_memory_mb()
    initial_vocab = profiler.get_vocab_stats(nlp_without)
    
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
            del docs
    
    batch_without_time = time.time() - batch_start
    final_memory_without = profiler.get_memory_mb()
    final_vocab_without = profiler.get_vocab_stats(nlp_without)
    del nlp_without
    gc.collect()
    
    print(f"Batch without zone: {batch_without_time:.2f}s, Memory: {final_memory_without:.1f}MB (+{final_memory_without - initial_memory:.1f}MB), Vocab: {final_vocab_without['vocab_size']} (+{final_vocab_without['vocab_size'] - initial_vocab['vocab_size']}), StringStore: {final_vocab_without['string_store_size']}")
    
    print("\nBatch processing WITH memory zone...")
    batch_with_start = time.time()
    nlp_with = spacy.load(spacy_model_name)
    
    if not hasattr(nlp_with, 'memory_zone'):
        print("Memory zone not available")
        del nlp_with
        return
    
    initial_memory_with = profiler.get_memory_mb()
    initial_vocab_with = profiler.get_vocab_stats(nlp_with)
    
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
                del docs
    
    batch_with_time = time.time() - batch_with_start
    final_memory_with = profiler.get_memory_mb()
    final_vocab_with = profiler.get_vocab_stats(nlp_with)
    del nlp_with
    gc.collect()
    
    print(f"Batch with zone: {batch_with_time:.2f}s, Memory: {final_memory_with:.1f}MB (+{final_memory_with - initial_memory_with:.1f}MB), Vocab: {final_vocab_with['vocab_size']} (+{final_vocab_with['vocab_size'] - initial_vocab_with['vocab_size']}), StringStore: {final_vocab_with['string_store_size']}")
    
    print(f"\nComparison:")
    print(f"Memory savings: {final_memory_without - final_memory_with:.1f}MB")
    print(f"Vocab savings: {final_vocab_without['vocab_size'] - final_vocab_with['vocab_size']} tokens")
    print(f"Time overhead: {((batch_without_time / single_time - 1) * 100):+.1f}%")
    
    if output_path:
        report_path = os.path.join(output_path, f'memory_comparison_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')
        with open(report_path, 'w') as f:
            f.write(f"Memory Zone Comparison Report\n")
            f.write(f"Generated: {datetime.now()}\n\n")
            f.write(f"Dataset size: {len(df)} transactions\n")
            f.write(f"Batch size: {batch_size}, Iterations: {iterations}\n\n")
            f.write(f"Single shot: {single_time:.2f}s, Memory: {single_memory:.1f}MB, Vocab: {single_vocab_stats['vocab_size']}, StringStore: {single_vocab_stats['string_store_size']}\n")
            f.write(f"Batch without zone: {batch_without_time:.2f}s, Memory: {final_memory_without:.1f}MB, Vocab: {final_vocab_without['vocab_size']}, StringStore: {final_vocab_without['string_store_size']}\n")
            f.write(f"Batch with zone: {batch_with_time:.2f}s, Memory: {final_memory_with:.1f}MB, Vocab: {final_vocab_with['vocab_size']}, StringStore: {final_vocab_with['string_store_size']}\n")
            f.write(f"Memory savings: {final_memory_without - final_memory_with:.1f}MB\n")
            f.write(f"Vocab savings: {final_vocab_without['vocab_size'] - final_vocab_with['vocab_size']} tokens\n")
        print(f"Report saved: {report_path}")

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
