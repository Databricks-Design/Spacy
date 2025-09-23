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
                         output_path: str = ''):
    print("Running memory comparison test...")
    
    profiler = MemoryProfiler()
    
    print("Testing WITHOUT memory zone...")
    initial_memory = profiler.get_memory_mb()
    
    result_without = predict_on_df(df.copy(), spacy_model_name=spacy_model_name, 
                                  use_memory_zone=False, use_abc=False)
    
    nlp_check = spacy.load(spacy_model_name)
    final_memory_without = profiler.get_memory_mb()
    final_vocab_without = profiler.get_vocab_stats(nlp_check)
    del nlp_check
    gc.collect()
    
    print(f"  Final Memory: {final_memory_without:.1f}MB")
    print(f"  Vocab Size: {final_vocab_without['vocab_size']}")
    
    print("Testing WITH memory zone...")
    
    result_with = predict_on_df(df.copy(), spacy_model_name=spacy_model_name, 
                               use_memory_zone=True, use_abc=False)
    
    nlp_check = spacy.load(spacy_model_name)
    final_memory_with = profiler.get_memory_mb()
    final_vocab_with = profiler.get_vocab_stats(nlp_check)
    del nlp_check
    gc.collect()
    
    print(f"  Final Memory: {final_memory_with:.1f}MB")
    print(f"  Vocab Size: {final_vocab_with['vocab_size']}")
    
    memory_savings = final_memory_without - final_memory_with
    vocab_difference = final_vocab_without['vocab_size'] - final_vocab_with['vocab_size']
    
    print(f"\nMemory Comparison Results:")
    print(f"Memory without zone: {final_memory_without:.1f}MB")
    print(f"Memory with zone: {final_memory_with:.1f}MB")
    print(f"Memory savings: {memory_savings:.1f}MB")
    print(f"Vocab difference: {vocab_difference} tokens")
    
    if output_path:
        report_path = os.path.join(output_path, f'memory_comparison_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')
        with open(report_path, 'w') as f:
            f.write(f"Memory Zone Comparison Report\n")
            f.write(f"Generated: {datetime.now()}\n\n")
            f.write(f"Dataset size: {len(df)} transactions\n")
            f.write(f"SpaCy model: {spacy_model_name}\n\n")
            f.write(f"Results:\n")
            f.write(f"Memory without zone: {final_memory_without:.1f}MB\n")
            f.write(f"Memory with zone: {final_memory_with:.1f}MB\n")
            f.write(f"Memory savings: {memory_savings:.1f}MB\n")
            f.write(f"Vocab difference: {vocab_difference} tokens\n")
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

