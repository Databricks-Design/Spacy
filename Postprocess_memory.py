#!/usr/bin/env python3
"""
SpaCy Memory Leak Investigation Script
=====================================
Comprehensive analysis of SpaCy memory consumption patterns and solution effectiveness.
Simulates production-like workloads and generates detailed reports.
"""

import spacy
import psutil
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time
import gc
import os
import sys
from datetime import datetime
from typing import List, Dict, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

# --- All classes (MemoryProfiler, DataGenerator) remain exactly the same ---
class MemoryProfiler:
    """Memory monitoring and data collection utilities"""
    
    def __init__(self):
        self.process = psutil.Process(os.getpid())
        self.baseline_memory = self.get_memory_mb()
        
    def get_memory_mb(self) -> float:
        """Get current memory usage in MB"""
        return self.process.memory_info().rss / 1024 / 1024
    
    def get_memory_percent(self) -> float:
        """Get current memory usage as percentage of system RAM"""
        return psutil.virtual_memory().percent
    
    def get_vocab_stats(self, nlp) -> Dict[str, int]:
        """Get vocabulary and string store statistics"""
        return {
            'vocab_size': len(nlp.vocab),
            'string_store_size': len(nlp.vocab.strings),
            'vocab_vectors': len(nlp.vocab.vectors) if hasattr(nlp.vocab, 'vectors') else 0
        }

class DataGenerator:
    """Generate realistic test data mimicking financial transactions"""
    
    @staticmethod
    def generate_financial_transactions(count: int) -> List[str]:
        """Generate financial transaction-like text with unique identifiers"""
        businesses = ["ABC PHARMACY", "WALMART STORE", "AMAZON.COM", "STARBUCKS", "MCDONALD'S", 
                     "TARGET", "SHELL OIL", "COSTCO", "HOME DEPOT", "BEST BUY"]
        locations = ["NEW YORK NY", "LOS ANGELES CA", "CHICAGO IL", "HOUSTON TX", "PHOENIX AZ"]
        
        transactions = []
        for i in range(count):
            business = np.random.choice(businesses)
            location = np.random.choice(locations)
            ref_id = f"#REF{np.random.randint(100000, 999999)}"
            session_id = f"SESSION_{np.random.randint(10000, 99999)}"
            amount = f"${np.random.randint(10, 500)}.{np.random.randint(10, 99)}"
            
            transaction = f"POS DEBIT {business} {ref_id} {location} {session_id} {amount}"
            transactions.append(transaction)
        
        return transactions

class SpacyMemoryInvestigator:
    """Main investigation class for testing SpaCy memory patterns"""
    
    def __init__(self):
        self.profiler = MemoryProfiler()
        self.data_gen = DataGenerator()
        self.results = {}
        
    def test_baseline_memory_leak(self, iterations: int = 500, batch_size: int = 50) -> Dict[str, List]:
        """Test 1: Reproduce realistic production memory leak - PERSISTENT MODEL"""
        print(f"\n🔬 TEST 1: Production Memory Leak Simulation (Original SpaCy Leak)")
        # ... (This function remains unchanged) ...
        print(f"Processing {iterations} batches of {batch_size} documents each")
        print("MODEL STAYS LOADED - simulating long-running service")
        nlp = spacy.load("en_core_web_sm")
        initial_memory = self.profiler.get_memory_mb()
        memory_data, vocab_data, string_store_data, timestamps = [], [], [], []
        for i in range(iterations):
            texts = self.data_gen.generate_financial_transactions(batch_size)
            docs = list(nlp.pipe(texts))
            results = [(doc.text, [ent.text for ent in doc.ents]) for doc in docs]
            if i % 10 == 0:
                memory_mb = self.profiler.get_memory_mb()
                vocab_stats = self.profiler.get_vocab_stats(nlp)
                memory_data.append(memory_mb)
                vocab_data.append(vocab_stats['vocab_size'])
                string_store_data.append(vocab_stats['string_store_size'])
                timestamps.append(i)
                growth = memory_mb - initial_memory
                print(f"  Batch {i}: Memory={memory_mb:.1f}MB (+{growth:.1f}MB), Vocab={vocab_stats['vocab_size']}")
            time.sleep(0.02)
        final_memory = self.profiler.get_memory_mb()
        final_vocab_stats = self.profiler.get_vocab_stats(nlp)
        del nlp, docs, results
        gc.collect()
        return {'test_name': '1. SpaCy Memory Leak', 'memory_mb': memory_data, 'vocab_size': vocab_data, 'string_store_size': string_store_data, 'iterations': timestamps, 'final_memory': final_memory, 'memory_growth': final_memory - initial_memory}

    def test_memory_zone_solution(self, iterations: int = 500, batch_size: int = 50) -> Dict[str, List]:
        """Test 2: Memory zone solution (SpaCy 3.8+) - PERSISTENT MODEL"""
        print(f"\n🧪 TEST 2: Memory Zone Solution (SpaCy Fix)")
        # ... (This function remains unchanged) ...
        nlp = spacy.load("en_core_web_sm")
        if not hasattr(nlp, 'memory_zone'):
            print("  ⚠️ Memory zone not available.")
            return {}
        initial_memory = self.profiler.get_memory_mb()
        memory_data, vocab_data, string_store_data, timestamps = [], [], [], []
        for i in range(iterations):
            texts = self.data_gen.generate_financial_transactions(batch_size)
            with nlp.memory_zone():
                docs = list(nlp.pipe(texts))
                results = [(doc.text, [ent.text for ent in doc.ents]) for doc in docs]
            if i % 10 == 0:
                memory_mb = self.profiler.get_memory_mb()
                vocab_stats = self.profiler.get_vocab_stats(nlp)
                memory_data.append(memory_mb)
                vocab_data.append(vocab_stats['vocab_size'])
                string_store_data.append(vocab_stats['string_store_size'])
                timestamps.append(i)
                growth = memory_mb - initial_memory
                print(f"  Batch {i}: Memory={memory_mb:.1f}MB (+{growth:.1f}MB), Vocab={vocab_stats['vocab_size']}")
            time.sleep(0.02)
        final_memory = self.profiler.get_memory_mb()
        del nlp, docs, results
        gc.collect()
        return {'test_name': '2. SpaCy Fix (Memory Zone)', 'memory_mb': memory_data, 'vocab_size': vocab_data, 'string_store_size': string_store_data, 'iterations': timestamps, 'final_memory': final_memory, 'memory_growth': final_memory - initial_memory}

    # +++ NEW TEST FUNCTION +++
    def test_processing_memory_leak(self, iterations: int = 500, batch_size: int = 50) -> Dict[str, List]:
        """Test 3: Simulate memory leak from post-processing and data aggregation."""
        print(f"\n📈 TEST 3: Processing Memory Leak Simulation (Your Pipeline's Logic)")
        print("Using Memory Zone to isolate SpaCy. Testing if collecting DataFrames causes growth.")
        
        # Mock objects to simulate your pipeline's Pydantic models
        class MockEntity:
            def __init__(self, text, label):
                self.extracted_text = text
                self.label = label
                self.standardized_name = text.upper()

        nlp = spacy.load("en_core_web_sm")
        if not hasattr(nlp, 'memory_zone'):
            print("  ⚠️ Memory zone not available. Cannot run this test.")
            return {}
            
        initial_memory = self.profiler.get_memory_mb()
        memory_data = []
        timestamps = []
        
        # This list will store all results, simulating the aggregation in your main app
        all_processed_dataframes = []

        for i in range(iterations):
            texts = self.data_gen.generate_financial_transactions(batch_size)
            
            # --- We ALWAYS use memory_zone to ensure SpaCy is not the leak source ---
            with nlp.memory_zone():
                docs = list(nlp.pipe(texts))
                # 1. Safely extract raw data
                raw_results = [[{'text': ent.text, 'label': ent.label_} for ent in doc.ents] for doc in docs]

            # --- Simulate your post-processing OUTSIDE the zone ---
            # 2. Convert raw data to mock Pydantic objects (like create_entities)
            processed_entities = [[MockEntity(e['text'], e['label']) for e in doc_ents] for doc_ents in raw_results]
            
            # 3. Build a DataFrame for the batch (like build_df_from_results)
            batch_df = pd.DataFrame({
                'transaction_id': range(i * batch_size, (i + 1) * batch_size),
                'text': texts,
                'entities': processed_entities
            })

            # 4. CRITICAL: Aggregate the results. This is the operation we are testing.
            all_processed_dataframes.append(batch_df)

            if i % 10 == 0:
                memory_mb = self.profiler.get_memory_mb()
                memory_data.append(memory_mb)
                timestamps.append(i)
                growth = memory_mb - initial_memory
                print(f"  Batch {i}: Memory={memory_mb:.1f}MB (+{growth:.1f}MB), Total DFs collected: {len(all_processed_dataframes)}")
            
            time.sleep(0.02)

        final_memory = self.profiler.get_memory_mb()
        del nlp, all_processed_dataframes, batch_df
        gc.collect()
        
        return {
            'test_name': '3. Processing Leak (Collecting DFs)',
            'memory_mb': memory_data,
            'iterations': timestamps,
            'final_memory': final_memory,
            'memory_growth': final_memory - initial_memory,
        }

    def run_investigation(self): # Modified to run the new test
        """Run complete investigation with all tests"""
        print("🔍 SPACY MEMORY LEAK INVESTIGATION")
        print("=" * 50)
        
        # --- ADDED THE NEW TEST TO THE LIST ---
        tests = [
            self.test_baseline_memory_leak,
            self.test_memory_zone_solution,
            self.test_processing_memory_leak, # Our new test
        ]
        
        all_results = []
        for test_func in tests:
            result = test_func()
            if result:
                all_results.append(result)
            gc.collect()
            time.sleep(1)
        
        self.generate_report(all_results)
        return all_results

    def generate_report(self, results: List[Dict]):
        """Generate comprehensive visual report (Modified to not require vocab)"""
        print(f"\n📊 GENERATING INVESTIGATION REPORT")
        successful_results = [r for r in results if 'error' not in r and 'memory_mb' in r and r['memory_mb']]
        if not successful_results:
            print("❌ No successful tests to report")
            return

        # --- Report generation is now more flexible ---
        num_plots = 2 
        fig, axes = plt.subplots(1, num_plots, figsize=(8 * num_plots, 6))
        fig.suptitle('Memory Investigation Report', fontsize=16, fontweight='bold')
        
        # Plot 1: Memory consumption over time
        ax1 = axes[0]
        ax1.set_title('Memory Consumption Over Time', fontweight='bold')
        for result in successful_results:
            ax1.plot(result['iterations'], result['memory_mb'], label=result['test_name'], marker='o', markersize=3)
        ax1.set_xlabel('Batch Number')
        ax1.set_ylabel('Total Memory Usage (MB)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Summary comparison
        ax2 = axes[1]
        ax2.set_title('Final Memory Growth Comparison', fontweight='bold')
        test_names = [r['test_name'] for r in successful_results]
        memory_growths = [r['memory_growth'] for r in successful_results]
        colors = plt.cm.viridis(np.linspace(0, 1, len(test_names)))
        bars = ax2.bar(test_names, memory_growths, color=colors)
        ax2.set_ylabel('Total Memory Growth (MB)')
        plt.setp(ax2.get_xticklabels(), rotation=15, ha="right")
        for bar, growth in zip(bars, memory_growths):
            height = bar.get_height()
            ax2.annotate(f'{growth:.1f}MB', xy=(bar.get_x() + bar.get_width() / 2, height), xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        report_filename = 'memory_investigation_report.png'
        plt.savefig(report_filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"📊 Visual report saved as: {report_filename}")


def main():
    """Main execution function"""
    investigator = SpacyMemoryInvestigator()
    results = investigator.run_investigation()

if __name__ == "__main__":
    main()
