import os
import sys
import numpy as np
import pandas as pd
import time
import gc
import psutil
import random
from datetime import datetime

current_dir = os.getcwd()
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from packages.spacy_model import SpacyModel
from triton_python_backend_utils import Tensor
from tests.mocks import mockInferenceRequest

# ============================================================================
# CONFIG
# ============================================================================
NUM_TRANSACTIONS = 1000
BATCH_SIZE = 50
MEMORY_THRESHOLD_MB = 0.5
LEAK_REPORT_FILE = "leak_report.txt"

# ============================================================================
# SYNTHETIC DATA GENERATION
# ============================================================================

def generate_unique_transaction(iteration: int, num_unique_tokens: int = 50) -> str:
    transaction_types = ["POS", "ATM", "ONLINE", "TRANSFER", "PAYMENT", "REFUND", "WITHDRAWAL", "DEPOSIT"]
    merchants = ["AMAZON", "WALMART", "STARBUCKS", "SHELL", "MCDONALDS", "TARGET", "COSTCO", "BESTBUY", 
                 "NETFLIX", "UBER", "AIRBNB", "BOOKING", "PAYPAL", "VENMO", "SQUARE", "SPOTIFY", 
                 "APPLE", "GOOGLE", "MICROSOFT", "CHIPOTLE"]
    
    txn_id = f"TXN{iteration:010d}"
    merchant = f"{random.choice(merchants)}{random.randint(1000, 9999)}"
    amount = f"${random.uniform(5.0, 999.99):.2f}"
    card = f"CARD-{random.randint(1000, 9999)}"
    acct = f"ACCT{random.randint(1000000, 9999999)}"
    auth = f"AUTH{random.randint(100000, 999999)}"
    ref = f"REF{iteration}{random.randint(1000, 9999)}"
    merchant_id = f"MID{random.randint(100000, 999999)}"
    terminal = f"T{random.randint(1000, 9999)}"
    batch = f"B{random.randint(100, 999)}"
    trans_type = random.choice(transaction_types)
    
    unique_tokens = []
    remaining = num_unique_tokens - 11
    for i in range(remaining):
        token = f"{random.choice(['LOC', 'ID', 'CODE', 'SEQ'])}{iteration}{i}{random.randint(100, 999)}"
        unique_tokens.append(token)
    
    description_parts = [txn_id, trans_type, merchant, amount, card, acct, auth, ref, merchant_id, terminal, batch] + unique_tokens
    return " ".join(description_parts)

# ============================================================================
# LEAK TRACKING
# ============================================================================

class LeakTracker:
    def __init__(self, report_file):
        self.report_file = report_file
        self.leaks = []
        
        # Initialize report file
        with open(self.report_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write("MEMORY LEAK DETECTION REPORT\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*80 + "\n\n")
    
    def record_leak(self, batch_num, mem_before, mem_after, vocab_before, vocab_after):
        leak_info = {
            'batch': batch_num,
            'mem_before': mem_before,
            'mem_after': mem_after,
            'mem_delta': mem_after - mem_before,
            'vocab_before': vocab_before,
            'vocab_after': vocab_after,
            'vocab_delta': vocab_after - vocab_before
        }
        self.leaks.append(leak_info)
        
        # Write to file
        with open(self.report_file, 'a') as f:
            f.write(f"LEAK #{len(self.leaks)} - Batch {batch_num}\n")
            f.write(f"{'='*60}\n")
            f.write(f"Memory:     {mem_before:.2f} → {mem_after:.2f} MB (Δ {leak_info['mem_delta']:+.2f} MB)\n")
            f.write(f"Vocab:      {vocab_before:,} → {vocab_after:,} (Δ {leak_info['vocab_delta']:+,})\n")
            f.write(f"\n")
    
    def finalize(self, initial_mem, final_mem, initial_vocab, final_vocab, elapsed_time):
        with open(self.report_file, 'a') as f:
            f.write("\n" + "="*80 + "\n")
            f.write("SUMMARY\n")
            f.write("="*80 + "\n")
            f.write(f"Total leaks detected: {len(self.leaks)}\n")
            f.write(f"Memory growth: {initial_mem:.2f} → {final_mem:.2f} MB (+{final_mem - initial_mem:.2f} MB)\n")
            f.write(f"Vocab growth: {initial_vocab:,} → {final_vocab:,} (+{final_vocab - initial_vocab:,})\n")
            f.write(f"Elapsed time: {elapsed_time:.2f} seconds\n")
            f.write("="*80 + "\n")

# ============================================================================
# PROFILING TEST
# ============================================================================

def run_test_profiling(ner_model: SpacyModel, df: pd.DataFrame, batch_size: int):
    
    profiler = psutil.Process(os.getpid())
    initial_memory = profiler.memory_info().rss / 1024 / 1024
    initial_vocab = sum(1 for _ in ner_model.nlp.vocab)
    initial_stringstore = len(ner_model.nlp.vocab.strings)
    
    print("="*80)
    print("BASELINE")
    print("="*80)
    print(f"Memory: {initial_memory:.2f} MB")
    print(f"Vocab: {initial_vocab:,}")
    print(f"StringStore: {initial_stringstore:,}")
    print("="*80)
    print()
    
    tracker = LeakTracker(LEAK_REPORT_FILE)
    
    num_batches = (len(df) + batch_size - 1) // batch_size
    start_time = time.time()
    
    print(f"Processing {num_batches} batches...\n")
    
    for i in range(0, len(df), batch_size):
        batch_num = (i // batch_size) + 1
        
        batch_df = df.iloc[i:i+batch_size]
        
        descriptions = batch_df['description'].to_list()
        memo = batch_df['memo'].to_list()
        
        descriptions_vec = np.array(descriptions, dtype='|S0').reshape(len(descriptions), 1)
        memos_vec = np.array(memo, dtype='|S0').reshape(len(memo), 1)
        
        requests = [
            mockInferenceRequest(inputs=[
                Tensor(data=descriptions_vec, name='description'),
                Tensor(data=memos_vec, name='memo')
            ])
        ]
        
        # Measure before
        mem_before = profiler.memory_info().rss / 1024 / 1024
        vocab_before = sum(1 for _ in ner_model.nlp.vocab)
        
        # Execute
        raw_results = ner_model.execute(requests, ner_model.add_memory_zone)
        
        # Measure after
        mem_after = profiler.memory_info().rss / 1024 / 1024
        vocab_after = sum(1 for _ in ner_model.nlp.vocab)
        
        delta = mem_after - mem_before
        
        # Detect and record leak
        if delta > MEMORY_THRESHOLD_MB:
            print(f"🚨 LEAK - Batch {batch_num}/{num_batches}: {mem_before:.2f} → {mem_after:.2f} MB (Δ {delta:+.2f} MB)")
            tracker.record_leak(batch_num, mem_before, mem_after, vocab_before, vocab_after)
        
        # Status every 10 batches
        if batch_num % 10 == 0:
            current_mem = profiler.memory_info().rss / 1024 / 1024
            total_growth = current_mem - initial_memory
            print(f"[Batch {batch_num}/{num_batches}] Memory: {current_mem:.2f} MB | Total Growth: {total_growth:+.2f} MB | Leaks: {len(tracker.leaks)}")
        
        del batch_df, descriptions, memo, requests, raw_results
        gc.collect()
    
    elapsed_time = time.time() - start_time
    final_memory = profiler.memory_info().rss / 1024 / 1024
    final_vocab = sum(1 for _ in ner_model.nlp.vocab)
    final_stringstore = len(ner_model.nlp.vocab.strings)
    
    tracker.finalize(initial_memory, final_memory, initial_vocab, final_vocab, elapsed_time)
    
    print()
    print("="*80)
    print("FINAL")
    print("="*80)
    print(f"Memory: {final_memory:.2f} MB (Growth: {final_memory - initial_memory:+.2f} MB)")
    print(f"Vocab: {final_vocab:,} (Growth: {final_vocab - initial_vocab:+,})")
    print(f"StringStore: {final_stringstore:,} (Growth: {final_stringstore - initial_stringstore:+,})")
    print(f"Elapsed: {elapsed_time:.2f} seconds")
    print(f"Leaks detected: {len(tracker.leaks)}")
    print(f"\nLeak report saved to: {LEAK_REPORT_FILE}")
    print("="*80)

# ============================================================================
# MAIN
# ============================================================================

def main():
    os.environ['DESCRIPTORS_TO_REMOVE'] = 'LLC,PTY,INC'
    
    print("="*80)
    print("MEMORY LEAK DETECTOR")
    print("="*80)
    print(f"Transactions: {NUM_TRANSACTIONS:,}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Threshold: {MEMORY_THRESHOLD_MB} MB")
    print("="*80)
    print()
    
    # Generate synthetic data
    print("Generating synthetic data...")
    descriptions = []
    memos = []
    
    for i in range(NUM_TRANSACTIONS):
        descriptions.append(generate_unique_transaction(i))
        memos.append("")
    
    df = pd.DataFrame({'description': descriptions, 'memo': memos})
    print(f"Generated {len(df):,} transactions\n")
    
    # Initialize model
    print("Initializing SpaCy model...")
    ner = SpacyModel()
    ner.initialize({'model_name': 'us_spacy_ner'})
    ner.add_memory_zone = True
    print("Model initialized!\n")
    
    # Run profiling
    run_test_profiling(ner, df, BATCH_SIZE)
    
    print("\n" + "="*80)
    print("PROFILING COMPLETE")
    print("="*80)
    print("Check memory_profiler output above to see which function/line caused leaks")
    print(f"Leak summary saved to: {LEAK_REPORT_FILE}")
    print("="*80)


if __name__ == '__main__':
    main()
