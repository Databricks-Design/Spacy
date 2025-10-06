import spacy
import psutil
import os
import pandas as pd
from typing import List, Dict
from pathlib import Path
from unidecode import unidecode


class MemoryZoneTester:
    
    def __init__(self, model_name: str, model_version: str = "1", use_memory_zones: bool = True, batch_size: int = 100):
        MODEL_PATH = str(Path(__file__).parent.joinpath(f"models/{model_name}/{model_version}"))
        print(f"MODEL NAME: {model_name}")
        print(f"MODEL PATH: {MODEL_PATH}")
        self.nlp = spacy.load(MODEL_PATH)
        print(f"Model loaded successfully")
        self.use_memory_zones = use_memory_zones
        self.batch_size = batch_size
        self.process = psutil.Process(os.getpid())
        self.memory_stats = []
        print(f"Memory Zones: {'ENABLED' if self.use_memory_zones else 'DISABLED'}")
        print(f"Batch Size: {self.batch_size}")
    
    def get_memory_stats(self, label: str) -> Dict:
        mem_info = self.process.memory_info()
        return {
            'label': label,
            'rss_mb': mem_info.rss / 1024 / 1024,
            'vocab_size': len(self.nlp.vocab),
            'stringstore_size': len(self.nlp.vocab.strings)
        }
    
    def combine_desc_memo(self, description: str, memo: str) -> str:
        return str(description) + " " + str(memo)
    
    def process_dataframe(self, df: pd.DataFrame) -> List[List[Dict]]:
        descriptions = df['description'].tolist()
        memos = df['memo'].tolist()
        
        texts = [self.combine_desc_memo(desc, memo) for desc, memo in zip(descriptions, memos)]
        
        print(f"\nProcessing {len(texts)} texts")
        
        initial_stats = self.get_memory_stats("initial")
        self.memory_stats.append(initial_stats)
        self._print_stats(initial_stats)
        
        all_results = []
        num_batches = (len(texts) + self.batch_size - 1) // self.batch_size
        
        for batch_idx in range(0, len(texts), self.batch_size):
            batch_num = batch_idx // self.batch_size
            batch_texts = texts[batch_idx:batch_idx + self.batch_size]
            
            print(f"\nBatch {batch_num + 1}/{num_batches} (samples {batch_idx}-{batch_idx + len(batch_texts) - 1})")
            
            before_stats = self.get_memory_stats(f"before_batch_{batch_num}")
            self.memory_stats.append(before_stats)
            self._print_stats(before_stats)
            
            if self.use_memory_zones:
                with self.nlp.memory_zone():
                    docs = list(self.nlp.pipe(batch_texts))
                    batch_results = []
                    for doc in docs:
                        entities = []
                        for ent in doc.ents:
                            entities.append({
                                'text': ent.text,
                                'label': ent.label_,
                                'start': ent.start_char,
                                'end': ent.end_char,
                                'ent_id': ent.ent_id_
                            })
                        batch_results.append(entities)
            else:
                docs = list(self.nlp.pipe(batch_texts))
                batch_results = []
                for doc in docs:
                    entities = []
                    for ent in doc.ents:
                        entities.append({
                            'text': ent.text,
                            'label': ent.label_,
                            'start': ent.start_char,
                            'end': ent.end_char,
                            'ent_id': ent.ent_id_
                        })
                    batch_results.append(entities)
            
            all_results.extend(batch_results)
            
            after_stats = self.get_memory_stats(f"after_batch_{batch_num}")
            self.memory_stats.append(after_stats)
            self._print_stats(after_stats)
            
            growth_mb = after_stats['rss_mb'] - before_stats['rss_mb']
            growth_vocab = after_stats['vocab_size'] - before_stats['vocab_size']
            print(f"Growth: +{growth_mb:.2f}MB, +{growth_vocab} vocab")
        
        final_stats = self.get_memory_stats("final")
        self.memory_stats.append(final_stats)
        self._print_stats(final_stats)
        
        self._print_summary(initial_stats, final_stats)
        
        return all_results
    
    def _print_stats(self, stats: Dict):
        print(f"[{stats['label']}] RSS={stats['rss_mb']:.2f}MB | "
              f"Vocab={stats['vocab_size']:,} | StringStore={stats['stringstore_size']:,}")
    
    def _print_summary(self, initial: Dict, final: Dict):
        print("\n" + "="*70)
        print("SUMMARY")
        print("="*70)
        print(f"Initial: RSS={initial['rss_mb']:.2f}MB, Vocab={initial['vocab_size']:,}")
        print(f"Final:   RSS={final['rss_mb']:.2f}MB, Vocab={final['vocab_size']:,}")
        mem_growth = final['rss_mb'] - initial['rss_mb']
        vocab_growth = final['vocab_size'] - initial['vocab_size']
        mem_growth_pct = (mem_growth / initial['rss_mb'] * 100) if initial['rss_mb'] > 0 else 0
        vocab_growth_pct = (vocab_growth / initial['vocab_size'] * 100) if initial['vocab_size'] > 0 else 0
        print(f"Growth:  RSS=+{mem_growth:.2f}MB ({mem_growth_pct:.1f}%), "
              f"Vocab=+{vocab_growth:,} ({vocab_growth_pct:.1f}%)")
        print("="*70 + "\n")
    
    def save_stats(self, filename: str):
        df = pd.DataFrame(self.memory_stats)
        df.to_csv(filename, index=False)
        print(f"Memory stats saved to {filename}")


if __name__ == "__main__":
    
    USE_MEMORY_ZONES = True
    
    os.environ['DESCRIPTORS_TO_REMOVE'] = 'LLD,PYV,INC'
    
    CSV_PATH = "/se_examples_ner.csv"
    MODEL_NAME = "_spacy.nr"
    MODEL_VERSION = "1"
    BATCH_SIZE = 100
    
    zone_label = "WITH" if USE_MEMORY_ZONES else "WITHOUT"
    
    print(f"\n{'='*70}")
    print(f"MEMORY ZONE TEST: {zone_label} ZONES")
    print(f"{'='*70}\n")
    
    print(f"Loading data from: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH, keep_default_na=False)
    print(f"Loaded {len(df)} rows")
    print(f"Columns: {list(df.columns)}\n")
    
    if 'description' not in df.columns or 'memo' not in df.columns:
        raise ValueError("DataFrame must have 'description' and 'memo' columns")
    
    df['description'] = df['description'].apply(unidecode)
    df['memo'] = df['memo'].apply(unidecode)
    
    tester = MemoryZoneTester(MODEL_NAME, MODEL_VERSION, USE_MEMORY_ZONES, BATCH_SIZE)
    results = tester.process_dataframe(df)
    
    stats_file = f"memory_stats_{zone_label}_zones.csv"
    results_file = f"ner_results_{zone_label}_zones.csv"
    
    tester.save_stats(stats_file)
    
    df['entities'] = [str(r) for r in results]
    df.to_csv(results_file, index=False)
    print(f"Results saved to {results_file}")
    
    print(f"\n{'='*70}")
    print(f"TEST COMPLETED")
    print(f"{'='*70}\n")
