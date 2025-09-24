# ADD 
import psutil
import os

class MemoryProfiler:
    def __init__(self):
        self.process = psutil.Process(os.getpid())
    
    def get_memory_mb(self):
        return self.process.memory_info().rss / 1024 / 1024
    
    def get_vocab_stats(self, nlp):
        return {
            'vocab_size': len(nlp.vocab),
            'string_store_size': len(nlp.vocab.strings)
        }

memory_profiler = MemoryProfiler()

# function - ADD THREE NEW PARAMETERS
def predict_on_df(df: pd.DataFrame,
                 abc_lut: PhraseLut = None,
                 spacy_model: spacy.Language = None,
                 spacy_model_name: str = 'en_obbere',
                 use_abc: bool = False,
                 exit_early: bool = True,
                 id_col: str = None,
                 source_col: str = 'result_source',
                 xyz_col: str = 'clean_xyz',
                 include_confidence_scores: bool = False,
                 batch_size: int = 100,
                 track_memory: bool = True,
                 use_memory_zone: bool = True) -> pd.DataFrame:
    
    if id_col is None:
        ids = [i + 1 for i in range(len(df))]
    else:
        ids = df[id_col].tolist()
    
    if spacy_model is None:
        if use_abc:
            if abc_lut is None:
                spacy_model, abc_lut = build_abc_and_spacy_model(spacy_model_name)
            else:
                spacy_model, _ = build_abc_and_spacy_model(spacy_model_name)
        else:
            spacy_model = spacy.load(spacy_model_name)
            abc_lut = None

    # Memory tracking start
    if track_memory:
        initial_memory = memory_profiler.get_memory_mb()
        initial_vocab_stats = memory_profiler.get_vocab_stats(spacy_model)
        print(f"START - Memory: {initial_memory:.1f}MB, Vocab: {initial_vocab_stats['vocab_size']}, StringStore: {initial_vocab_stats['string_store_size']}")

    # Process in batches if DataFrame is large
    if len(df) > batch_size:
        all_results = []
        for batch_idx, start_idx in enumerate(range(0, len(df), batch_size)):
            batch_df = df.iloc[start_idx:start_idx + batch_size]
            batch_ids = batch_df[id_col].tolist() if id_col else [i + 1 for i in range(start_idx, start_idx + batch_size)]
            texts = batch_df[xyz_col].tolist()
            
            # Process batch with memory zone
            has_memory_zone = hasattr(spacy_model, 'memory_zone') and use_memory_zone
            if has_memory_zone:
                with spacy_model.memory_zone():
                    result_docs = spacy_model.pipe(texts)
                    if include_confidence_scores:
                        confidence_scores = [get_confidence_score(text, spacy_model) for text in texts]
                    else:
                        confidence_scores = [dict()] * len(batch_df)
                    
                    entity_enrichments = [
                        create_entities(doc, confidence_score, txn_id)
                        for txn_id, doc, confidence_score in zip(batch_ids, result_docs, confidence_scores)
                    ]
            else:
                result_docs = spacy_model.pipe(texts)
                if include_confidence_scores:
                    confidence_scores = [get_confidence_score(text, spacy_model) for text in texts]
                else:
                    confidence_scores = [dict()] * len(batch_df)
                
                entity_enrichments = [
                    create_entities(doc, confidence_score, txn_id)
                    for txn_id, doc, confidence_score in zip(batch_ids, result_docs, confidence_scores)
                ]
            
            batch_results = build_df_from_results(batch_ids, entity_enrichments)
            batch_df['transactionId'] = batch_ids
            batch_results = batch_df.merge(batch_results, on='transactionId')
            batch_results[source_col] = ['spacy'] * len(batch_results)
            all_results.append(batch_results)
            
            # Memory tracking after batch
            if track_memory:
                current_memory = memory_profiler.get_memory_mb()
                current_vocab_stats = memory_profiler.get_vocab_stats(spacy_model)
                memory_growth = current_memory - initial_memory
                vocab_growth = current_vocab_stats['vocab_size'] - initial_vocab_stats['vocab_size']
                print(f"Batch {batch_idx}: Memory={current_memory:.1f}MB (+{memory_growth:.1f}MB), Vocab={current_vocab_stats['vocab_size']} (+{vocab_growth}), StringStore={current_vocab_stats['string_store_size']}")
        
        results_df = pd.concat(all_results, ignore_index=True)
    else:
        # Original single batch processing with memory zone
        texts = df[xyz_col].tolist()
        
        has_memory_zone = hasattr(spacy_model, 'memory_zone') and use_memory_zone
        if has_memory_zone:
            with spacy_model.memory_zone():
                result_docs = spacy_model.pipe(texts)
                result_sources = ['spacy'] * len(texts)
                if include_confidence_scores:
                    confidence_scores = [get_confidence_score(text, spacy_model) for text in texts]
                else:
                    confidence_scores = [dict()] * len(df)
                
                entity_enrichments = [
                    create_entities(doc, confidence_score, txn_id)
                    for txn_id, doc, confidence_score in zip(ids, result_docs, confidence_scores)
                ]
        else:
            result_docs = spacy_model.pipe(texts)
            result_sources = ['spacy'] * len(texts)
            if include_confidence_scores:
                confidence_scores = [get_confidence_score(text, spacy_model) for text in texts]
            else:
                confidence_scores = [dict()] * len(df)
            
            entity_enrichments = [
                create_entities(doc, confidence_score, txn_id)
                for txn_id, doc, confidence_score in zip(ids, result_docs, confidence_scores)
            ]

        results_df = build_df_from_results(ids, entity_enrichments)
        df['transactionId'] = ids
        results_df = df.merge(results_df, on='transactionId')
        results_df[source_col] = result_sources

    # Final memory tracking
    if track_memory:
        final_memory = memory_profiler.get_memory_mb()
        final_vocab_stats = memory_profiler.get_vocab_stats(spacy_model)
        total_growth = final_memory - initial_memory
        total_vocab_growth = final_vocab_stats['vocab_size'] - initial_vocab_stats['vocab_size']
        print(f"FINAL - Memory Growth: {total_growth:.1f}MB, Vocab Growth: {total_vocab_growth}, StringStore Final: {final_vocab_stats['string_store_size']}")

    return results_df
