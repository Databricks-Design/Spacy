# Import Packages
import os
import json
import numpy as np
from unidecode import unidecode
from packages.spacy_model import SpacyModel
from triton_python_backend_utils import Tensor
from tests.mocks import mockInferenceRequest
"""

"""
# Set environment variables
os.environ['DESCRIPTORS_TO_REMOVE'] = 'LLD,PYV,INC'  # Descriptors to remove

import pandas as pd

# Configuration
TEST_MODE = 'full_pipeline'  # Options: 'full_pipeline' or 'isolation'
USE_MEMORY_ZONES = True      # Change to False to see memory leak
BATCH_SIZE = 100

print(f"\n{'='*70}")
print(f"NER PIPELINE TEST CONFIGURATION")
print(f"{'='*70}")
print(f"Test Mode: {TEST_MODE}")
print(f"Memory Zones: {'ENABLED' if USE_MEMORY_ZONES else 'DISABLED'}")
print(f"Batch Size: {BATCH_SIZE}")
print(f"{'='*70}\n")

# Create Spacy Model
ner = SpacyModel()

# Initialize Spacy Model
init_args = {
    'model_name': '__spacy.ner',
    'use_memory_zones': USE_MEMORY_ZONES,
    'batch_size': BATCH_SIZE
}
ner.initialize(init_args)

# Load Data (or use mock data)
print("\nLoading data...")
df = pd.read_csv("test_NER/case_examples_ner.csv", keep_default_na=False)
print(df.columns)
df['description'] = df['description'].apply(unidecode)
df['memo'] = df['memo'].apply(unidecode)
descriptions = df['description'].to_list()
memos = df['memo'].to_list()
print(f"Loaded {len(descriptions)} descriptions and {len(memos)} memos")

# Format data into requests structure (expected by model class)
assert len(descriptions) == len(memos), "The number of descriptions and memos should match!"

if TEST_MODE == 'isolation':
    texts = ner.preprocess_input(descriptions, memos)
    isolation_results, memory_stats = ner.test_isolation(texts, batch_size=BATCH_SIZE)
    
    outputs = isolation_results
    
    mem_filename = f"memory_tracking_isolation_{'WITH' if USE_MEMORY_ZONES else 'WITHOUT'}_zones.csv"
    pd.DataFrame(memory_stats).to_csv(mem_filename, index=False)
    
    if memory_stats:
        initial = memory_stats[0]
        final = memory_stats[-1]
        
        print(f"\n{'='*70}")
        print(f"MEMORY SUMMARY - ISOLATION TEST")
        print(f"{'='*70}")
        print(f"Initial State:")
        print(f"  RSS Memory: {initial['rss_mb']:.2f} MB")
        print(f"  Vocab Size: {initial['vocab_size']:,}")
        print(f"  StringStore Size: {initial['stringstore_size']:,}")
        print(f"\nFinal State:")
        print(f"  RSS Memory: {final['rss_mb']:.2f} MB")
        print(f"  Vocab Size: {final['vocab_size']:,}")
        print(f"  StringStore Size: {final['stringstore_size']:,}")
        print(f"\nGrowth:")
        mem_growth = final['rss_mb'] - initial['rss_mb']
        vocab_growth = final['vocab_size'] - initial['vocab_size']
        print(f"  RSS Memory: +{mem_growth:.2f} MB ({(mem_growth / initial['rss_mb'] * 100):.1f}%)")
        print(f"  Vocab Size: +{vocab_growth:,} entries ({(vocab_growth / initial['vocab_size'] * 100):.1f}%)")
        print(f"  StringStore: +{final['stringstore_size'] - initial['stringstore_size']:,} entries")
        print(f"{'='*70}\n")
    
    print(f"Memory stats saved to {mem_filename}")
    
elif TEST_MODE == 'full_pipeline':
    descriptions_vec = np.array(descriptions, dtype='|S0').reshape(len(descriptions), 1)
    memos_vec = np.array(memos, dtype='|S0').reshape(len(memos), 1)
    requests = [
        mockInferenceRequest(inputs=[
            Tensor(data=descriptions_vec, name='description'),
            Tensor(data=memos_vec, name='memo')
        ])
    ]
    
    # Execute NER pipeline
    raw_results = ner.execute(requests)
    
    # Format outputs to lists
    outputs = []
    for raw_result in raw_results:
        labels, extracted_texts, entity_ids = raw_result.output_tensors()
        # Check the names to make sure we have the right tensors
        assert labels.name() == 'label'
        assert extracted_texts.name() == 'extractedText'
        assert entity_ids.name() == 'entityId'
        # Turn them into numpy array objects
        labels = labels.as_numpy().tolist()
        extracted_texts = extracted_texts.as_numpy().tolist()
        entity_ids = entity_ids.as_numpy().tolist()
        # Iterate through them and get the results
        for label_list, extracted_text_list, entity_id_list in zip(labels, extracted_texts, entity_ids):
            these_outputs = []
            # Decode the binary representations
            decoded_labels = [x.decode('utf-8') for x in label_list]
            decoded_extracted_texts = [x for x in extracted_text_list]  # For some reason this doesn't need to be decoded?
            decoded_entity_ids = [x.decode('utf-8') for x in entity_id_list]
            # Add to outputs
            for label, extracted_text, entity_id in zip(decoded_labels, decoded_extracted_texts, decoded_entity_ids):
                if label != '' and label is not None:  # Add additional label filters if needed
                    these_outputs.append({
                        'entity_type': label,
                        'extracted_entity': extracted_text,
                        'standardized_entity': entity_id
                    })
            
            outputs.append(these_outputs)
    
    mem_filename = f"memory_tracking_full_pipeline_{'WITH' if USE_MEMORY_ZONES else 'WITHOUT'}_zones.csv"
    pd.DataFrame(ner.memory_stats).to_csv(mem_filename, index=False)
    
    if ner.memory_stats:
        initial = ner.memory_stats[0]
        final = ner.memory_stats[-1]
        
        print(f"\n{'='*70}")
        print(f"MEMORY SUMMARY - FULL PIPELINE")
        print(f"{'='*70}")
        print(f"Initial State:")
        print(f"  RSS Memory: {initial['rss_mb']:.2f} MB")
        print(f"  Vocab Size: {initial['vocab_size']:,}")
        print(f"  StringStore Size: {initial['stringstore_size']:,}")
        print(f"\nFinal State:")
        print(f"  RSS Memory: {final['rss_mb']:.2f} MB")
        print(f"  Vocab Size: {final['vocab_size']:,}")
        print(f"  StringStore Size: {final['stringstore_size']:,}")
        print(f"\nGrowth:")
        mem_growth = final['rss_mb'] - initial['rss_mb']
        vocab_growth = final['vocab_size'] - initial['vocab_size']
        print(f"  RSS Memory: +{mem_growth:.2f} MB ({(mem_growth / initial['rss_mb'] * 100):.1f}%)")
        print(f"  Vocab Size: +{vocab_growth:,} entries ({(vocab_growth / initial['vocab_size'] * 100):.1f}%)")
        print(f"  StringStore: +{final['stringstore_size'] - initial['stringstore_size']:,} entries")
        print(f"{'='*70}\n")
    
    print(f"Memory stats saved to {mem_filename}")

# Read results
print(f"\nFirst 5 results preview:")
print(json.dumps(outputs[:5], indent=2))
df['outputs_ner'] = outputs

output_filename = f"case_examples_scored_{TEST_MODE}_{'WITH' if USE_MEMORY_ZONES else 'WITHOUT'}_zones.csv"
df.to_csv(output_filename, index=False)
print(f"\nResults saved to {output_filename}")

print(f"\n{'='*70}")
print(f"TEST COMPLETED SUCCESSFULLY")
print(f"{'='*70}\n")
