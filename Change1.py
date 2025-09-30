# In class SpacyModel:
    
    def execute(self, requests, use_memory_zone: bool = True): # <-- Flag added here
        responses, request_sizes, input_all, descriptions_all, memos_all = [], [], [], [], []
        for request in requests:
            descriptions = pb_utils.get_input_tensor_by_name(request, "description")
            descriptions_ls = [x[0].decode('utf-8') for x in descriptions.as_numpy()]
            memos = pb_utils.get_input_tensor_by_name(request, "memo")
            memos_ls = [x[0].decode('utf-8') for x in memos.as_numpy()]
            request_sizes.append(len(descriptions_ls))
            inputs = self.preprocess_input(descriptions_ls, memos_ls)
            input_all.extend(inputs)
            descriptions_all.extend(descriptions_ls)
            memos_all.extend(memos_ls)

        self.logger.log_info(get_log(logging.INFO, f"request_sizes = {request_sizes}"))

        # This block now controls whether to use memory_zone based on the flag
        if use_memory_zone:
            self.logger.log_info(get_log(logging.INFO, "Executing with memory_zone."))
            with self.nlp.memory_zone():
                result_docs = self.nlp.pipe(input_all)
                _, ret_labels, ret_extracted_texts, ret_ent_ids = self.extract_results(result_docs, descriptions_all, memos_all)
        else:
            self.logger.log_info(get_log(logging.INFO, "Executing WITHOUT memory_zone."))
            result_docs = self.nlp.pipe(input_all)
            _, ret_labels, ret_extracted_texts, ret_ent_ids = self.extract_results(result_docs, descriptions_all, memos_all)

        ret_labels = np.array(ret_labels, dtype='|S0')
        ret_extracted_texts = np.array(ret_extracted_texts, dtype=object)
        ret_ent_ids = np.array(ret_ent_ids, dtype='|S0')

        start_idx = 0
        for request_size in request_sizes:
            end_idx = start_idx + request_size
            ret_labels_tensor = pb_utils.Tensor("label", ret_labels[start_idx:end_idx])
            ret_extracted_texts_tensor = pb_utils.Tensor("extractedText", ret_extracted_texts[start_idx:end_idx])
            ret_ent_ids_tensor = pb_utils.Tensor("entityId", ret_ent_ids[start_idx:end_idx])
            inference_response = pb_utils.InferenceResponse(
                output_tensors=[ret_labels_tensor, ret_extracted_texts_tensor, ret_ent_ids_tensor]
            )
            start_idx = end_idx
            responses.append(inference_response)
        return responses





# Import Packages
import os
import json
import numpy as np
import pandas as pd
from unidecode import unidecode
from packages.spacy_model import SpacyModel
from triton_python_backend_utils import Tensor
from tests.mocks import mockInferenceRequest
import psutil
import time
import gc
import argparse # Added for command-line arguments

# --- HELPER CLASS FOR MEMORY PROFILING ---
class MemoryProfiler:
    def __init__(self):
        self.process = psutil.Process(os.getpid())
    def get_memory_mb(self) -> float:
        return self.process.memory_info().rss / 1024 / 1024

# --- MAIN SCRIPT LOGIC ---

# 1. Standard Model Initialization
ner = SpacyModel()
init_args = {'model_name': 'ner'}
ner.initialize(init_args)
profiler = MemoryProfiler()

# 2. Setup Command-Line Argument Parser
parser = argparse.ArgumentParser(description="Run a specific NER memory test scenario.")
parser.add_argument(
    '--test', 
    type=str, 
    required=True, 
    choices=['baseline', 'fix_only', 'full_pipeline'],
    help="The name of the test to run."
)
args = parser.parse_args()

# 3. Load Full Dataset
full_df = pd.read_csv("", keep_default_na=False)
full_df['description'] = full_df['description'].apply(unidecode)
full_df['memo'] = full_df['memo'].apply(unidecode)

# 4. Define All Possible Test Scenarios
test_scenarios = {
    'baseline': {'name': 'Baseline (No memory_zone)', 'use_memory_zone': False, 'aggregate_results': True},
    'fix_only': {'name': 'memory_zone Only (No Aggregation)', 'use_memory_zone': True, 'aggregate_results': False},
    'full_pipeline': {'name': 'Full Pipeline (memory_zone + Aggregation)', 'use_memory_zone': True, 'aggregate_results': True}
}

# 5. Select and Run the Chosen Scenario
scenario = test_scenarios[args.test]
output_filename = f"{args.test}_results.json"

print(f"\n{'='*60}\n🔬 RUNNING TEST: {scenario['name']}\n{'='*60}")

memory_data, timestamps = [], []
all_final_outputs = []
initial_memory = profiler.get_memory_mb()
iterations = 500
batch_size = 100

for i in range(iterations):
    start_row = i * batch_size
    end_row = start_row + batch_size
    df_batch = full_df.iloc[start_row:end_row]
    
    if len(df_batch) == 0:
        print("  - No more data to process.")
        break

    # Your original request formatting logic
    descriptions = df_batch['description'].to_list()
    memos = df_batch['memo'].to_list()
    descriptions_vec = np.array(descriptions, dtype='|S0').reshape(len(descriptions), 1)
    memos_vec = np.array(memos, dtype='|S0').reshape(len(memos), 1)
    requests = [mockInferenceRequest(inputs=[Tensor(data=descriptions_vec, name='description'), Tensor(data=memos_vec, name='memo')])]

    # Execute the pipeline with the correct configuration
    raw_results = ner.execute(requests, use_memory_zone=scenario['use_memory_zone'])

    # Your real post-processing and aggregation logic
    if scenario['aggregate_results']:
        outputs = []
        for raw_result in raw_results:
            labels, extracted_texts, entity_ids = raw_result.output_tensors()
            labels, extracted_texts, entity_ids = labels.as_numpy().tolist(), extracted_texts.as_numpy().tolist(), entity_ids.as_numpy().tolist()
            for label_list, extracted_text_list, entity_id_list in zip(labels, extracted_texts, entity_ids):
                these_outputs = []
                decoded_labels = [x.decode('utf-8') for x in label_list]
                decoded_extracted_texts = [x for x in extracted_text_list]
                decoded_entity_ids = [x.decode('utf-8') for x in entity_id_list]
                for label, extracted_text, entity_id in zip(decoded_labels, decoded_extracted_texts, decoded_entity_ids):
                    if label != '' and label is not None:
                        these_outputs.append({'entity_type': label, 'extracted_entity': extracted_text, 'standardized_entity': entity_id})
                outputs.append(these_outputs)
        all_final_outputs.extend(outputs)

    if (i + 1) % 20 == 0:
        mem_mb = profiler.get_memory_mb()
        growth = mem_mb - initial_memory
        memory_data.append(mem_mb)
        timestamps.append(i + 1)
        print(f"  Batch {i+1}: Memory={mem_mb:.1f}MB (+{growth:.1f}MB)")

gc.collect()
final_memory = profiler.get_memory_mb()

# 6. Save results to a JSON file
results_data = {
    'test_name': scenario['name'],
    'memory_mb': memory_data,
    'iterations': timestamps,
    'memory_growth': final_memory - initial_memory
}
with open(output_filename, 'w') as f:
    json.dump(results_data, f, indent=4)

print(f"\n✅ Test complete. Results saved to '{output_filename}'")

