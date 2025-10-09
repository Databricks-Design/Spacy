#!/usr/bin/env python3

import os
import pandas as pd
import json
from typing import Any, Dict, List, Tuple
import glob
import warnings
warnings.filterwarnings('ignore')

def normalize_json(obj: Any) -> Any:
    """Recursively normalize JSON structure for comparison"""
    if isinstance(obj, dict):
        # Sort dict keys and recursively normalize values
        return {k: normalize_json(v) for k, v in sorted(obj.items())}
    elif isinstance(obj, list):
        # Sort list if it contains dicts (for order-independent comparison)
        if obj and isinstance(obj[0], dict):
            return sorted([normalize_json(item) for item in obj], 
                         key=lambda x: json.dumps(x, sort_keys=True))
        return [normalize_json(item) for item in obj]
    else:
        return obj

def parse_outputs_ner(value: Any) -> List[Dict]:
    """Parse outputs_ner column which can be string or list"""
    if pd.isna(value) or value == '' or value == '[]':
        return []
    
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, list) else [parsed]
        except:
            return []
    elif isinstance(value, list):
        return value
    else:
        return []

def get_original_outputs_ner_string(df: pd.DataFrame, idx: int) -> str:
    """Get the original outputs_ner value as string from the dataframe"""
    try:
        value = df.iloc[idx].get('outputs_ner', '')
        if pd.isna(value) or value == '':
            return '[]'
        # Return as-is if it's already a string, otherwise convert
        if isinstance(value, str):
            return value
        else:
            return json.dumps(value, ensure_ascii=False)
    except:
        return '[]'

def compare_outputs(output1: List[Dict], output2: List[Dict]) -> Tuple[bool, str]:
    """
    Compare two outputs_ner values
    Returns: (is_match, comment)
    """
    # Normalize both outputs
    norm1 = normalize_json(output1)
    norm2 = normalize_json(output2)
    
    # Check if they're identical
    if norm1 == norm2:
        return True, "MATCH"
    
    # If not identical, provide detailed comment
    comments = []
    
    # Check length difference
    if len(output1) != len(output2):
        comments.append(f"Length mismatch: without_zone={len(output1)}, with_zone={len(output2)}")
    
    # Check for missing/extra entities
    if isinstance(output1, list) and isinstance(output2, list):
        # Compare each entity
        max_len = max(len(output1), len(output2))
        for i in range(max_len):
            if i >= len(output1):
                comments.append(f"Extra entity in with_zone at index {i}: {output2[i]}")
            elif i >= len(output2):
                comments.append(f"Missing entity in with_zone at index {i}: {output1[i]}")
            else:
                # Compare individual entities
                if normalize_json(output1[i]) != normalize_json(output2[i]):
                    diff_keys = set(output1[i].keys()) ^ set(output2[i].keys())
                    if diff_keys:
                        comments.append(f"Entity {i} - Different keys: {diff_keys}")
                    else:
                        for key in output1[i].keys():
                            if output1[i].get(key) != output2[i].get(key):
                                comments.append(
                                    f"Entity {i} - Key '{key}': "
                                    f"without_zone='{output1[i].get(key)}', "
                                    f"with_zone='{output2[i].get(key)}'"
                                )
    
    return False, " | ".join(comments) if comments else "Structure mismatch"

def verify_same_input_data(df_without: pd.DataFrame, df_with: pd.DataFrame, 
                           id_col: str, desc_col: str) -> Tuple[bool, str]:
    """Verify that both dataframes contain the same input data"""
    
    if len(df_without) != len(df_with):
        return False, f"Different number of rows: without_zone={len(df_without)}, with_zone={len(df_with)}"
    
    # Check if ID columns match
    if id_col in df_without.columns and id_col in df_with.columns:
        ids_match = (df_without[id_col].astype(str) == df_with[id_col].astype(str)).all()
        if not ids_match:
            return False, "ID columns don't match - files may not be aligned"
    
    # Check if description/dsp columns match
    if desc_col in df_without.columns and desc_col in df_with.columns:
        desc_match = (df_without[desc_col].astype(str) == df_with[desc_col].astype(str)).all()
        if not desc_match:
            return False, "Description columns don't match - files may not be aligned"
    
    return True, "Input data verified as identical"

def compare_csv_files(without_zone_csv: str, with_zone_csv: str) -> pd.DataFrame:
    """Compare two CSV files and return comparison dataframe"""
    
    print(f"\nComparing:")
    print(f"  Without Zone: {without_zone_csv}")
    print(f"  With Zone: {with_zone_csv}")
    
    # Read both CSVs with low_memory=False to handle mixed types
    df_without = pd.read_csv(without_zone_csv, keep_default_na=False, low_memory=False)
    df_with = pd.read_csv(with_zone_csv, keep_default_na=False, low_memory=False)
    
    # Find ID and description columns (handle unnamed columns)
    id_col = None
    desc_col = None
    
    for col in df_without.columns:
        if 'unnamed' in col.lower() and ('0' in col or 'id' in col.lower()):
            id_col = col
        elif 'description' in col.lower() or 'dsp' in col.lower():
            desc_col = col
    
    # If not found, use first few columns
    if id_col is None:
        id_col = df_without.columns[0]
    if desc_col is None:
        desc_col = df_without.columns[1] if len(df_without.columns) > 1 else id_col
    
    print(f"  Using ID column: {id_col}")
    print(f"  Using Description column: {desc_col}")
    
    # VERIFY SAME INPUT DATA
    is_same, verification_msg = verify_same_input_data(df_without, df_with, id_col, desc_col)
    print(f"  Input Data Verification: {verification_msg}")
    
    if not is_same:
        print(f"  ⚠️ CRITICAL WARNING: Input data mismatch detected!")
    
    # Create comparison dataframe
    comparison_data = []
    
    for idx in range(max(len(df_without), len(df_with))):
        if idx >= len(df_without):
            # Row only in with_zone
            row_id = str(df_with.iloc[idx][id_col]) if id_col in df_with.columns else f"Row_{idx}"
            row_desc = str(df_with.iloc[idx][desc_col]) if desc_col in df_with.columns else ""
            without_output_str = "MISSING ROW"
            with_output_str = get_original_outputs_ner_string(df_with, idx)
            
            comparison_data.append({
                'row_number': idx + 1,
                id_col: row_id,
                desc_col: row_desc,
                'without_zone_output': without_output_str,
                'with_zone_output': with_output_str,
                'match_status': 'MISMATCH',
                'comments': "Row exists only in with_zone"
            })
        elif idx >= len(df_with):
            # Row only in without_zone
            row_id = str(df_without.iloc[idx][id_col]) if id_col in df_without.columns else f"Row_{idx}"
            row_desc = str(df_without.iloc[idx][desc_col]) if desc_col in df_without.columns else ""
            without_output_str = get_original_outputs_ner_string(df_without, idx)
            with_output_str = "MISSING ROW"
            
            comparison_data.append({
                'row_number': idx + 1,
                id_col: row_id,
                desc_col: row_desc,
                'without_zone_output': without_output_str,
                'with_zone_output': with_output_str,
                'match_status': 'MISMATCH',
                'comments': "Row exists only in without_zone"
            })
        else:
            # Both rows exist - verify they're comparing same input
            row_id_without = str(df_without.iloc[idx][id_col]) if id_col in df_without.columns else f"Row_{idx}"
            row_id_with = str(df_with.iloc[idx][id_col]) if id_col in df_with.columns else f"Row_{idx}"
            row_desc_without = str(df_without.iloc[idx][desc_col]) if desc_col in df_without.columns else ""
            row_desc_with = str(df_with.iloc[idx][desc_col]) if desc_col in df_with.columns else ""
            
            # Check if we're comparing the same row
            if row_id_without != row_id_with or row_desc_without != row_desc_with:
                comparison_data.append({
                    'row_number': idx + 1,
                    id_col: f"without={row_id_without}, with={row_id_with}",
                    desc_col: row_desc_without,
                    'without_zone_output': "N/A",
                    'with_zone_output': "N/A",
                    'match_status': 'DATA_MISMATCH',
                    'comments': "Input data doesn't match - rows are not aligned!"
                })
                continue
            
            # Get original outputs_ner strings from CSVs
            without_output_str = get_original_outputs_ner_string(df_without, idx)
            with_output_str = get_original_outputs_ner_string(df_with, idx)
            
            # Parse for comparison
            output_without = parse_outputs_ner(df_without.iloc[idx].get('outputs_ner', []))
            output_with = parse_outputs_ner(df_with.iloc[idx].get('outputs_ner', []))
            
            is_match, comment = compare_outputs(output_without, output_with)
            
            comparison_data.append({
                'row_number': idx + 1,
                id_col: row_id_without,
                desc_col: row_desc_without,
                'without_zone_output': without_output_str,  # Use original string
                'with_zone_output': with_output_str,        # Use original string
                'match_status': 'MATCH' if is_match else 'MISMATCH',
                'comments': comment
            })
    
    return pd.DataFrame(comparison_data)

def compare_all_parts(output_dir: str):
    """Compare all CSV parts between without_zone and with_zone folders"""
    
    without_zone_folder = os.path.join(output_dir, "without_zone")
    with_zone_folder = os.path.join(output_dir, "with_zone")
    comparison_folder = os.path.join(output_dir, "comparison")
    
    os.makedirs(comparison_folder, exist_ok=True)
    
    # Get all CSV files from both folders
    without_zone_files = sorted(glob.glob(os.path.join(without_zone_folder, "output_part_*.csv")))
    with_zone_files = sorted(glob.glob(os.path.join(with_zone_folder, "output_part_*.csv")))
    
    if not without_zone_files:
        print("ERROR: No CSV files found in without_zone folder!")
        return
    
    if not with_zone_files:
        print("ERROR: No CSV files found in with_zone folder!")
        return
    
    print(f"\nFound {len(without_zone_files)} files in without_zone")
    print(f"Found {len(with_zone_files)} files in with_zone")
    
    if len(without_zone_files) != len(with_zone_files):
        print(f"⚠️ WARNING: Different number of CSV files! This may indicate incomplete processing.")
    
    # Compare each pair
    all_comparisons = []
    total_matches = 0
    total_mismatches = 0
    total_data_mismatches = 0
    
    num_pairs = min(len(without_zone_files), len(with_zone_files))
    
    for i in range(num_pairs):
        without_file = without_zone_files[i]
        with_file = with_zone_files[i]
        
        print(f"\n{'='*80}")
        print(f"Processing Part {i+1}/{num_pairs}")
        
        comparison_df = compare_csv_files(without_file, with_file)
        
        # Save individual comparison
        part_comparison_file = os.path.join(comparison_folder, f"comparison_part_{i+1:03d}.csv")
        comparison_df.to_csv(part_comparison_file, index=False, escapechar='\\')
        print(f"  ✓ Saved: {part_comparison_file}")
        
        # Statistics
        matches = (comparison_df['match_status'] == 'MATCH').sum()
        mismatches = (comparison_df['match_status'] == 'MISMATCH').sum()
        data_mismatches = (comparison_df['match_status'] == 'DATA_MISMATCH').sum()
        
        total_matches += matches
        total_mismatches += mismatches
        total_data_mismatches += data_mismatches
        
        print(f"  Matches: {matches} | Mismatches: {mismatches} | Data Mismatches: {data_mismatches}")
        
        all_comparisons.append(comparison_df)
    
    # Combine all comparisons
    print(f"\n{'='*80}")
    print("Combining all comparisons...")
    combined_df = pd.concat(all_comparisons, ignore_index=True)
    
    # Save combined comparison with proper escaping
    combined_file = os.path.join(comparison_folder, "comparison_all_combined.csv")
    combined_df.to_csv(combined_file, index=False, escapechar='\\')
    print(f"✓ Saved combined comparison: {combined_file}")
    
    # Save MISMATCHES ONLY
    mismatches_df = combined_df[combined_df['match_status'].isin(['MISMATCH', 'DATA_MISMATCH'])]
    if len(mismatches_df) > 0:
        mismatches_file = os.path.join(comparison_folder, "mismatches_only.csv")
        mismatches_df.to_csv(mismatches_file, index=False, escapechar='\\')
        print(f"✓ Saved mismatches only: {mismatches_file}")
    
    # Generate summary report
    print(f"\n{'='*80}")
    print("COMPARISON SUMMARY")
    print(f"{'='*80}")
    print(f"Total Rows Compared: {len(combined_df)}")
    print(f"Total Matches: {total_matches} ({total_matches/len(combined_df)*100:.2f}%)")
    print(f"Total Mismatches: {total_mismatches} ({total_mismatches/len(combined_df)*100:.2f}%)")
    
    if total_data_mismatches > 0:
        print(f" Data Alignment Issues: {total_data_mismatches} ({total_data_mismatches/len(combined_df)*100:.2f}%)")
        print(f"   This means input data doesn't match between without_zone and with_zone!")
    
    if total_mismatches > 0:
        print(f"\n✓ Review '{mismatches_file}' to see all {total_mismatches + total_data_mismatches} issues")
    else:
        print("\n🎉 PERFECT MATCH! All outputs are identical between without_zone and with_zone!")
    
    # Generate summary statistics file
    summary_file = os.path.join(comparison_folder, "summary_report.txt")
    with open(summary_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("OUTPUT COMPARISON SUMMARY REPORT\n")
        f.write("="*80 + "\n\n")
        f.write(f"Total Rows Compared: {len(combined_df)}\n")
        f.write(f"Total Matches: {total_matches} ({total_matches/len(combined_df)*100:.2f}%)\n")
        f.write(f"Total Output Mismatches: {total_mismatches} ({total_mismatches/len(combined_df)*100:.2f}%)\n")
        f.write(f"Total Data Alignment Issues: {total_data_mismatches} ({total_data_mismatches/len(combined_df)*100:.2f}%)\n\n")
        
        if total_mismatches > 0:
            f.write("Top 20 Mismatch Reasons:\n")
            f.write("-"*80 + "\n")
            mismatch_df = combined_df[combined_df['match_status'] == 'MISMATCH']
            if len(mismatch_df) > 0:
                mismatch_reasons = mismatch_df['comments'].value_counts().head(20)
                for reason, count in mismatch_reasons.items():
                    f.write(f"  {count:6d}x - {reason[:200]}\n")  # Truncate long comments
        
        if total_data_mismatches > 0:
            f.write("\nCRITICAL: Data alignment issues detected!\n")
            f.write("This suggests the input data in both tests may not be identical.\n")
            f.write("Please verify that both tests processed the same input CSV with same batch order.\n")
    
    print(f"\n✓ Saved summary report: {summary_file}")
    print(f"\n{'='*80}")

def main():
    OUTPUT_DIR = "./output"
    
    if not os.path.exists(OUTPUT_DIR):
        print(f"ERROR: Output directory '{OUTPUT_DIR}' not found!")
        return
    
    compare_all_parts(OUTPUT_DIR)
    print("\nComparison complete!")

if __name__ == "__main__":
    main()
