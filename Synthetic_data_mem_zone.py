# ============================================
# ADD THIS FUNCTION AFTER THE IMPORTS SECTION
# ============================================

def generate_unique_transaction(iteration: int, num_unique_tokens: int = 50) -> str:
    """
    Generate realistic financial transaction descriptions only.
    This simulates production transaction NER where each transaction description has:
    - Unique transaction IDs
    - Merchant names with variations
    - Account numbers
    - Card numbers (last 4 digits)
    - Reference codes
    - Customer IDs
    
    These unique identifiers in transaction descriptions cause vocabulary growth
    in real production systems.
    
    Example output:
    "TXN000001234 STARBUCKS5678 $45.67 CARD-1234 ACCT7890123 AUTH456789"
    """
    # Transaction types
    transaction_types = [
        "POS", "ATM", "ONLINE", "TRANSFER", "PAYMENT",
        "REFUND", "WITHDRAWAL", "DEPOSIT"
    ]
    
    # Merchant variations (real-world merchant name patterns)
    merchants = [
        "AMAZON", "WALMART", "STARBUCKS", "SHELL", "MCDONALDS",
        "TARGET", "COSTCO", "BESTBUY", "NETFLIX", "UBER",
        "AIRBNB", "BOOKING", "PAYPAL", "VENMO", "SQUARE",
        "SPOTIFY", "APPLE", "GOOGLE", "MICROSOFT", "CHIPOTLE"
    ]
    
    # Generate unique identifiers for this transaction
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
    
    # Add more unique tokens to reach the target
    unique_tokens = []
    remaining = num_unique_tokens - 11  # We already have 11 base identifiers
    
    for i in range(remaining):
        token = f"{random.choice(['LOC', 'ID', 'CODE', 'SEQ'])}{iteration}{i}{random.randint(100, 999)}"
        unique_tokens.append(token)
    
    # Create simple transaction description (space-separated identifiers)
    description_parts = [
        txn_id,
        trans_type,
        merchant,
        amount,
        card,
        acct,
        auth,
        ref,
        merchant_id,
        terminal,
        batch
    ] + unique_tokens
    
    # Return space-separated transaction description
    return " ".join(description_parts)


# ============================================
# ADD THIS IMPORT AT THE TOP (with other imports)
# ============================================
import random


# ============================================
# REPLACE THE main() FUNCTION WITH THIS VERSION
# ============================================

def main():
    # ===== SYNTHETIC DATA CONFIGURATION =====
    NUM_ROWS_TO_GENERATE = 100000  # <-- CHANGE THIS TO CONTROL HOW MANY ROWS TO GENERATE
    
    # 1. Set the batch size for processing records.10k
    BATCH_SIZE = 50
    
    # 2. Set how many batches per CSV file (to avoid file too large errors)10
    BATCHES_PER_FILE = 2000  # 10 batches = 10,000 records = 100,000 records per file
    
    # REMOVED: INPUT_CSV path is no longer needed
    OUTPUT_DIR = ""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.environ['DESCRIPTORS_TO_REMOVE'] = 'LLC,PTY,INC'
    
    # Initialize Model
    init_args = {
        'model_name': ''
    }
    
    # ===== GENERATE SYNTHETIC DATA =====
    print(f"\nGenerating {NUM_ROWS_TO_GENERATE} synthetic transaction descriptions...")
    descriptions = []
    memos = []
    
    for i in range(NUM_ROWS_TO_GENERATE):
        # Generate unique transaction description
        description = generate_unique_transaction(iteration=i, num_unique_tokens=50)
        descriptions.append(description)
        memos.append("")  # Empty memo as requested
        
        if (i + 1) % 10000 == 0:
            print(f"  Generated {i + 1}/{NUM_ROWS_TO_GENERATE} transactions...")
    
    # Create DataFrame with synthetic data
    df = pd.DataFrame({
        'description': descriptions,
        'memo': memos
    })
    
    print(f"\nGenerated {len(df)} synthetic records.")
    print(f"Sample transaction:\n{df['description'].iloc[0]}\n")
    print(f"Each CSV file will contain {BATCHES_PER_FILE} batches ({BATCHES_PER_FILE * BATCH_SIZE} records).")
    
    # Run Investigation
    all_results = []
    
    # TEST 1: Without Memory Zone
    # print("INIT model for test 1")
    # ner_test1 = SpacyModel()
    # ner_test1.initialize(init_args)
    # ner_test1.use_memory_zone = False
    # without_zone_folder = os.path.join(OUTPUT_DIR, "without_zone")
    # results_without_zone = run_test(
    #     ner_test1, df, BATCH_SIZE, "Without Memory Zone",
    #     without_zone_folder,
    #     BATCHES_PER_FILE
    # )
    # all_results.append(results_without_zone)
    # del ner_test1
    # gc.collect()
    # time.sleep(60)
    
    ## TEST 2: With Memory Zone
    print("REINIT for test 2")
    ner_test2 = SpacyModel()
    ner_test2.initialize(init_args)
    ner_test2.use_memory_zone = True
    
    with_zone_folder = os.path.join(OUTPUT_DIR, "with_zone")
    results_with_zone = run_test(
        ner_test2, df, BATCH_SIZE, "With Memory Zone",
        with_zone_folder,
        BATCHES_PER_FILE
    )
    
    all_results.append(results_with_zone)
    
    # Generate Final Report
    # generate_report(all_results, OUTPUT_DIR)
    
    print("\nInvestigation Finished.")
    
    if not all_results:
        print("Attempt to load save json")
        without_zone_folder = os.path.join(OUTPUT_DIR, "without_zone")
        with_zone_folder = os.path.join(OUTPUT_DIR, "with_zone")
        without_json = os.path.join(without_zone_folder, 'results.json')
        with_json = os.path.join(with_zone_folder, 'results.json')
        
        if os.path.exists(without_json) and os.path.exists(with_json):
            with open(without_json, 'r') as f:
                all_results.append(json.load(f))
            with open(with_json, 'r') as f:
                all_results.append(json.load(f))
            print("Loaded both json files")
            generate_report(all_results, OUTPUT_DIR)
            print("Generated report!")
    else:
        print("No json files found")


if __name__ == '__main__':
    main()
