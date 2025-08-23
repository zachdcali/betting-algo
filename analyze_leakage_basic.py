import csv
import statistics

print("Loading dataset...")

# Read the CSV file and analyze key features
key_features = ['Player1_Wins', 'Player1_Rank_Advantage', 'Player1_Height_Advantage', 
                'Player1_Age_Advantage', 'Player1_Points_Advantage']

data = {}
row_count = 0

with open('/Users/zachdodson/Documents/betting-algo/data/JeffSackmann/jeffsackmann_ml_ready_LEAK_FREE.csv', 'r') as file:
    reader = csv.DictReader(file)
    
    # Initialize data storage
    for feature in key_features:
        data[feature] = []
    
    # Process rows
    for row in reader:
        row_count += 1
        if row_count % 100000 == 0:
            print(f"Processed {row_count} rows...")
        
        for feature in key_features:
            if feature in row and row[feature]:
                try:
                    value = float(row[feature])
                    data[feature].append(value)
                except ValueError:
                    pass  # Skip non-numeric values

print(f"\nTotal rows processed: {row_count}")

print("\n" + "="*50)
print("ANALYZING KEY FEATURES FOR DATA LEAKAGE")
print("="*50)

# 1. Check mean values of key features
print("\n1. MEAN VALUES OF KEY FEATURES (should be ~0.5):")
print("-" * 50)
for feature in key_features:
    if feature in data and data[feature]:
        mean_val = statistics.mean(data[feature])
        print(f"{feature}: {mean_val:.4f} (n={len(data[feature])})")
    else:
        print(f"{feature}: NO DATA FOUND")

# 2. Check Player1_Wins distribution specifically
if 'Player1_Wins' in data and data['Player1_Wins']:
    wins_data = data['Player1_Wins']
    win_count = sum(1 for x in wins_data if x == 1.0)
    loss_count = sum(1 for x in wins_data if x == 0.0)
    print(f"\nPlayer1_Wins distribution:")
    print(f"Wins (1.0): {win_count}")
    print(f"Losses (0.0): {loss_count}")
    print(f"Win proportion: {win_count / len(wins_data):.4f}")

# 3. Calculate correlations between advantage features and Player1_Wins
print("\n2. CORRELATIONS WITH Player1_Wins:")
print("-" * 50)

def correlation(x, y):
    """Calculate Pearson correlation coefficient"""
    if len(x) != len(y) or len(x) == 0:
        return None
    
    mean_x = statistics.mean(x)
    mean_y = statistics.mean(y)
    
    numerator = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(len(x)))
    sum_sq_x = sum((x[i] - mean_x) ** 2 for i in range(len(x)))
    sum_sq_y = sum((y[i] - mean_y) ** 2 for i in range(len(y)))
    
    denominator = (sum_sq_x * sum_sq_y) ** 0.5
    
    if denominator == 0:
        return None
    
    return numerator / denominator

if 'Player1_Wins' in data and data['Player1_Wins']:
    wins_data = data['Player1_Wins']
    
    for feature in key_features[1:]:  # Skip Player1_Wins itself
        if feature in data and data[feature]:
            # Make sure we have matching data points
            feature_data = data[feature]
            min_len = min(len(wins_data), len(feature_data))
            
            if min_len > 0:
                corr = correlation(feature_data[:min_len], wins_data[:min_len])
                if corr is not None:
                    print(f"{feature} vs Player1_Wins: {corr:.4f}")
                else:
                    print(f"{feature} vs Player1_Wins: CORRELATION FAILED")
            else:
                print(f"{feature} vs Player1_Wins: NO MATCHING DATA")

# 4. Check variance in advantage features
print("\n3. VARIANCE CHECK (higher variance indicates better randomization):")
print("-" * 50)
for feature in key_features[1:]:  # Skip Player1_Wins
    if feature in data and data[feature] and len(data[feature]) > 1:
        variance = statistics.variance(data[feature])
        stdev = statistics.stdev(data[feature])
        print(f"{feature} - Variance: {variance:.4f}, Std: {stdev:.4f}")

# 5. Descriptive statistics
print("\n4. DESCRIPTIVE STATISTICS:")
print("-" * 50)
for feature in key_features:
    if feature in data and data[feature]:
        values = data[feature]
        print(f"\n{feature}:")
        print(f"  Count: {len(values)}")
        print(f"  Mean: {statistics.mean(values):.4f}")
        print(f"  Min: {min(values):.4f}")
        print(f"  Max: {max(values):.4f}")
        if len(values) > 1:
            print(f"  Std: {statistics.stdev(values):.4f}")

print("\n" + "="*50)
print("ANALYSIS COMPLETE")
print("="*50)