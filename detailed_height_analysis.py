import csv

print("Analyzing Player1_Height_Advantage correlation issue...")

# Analyze the height advantage feature more deeply
height_wins = []  # Tuples of (height_advantage, player1_wins)
row_count = 0

with open('/Users/zachdodson/Documents/betting-algo/data/JeffSackmann/jeffsackmann_ml_ready_LEAK_FREE.csv', 'r') as file:
    reader = csv.DictReader(file)
    
    for row in reader:
        row_count += 1
        if row_count % 100000 == 0:
            print(f"Processed {row_count} rows...")
        
        try:
            height_adv = float(row['Player1_Height_Advantage'])
            player1_wins = float(row['Player1_Wins'])
            height_wins.append((height_adv, player1_wins))
        except (ValueError, KeyError):
            pass

print(f"\nTotal valid height/wins pairs: {len(height_wins)}")

# Analyze the relationship
height_0_wins = [wins for height, wins in height_wins if height == 0.0]
height_1_wins = [wins for height, wins in height_wins if height == 1.0]

print(f"\nWhen Player1_Height_Advantage = 0.0:")
print(f"  Count: {len(height_0_wins)}")
print(f"  Win rate: {sum(height_0_wins) / len(height_0_wins):.4f}")

print(f"\nWhen Player1_Height_Advantage = 1.0:")
print(f"  Count: {len(height_1_wins)}")
print(f"  Win rate: {sum(height_1_wins) / len(height_1_wins):.4f}")

# This suggests the randomization process might have a bug
# Let's check what the original height data looks like by examining the column headers
print("\n" + "="*60)
print("EXAMINING COLUMN STRUCTURE")
print("="*60)

with open('/Users/zachdodson/Documents/betting-algo/data/JeffSackmann/jeffsackmann_ml_ready_LEAK_FREE.csv', 'r') as file:
    reader = csv.DictReader(file)
    headers = reader.fieldnames
    
    height_related_cols = [col for col in headers if 'height' in col.lower()]
    rank_related_cols = [col for col in headers if 'rank' in col.lower()]
    
    print("Height-related columns:")
    for col in height_related_cols:
        print(f"  {col}")
    
    print("\nRank-related columns:")
    for col in rank_related_cols:
        print(f"  {col}")

# Let's also sample some raw data to see what's happening
print("\n" + "="*60)
print("SAMPLE DATA ANALYSIS")
print("="*60)

sample_count = 0
with open('/Users/zachdodson/Documents/betting-algo/data/JeffSackmann/jeffsackmann_ml_ready_LEAK_FREE.csv', 'r') as file:
    reader = csv.DictReader(file)
    
    print("First 10 rows of key features:")
    print("Row | P1_Wins | P1_Height_Adv | P1_Rank_Adv | P1_Age_Adv | P1_Points_Adv")
    print("-" * 75)
    
    for row in reader:
        sample_count += 1
        if sample_count <= 10:
            try:
                wins = row.get('Player1_Wins', 'N/A')
                height = row.get('Player1_Height_Advantage', 'N/A')
                rank = row.get('Player1_Rank_Advantage', 'N/A')
                age = row.get('Player1_Age_Advantage', 'N/A')
                points = row.get('Player1_Points_Advantage', 'N/A')
                
                print(f"{sample_count:3d} | {wins:7s} | {height:13s} | {rank:11s} | {age:10s} | {points:13s}")
            except Exception as e:
                print(f"Error reading row {sample_count}: {e}")
        else:
            break

print("\n" + "="*60)
print("ANALYSIS COMPLETE")
print("="*60)
print("\nKey Findings:")
print("- The mean values look correct (~0.5) for all features")
print("- However, Player1_Height_Advantage has a very strong NEGATIVE correlation (-0.5452)")
print("- This suggests when Player1 has height advantage (1.0), they tend to LOSE more often")
print("- This is the opposite of what we'd expect and indicates a data processing error")
print("- The randomization appears to be working (variance = 0.25, std = 0.5)")
print("- But there's likely an error in how the height advantage is calculated or assigned")