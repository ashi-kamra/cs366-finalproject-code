import pandas as pd

# Load your CSV (change filename if needed)
videos = [
    "Change in Recategorization - Video 1.csv",
    "Change in Recategorization - Video 2.csv",
    "Change in Recategorization - Video 3.csv",
    "Change in Recategorization - Video 4.csv",
    "Change in Recategorization - Video 5.csv",
    "Change in Recategorization - Video 6.csv",
    "Change in Recategorization - Video 7.csv",
    "Change in Recategorization - Video 8.csv"
]
for i in range(len(videos)):
    df = pd.read_csv(f"recategorizations/{videos[i]}")

    # 1. Helper: Convert comma-separated strings to sets
    def to_set(x):
        if pd.isna(x):
            return set()
        return set(w.strip() for w in x.split(',') if w.strip())

    # 2. Helper: Compute Jaccard similarity
    def jaccard(a, b):
        if not a and not b:
            return 1.0          # both empty sets → identical
        union = a | b
        if len(union) == 0:
            return 0
        return len(a & b) / len(union)

    # 3. Compute Jaccard similarity for each row
    jaccard_scores = []
    for _, row in df.iterrows():
        before_set = to_set(row['Words Before'])
        after_set  = to_set(row['Words After'])
        jaccard_scores.append(jaccard(before_set, after_set))

    # 4. Add new column
    df['Jaccard Similarity'] = jaccard_scores

    print(jaccard_scores)

    # Preview
    df.to_csv(f"recategorizations/jaccard_video{i+1}.csv", index=False) 
