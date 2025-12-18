import pandas as pd
import numpy as np

# ----- 1. Load VAD lexicon -----
vad_df = pd.read_csv("vad_word_embeddings.csv")


# Build a dictionary: word -> np.array([V, A, D])
vad_dict = {
    row["word"]: np.array([row["valence"], row["arousal"], row["dominance"]], dtype=float)
    for _, row in vad_df.iterrows()
}

# ----- 2. Helper functions -----
def words_to_list(s):
    """Getting list of emotion words from strings."""
    if pd.isna(s):
        return []
    return [w.strip() for w in str(s).split(",") if w.strip()]

def centroid(words, vad_lookup):
    """Compute the VAD centroid (mean vector) for a list of emotion words."""
    vecs = []
    for w in words:
        key = w.lower()
        if key in vad_lookup:
            vecs.append(vad_lookup[key])
        else:
            pass
    if not vecs:
        return None
    return np.mean(vecs, axis=0)

def vad_centroid_distance(before_words_str, after_words_str, vad_lookup):
    """Compute Euclidean distance between centroids of before/after word sets."""
    before_list = words_to_list(before_words_str)
    after_list  = words_to_list(after_words_str)

    c_before = centroid(before_list, vad_lookup)
    c_after  = centroid(after_list, vad_lookup)

    if c_before is None or c_after is None:
        return np.nan  # no valid words → undefined distance
    return float(np.linalg.norm(c_before - c_after))

# ----- 3. Process all video files -----

videos = [
    "Human vs. Ground Truth - Video 1.csv",
    "Human vs. Ground Truth - Video 2.csv",
    "Human vs. Ground Truth - Video 3.csv",
    "Human vs. Ground Truth - Video 4.csv",
    "Human vs. Ground Truth - Video 5.csv",
    "Human vs. Ground Truth - Video 6.csv",
    "Human vs. Ground Truth - Video 7.csv",
    "Human vs. Ground Truth - Video 8.csv"
]

for i in range(len(videos)):
    # Read the original recategorization file
    df = pd.read_csv(f"accuracy/{videos[i]}")

    # Compute VAD centroid distance for each row
    distances = []
    for _, row in df.iterrows():
        d = vad_centroid_distance(
            row["Human Categorization"],
            row["Ground Truth"],
            vad_dict
        )
        distances.append(d)

    # Add new column
    df["VAD Centroid Distance"] = distances

    # Save to new file
    out_path = f"accuracy/vad_centroid_video_{i+1}.csv"
    df.to_csv(out_path, index=False)
    print(f"Processed {videos[i]} → {out_path}")