import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # needed for 3D projection

def plot_vad_points(csv_path="vad_word_embeddings.csv", use_list_order=True):
    # Optional: enforce a specific word order
    target_words = [
        "Curiosity","Uncertainty","Excitement","Happiness",
        "Surprise","Disgust","Fear","Frustration"
    ]

    # 1) Load CSV with headers: word,valence,arousal,dominance
    df = pd.read_csv(csv_path)

    # Basic validation
    required_cols = {"word","valence","arousal","dominance"}
    missing = required_cols - set(df.columns.str.lower())
    if missing:
        raise ValueError(f"CSV is missing columns: {missing}")

    # Normalize column names just in case
    df.columns = [c.lower() for c in df.columns]

    # 2) Ensure desired order
    if use_list_order:
        df = df.set_index("word").reindex(target_words).reset_index()
        if df["word"].isna().any():
            missing_words = [w for w in target_words if w not in df["word"].tolist()]
            raise ValueError(f"Missing words in CSV: {missing_words}")

    words = df["word"].tolist()
    X = df[["valence","arousal","dominance"]].to_numpy(dtype=float)

    # 3) Plot 3D scatter
    fig = plt.figure(figsize=(7,6))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(X[:,0], X[:,1], X[:,2], s=80)

    # Label each point
    for i, w in enumerate(words):
        ax.text(X[i,0] + 0.01, X[i,1] + 0.01, X[i,2] + 0.01, w)

    ax.set_xlabel("Valence")
    ax.set_ylabel("Arousal")
    ax.set_zlabel("Dominance")
    ax.set_title("Emotion Words in Valence–Arousal–Dominance (VAD) Space")

    # Since your values are 0–1, fix axes to that range
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_zlim(0, 1)

    plt.tight_layout()
    plt.show()

# Run it
plot_vad_points("vad_word_embeddings.csv", use_list_order=True)
