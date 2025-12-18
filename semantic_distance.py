import spacy, pandas as pd, numpy as np
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # activates 3D plotting


def calculate_distances():

    nlp = spacy.load("en_core_web_sm")

    words = ["Curiosity","Uncertainty","Excitement","Happiness",
            "Surprise","Disgust","Fear","Frustration"]

    # 1) load a VAD lexicon (CSV with columns: word,valence,arousal,dominance)
    vad = pd.read_csv("vad_word_embeddings.csv")  # path to your copy
    vad["word"] = vad["word"].str.lower()
    vad = vad.set_index("word")[["valence","arousal","dominance"]]

    # 2) normalize scales if needed (e.g., z-score each column)
    vad_norm = vad.apply(lambda col: (col - col.mean())/col.std())

    # 3) build your matrix
    def get_vad(w):
        lemma = nlp(w)[0].lemma_.lower()
        return vad_norm.loc[lemma].to_numpy()

    X = np.vstack([get_vad(w) for w in words]) # shape (8,3)

    # 4) distances in VAD space
    dist_vad = cdist(X, X, metric="euclidean")  

    print(dist_vad)
