import matplotlib.pyplot as plt
import numpy as np

# Data
videos = ["Video 1","Video 2","Video 3","Video 4","Video 5",
          "Video 6","Video 7","Video 8"]

#valence changes
# valence_averages = [0.4666, 0.2666, 0.2666, 0.333, 0.2666, 0.066, 0.333, 0.1333]
# valence_stdev = [0.6399, 0.5936, 0.4577, 0.4879, 0.4577, 0.2581, 0.617, 0.351]

# # word changes
word_averages = [1.2666, 1, 0.4, 1.0666, 0.8666, 1.1333, 0.6, 0.666]
word_stdev = [1.279880947, 1.069, 0.6324, 1.1629, 1.3557, 0.9154, 1.0555, 0.8164]

# avg = np.array(valence_averages)
# std = np.array(valence_stdev)

avg = np.array(word_averages)
std = np.array(word_stdev)

# Compute lower & upper errors
lower = np.minimum(std, avg)   # prevents lower bound from going below zero
upper = std

errors = [lower, upper]

plt.figure(figsize=(10,5))
x = np.arange(len(videos))

plt.bar(x, avg, yerr=errors, capsize=5, color=["green"])
plt.ylim(0, 8)
plt.xticks(x, videos, rotation=45)
# plt.ylabel("Magnitude of Valence Change")
# plt.title("Average Magnitude of Valence Change per Participant (SD Error Bars, Floor at 0)")

plt.ylabel("Magnitude of Word Changes", fontweight='bold')
plt.xlabel("Video #", fontweight='bold')
plt.title("Average Magnitude of Label Changes per Participant (SD Error Bars, Floor at 0)")


plt.tight_layout()
plt.show()
