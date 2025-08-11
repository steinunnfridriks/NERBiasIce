import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib
matplotlib.use('Agg')

# Your results dictionary
results = {
    'eval_loss': 0.017549900338053703, 'eval_macro_precision': 0.9836092654827091, 'eval_macro_recall': 0.9401801380409399, 'eval_macro_f1': 0.9608340736554706, 'eval_micro_precision': 0.9851470142467414, 'eval_micro_recall': 0.9606857818504286, 'eval_micro_f1': 0.9727626459143969, 'eval_ADDICTION_precision': 0.9935760171306209, 'eval_ADDICTION_recall': 0.9809725158562368, 'eval_ADDICTION_f1': 0.9872340425531915, 'eval_DISABILITY_precision': 0.9902557856272838, 'eval_DISABILITY_recall': 0.9655581947743468, 'eval_DISABILITY_f1': 0.9777510523150932, 'eval_GENERAL_precision': 0.9848484848484849, 'eval_GENERAL_recall': 0.9609292502639916, 'eval_GENERAL_f1': 0.9727418492784607, 'eval_LGBTQIA_precision': 0.9937888198757764, 'eval_LGBTQIA_recall': 0.8791208791208791, 'eval_LGBTQIA_f1': 0.9329446064139941, 'eval_LOOKS_precision': 0.9884526558891455, 'eval_LOOKS_recall': 0.9672316384180791, 'eval_LOOKS_f1': 0.9777270131353514, 'eval_ORIGIN_precision': 0.9657142857142857, 'eval_ORIGIN_recall': 0.8284313725490197, 'eval_ORIGIN_f1': 0.8918205804749341, 'eval_PERSONAL_precision': 0.9892857142857143, 'eval_PERSONAL_recall': 0.9685314685314685, 'eval_PERSONAL_f1': 0.9787985865724381, 'eval_PROFANITY_precision': 0.9888623707239459, 'eval_PROFANITY_recall': 0.9865079365079366, 'eval_PROFANITY_f1': 0.9876837504966229, 'eval_RELIGION_precision': 0.9910714285714286, 'eval_RELIGION_recall': 0.8671875, 'eval_RELIGION_f1': 0.9249999999999999, 'eval_SEXUAL_precision': 0.9818913480885312, 'eval_SEXUAL_recall': 0.9207547169811321, 'eval_SEXUAL_f1': 0.9503407984420643, 'eval_SOCIAL_STATUS_precision': 0.9627329192546584, 'eval_SOCIAL_STATUS_recall': 0.9528688524590164, 'eval_SOCIAL_STATUS_f1': 0.9577754891864058, 'eval_STUPIDITY_precision': 0.9877094972067039, 'eval_STUPIDITY_recall': 0.9822222222222222, 'eval_STUPIDITY_f1': 0.984958217270195, 'eval_VULGAR_precision': 0.984504132231405, 'eval_VULGAR_recall': 0.9684959349593496, 'eval_VULGAR_f1': 0.9764344262295082, 'eval_WOMEN_precision': 0.9678362573099415, 'eval_WOMEN_recall': 0.9337094499294781, 'eval_WOMEN_f1': 0.9504666188083274, 'eval_runtime': 41.1121, 'eval_samples_per_second': 218.938, 'eval_steps_per_second': 27.389, 'epoch': 10.0
}

# Extract class names
classes = [
    'ADDICTION', 'DISABILITY', 'GENERAL', 'LGBTQIA', 'LOOKS', 'ORIGIN', 'PERSONAL',
    'PROFANITY', 'RELIGION', 'SEXUAL', 'SOCIAL_STATUS', 'STUPIDITY', 'VULGAR', 'WOMEN'
]

metrics = ['precision', 'recall', 'f1']

# Extract F1 scores for sorting
f1_scores = [(cls, results.get(f'eval_{cls}_f1', np.nan)) for cls in classes]
# Sort classes by F1 score descending
sorted_classes = [cls for cls, _ in sorted(f1_scores, key=lambda x: x[1], reverse=True)]

# Build data matrix in sorted order
data = []
for cls in sorted_classes:
    row = []
    for metric in metrics:
        key = f'eval_{cls}_{metric}'
        row.append(results.get(key, np.nan))
    data.append(row)

# Plot heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(data, annot=True, fmt=".3f", cmap="YlGnBu", xticklabels=metrics, yticklabels=sorted_classes)
ax = plt.gca()  # Get current axes
ax.set_xticklabels(['Precision', 'Recall', 'F1'], rotation=0)
plt.title("Performance by Class")
plt.tight_layout()
plt.savefig("heatmap.png", format="png", dpi=300)