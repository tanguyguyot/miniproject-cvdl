import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Updated metrics
df_metrics = pd.DataFrame({
    "Model Type": ["LiDAR", "LiDAR", "LiDAR", "LiDAR", "LiDAR", "RGB", "RGB", "RGB", "RGB"],
    "Model Name": ["v8m", "v8s", "v5m", "v5s", "v8n", "v8s", "v5s", "v8n", "v5n"],
    "Precision": [0.832, 0.816, 0.839, 0.816, 0.623, 0.808, 0.781, 0.876, 0.829],
    "Recall": [0.700, 0.662, 0.669, 0.662, 0.563, 0.745, 0.676, 0.765, 0.784],
    "mAP@0.5": [0.755, 0.727, 0.754, 0.727, 0.574, 0.813, 0.784, 0.861, 0.847],
    "mAP@0.5:0.95": [0.325, 0.321, 0.332, 0.321, 0.208, 0.486, 0.462, 0.491, 0.461]
})

# Updated inference speed
df_speed = pd.DataFrame({
    "Model Type": ["LiDAR", "LiDAR", "LiDAR", "LiDAR", "LiDAR", "RGB", "RGB", "RGB", "RGB"],
    "Model Name": ["v8m", "v8s", "v5m", "v5s", "v8n", "v8s", "v5s", "v8n", "v5n"],
    "Inference Time (s)": [22.90, 7.58, 7.19, 6.20, 6.19, 5.32, 3.96, 4.24, 4.04],
    "Num Images": [197, 197, 197, 197, 197, 46, 46, 46, 46],
    "FPS": [8.60, 25.98, 27.40, 31.78, 31.82, 8.65, 11.62, 10.85, 11.39]
})

# Merge
df = pd.merge(df_metrics, df_speed, on=["Model Type", "Model Name"])

# Plot
plt.figure(figsize=(10, 6))
sns.set(style="whitegrid")

palette = {"LiDAR": "#1f77b4", "RGB": "#ff7f0e"}

sns.scatterplot(
    data=df,
    x="FPS",
    y="mAP@0.5:0.95",
    hue="Model Type",
    style="Model Type",
    palette=palette,
    s=100
)

# Annotate points
for _, row in df.iterrows():
    plt.text(row["FPS"] + 0.3, row["mAP@0.5:0.95"], row["Model Name"], fontsize=9)

# Labels in English
plt.title("YOLO Model Comparison: Speed vs Accuracy (mAP@0.5:0.95)")
plt.xlabel("FPS (Frames per Second)")
plt.ylabel("mAP@0.5:0.95")
plt.legend(title="Model Type")
plt.tight_layout()

# Save the plot
plt.savefig("model_comparison.png", dpi=300)
plt.close()
