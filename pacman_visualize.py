import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("data/pacman_history.csv", parse_dates=["timestamp"])
df["action"] = df["action"].str.lower()
df["month"] = df["timestamp"].dt.to_period("M")

# Draw monthly actions
#
monthly_counts = df.groupby(["month", "action"]).size().unstack()
monthly_counts.plot(kind="bar", figsize=(32, 18))
plt.title("Package Events Per Month")
plt.xlabel("Month")
plt.ylabel("# Events")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("visualizations/events_per_month.svg")

# Draw top packages based on each action
#
fig, axes = plt.subplots(2, 2, figsize=(24, 16))
axes = axes.flatten()
for i, action in enumerate(df["action"].unique()):
    data = df[df["action"] == action]["package"].value_counts().head(20)
    data.plot(kind="barh", ax=axes[i], title=action.capitalize())
    axes[i].set_ylabel("Package")
    axes[i].set_xlabel("# Changes")
    axes[i].invert_yaxis()
plt.suptitle("Top 20 Most Modified Packages By Action")
plt.tight_layout()
plt.savefig("visualizations/top_packages.svg")
