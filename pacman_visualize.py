import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("data/pacman_history.csv", parse_dates=["timestamp"])
df["action"] = df["action"].str.lower()
df["month"] = df["timestamp"].dt.to_period("M")

# Draw hourly distribution of pacman updates
#
hourly_counts = df.groupby(df['timestamp'].dt.hour).size()
hourly_counts.plot(kind="bar", figsize=(14, 8))
plt.title("Package Event Distribution Over The Day")
plt.xlabel("Hour of the day")
plt.ylabel("# Events")
plt.tight_layout()
plt.savefig("visualizations/events_per_hour.svg")

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

# Action Distribution
#
action_counts = df["action"].value_counts()
plt.figure(figsize=(6, 6))
action_counts.plot(kind="pie", autopct="%1.1f%%", startangle=140)
plt.title("Distribution of Package Actions")
plt.ylabel("")
plt.tight_layout()
plt.savefig("visualizations/action_distribution.svg")

# Time between upgrades for most upgraded packages (optional)
#
upgrade_df = df[df["action"] == "upgraded"]
pkg_times = upgrade_df.groupby(["package"]).apply(lambda x: x["timestamp"].sort_values().diff().dropna().dt.days, include_groups=False)
plt.figure(figsize=(6, 10))
pkg_times.plot.box()
plt.title("Upgrade interval")
plt.tight_layout()
plt.savefig("visualizations/upgrade_interval.svg")

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
