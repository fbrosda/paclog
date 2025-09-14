import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("data/pacman_history.csv", parse_dates=["timestamp"])
df["action"] = df["action"].str.lower()
df["month"] = df["timestamp"].dt.to_period("M")

monthly_counts = df.groupby(["month", "action"]).size().unstack()
monthly_counts.plot(kind="bar", figsize=(32, 18))
plt.title("Package Events Per Month")
plt.xlabel("Month")
plt.ylabel("# Events")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("visualizations/events_per_month.svg")
# plt.show()

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
# plt.show()

# # -------------------------------
# # 3. 📊 Action type distribution
# # -------------------------------
# action_counts = df["action"].value_counts()

# plt.figure(figsize=(6, 6))
# action_counts.plot(kind="pie", autopct="%1.1f%%", startangle=140)
# plt.title("📊 Distribution of Package Actions")
# plt.ylabel("")
# plt.tight_layout()
# plt.savefig("action_distribution.png")
# plt.show()

# # -------------------------------
# # 4. ⏱️ Time between upgrades for most upgraded packages (optional)
# # -------------------------------
# # Focus on upgrades only
# upgrade_df = df[df["action"] == "upgraded"]

# # Example: Top 3 most upgraded packages
# top_upgraded = upgrade_df["package"].value_counts().head(3).index

# plt.figure(figsize=(10, 6))

# for pkg in top_upgraded:
#     pkg_times = upgrade_df[upgrade_df["package"] == pkg]["timestamp"].sort_values()
#     if len(pkg_times) < 2:
#         continue
#     time_deltas = pkg_times.diff().dropna().dt.days
#     plt.plot(time_deltas.values, label=pkg, marker='o')

# plt.title("⏱️ Days Between Upgrades for Top Packages")
# plt.xlabel("Upgrade Instance")
# plt.ylabel("Days Between Upgrades")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.savefig("upgrade_intervals.png")
# plt.show()
