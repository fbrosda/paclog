import pandas as pd
import random as rand
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

df = pd.read_csv("data/pacman_history.csv", parse_dates=["timestamp"])
df.sort_values(by=["package", "timestamp"], inplace=True)

install_periods = []
for package, group in df.groupby("package"):
    installed_at = None
    for _, row in group.iterrows():
        if row["action"] == "installed":
            installed_at = row["timestamp"]
        elif row["action"] == "removed" and installed_at is not None:
            install_periods.append((package, installed_at, row["timestamp"]))
            installed_at = None
    now = pd.Timestamp.now()
    if installed_at is not None:
        install_periods.append((package, installed_at, now))


periods_df = pd.DataFrame(install_periods, columns=["package", "start", "end"])
packages = sorted(periods_df["package"].unique())
package_to_y = {pkg: i for i, pkg in enumerate(packages)}
periods_df["y"] = periods_df["package"].map(package_to_y)

cmap = plt.get_cmap("viridis")
package_colors = {
    pkg: cmap(rand.uniform(0, 1)) for i, pkg in enumerate(packages)
}


fig, ax = plt.subplots(figsize=(16, len(packages) * 0.4))
for _, row in periods_df.iterrows():
    ax.barh(
        y=row["y"],
        width=row["end"] - row["start"],
        left=row["start"],
        height=0.6,
        align='center',
        color=package_colors[row["package"]],
        edgecolor='black'
    )

ax.set_yticks(range(len(packages)))
ax.set_yticklabels(packages)
ax.invert_yaxis()
ax.set_ylim(len(packages) - 0.5, -0.5)
ax.set_xlabel("Time")
ax.set_title("Package Installation Periods")

ax.set_xlim(df["timestamp"].min(), now)
ax.xaxis.set_major_locator(mdates.AutoDateLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig("visualizations/timeline.svg", format="svg", bbox_inches='tight')

