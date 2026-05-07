import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests

hs_df = pd.read_csv('hs_results.csv')
cs_df = pd.read_csv('cs_results.csv')

p_vals = []
stats = []
diff_means = []
for i in range(13):
    hs_vals = hs_df.iloc[i]
    cs_vals = cs_df.iloc[i]

    hs_vals = np.array(hs_vals).squeeze()
    cs_vals = np.array(cs_vals).squeeze()

    diff = hs_vals - cs_vals
    diff_means.append(np.mean(diff))

    stat, p = wilcoxon(diff)
    p_vals.append(p)
    stats.append(stat)

p_vals = np.array(p_vals)
stats = np.array(stats)

reject, p_corrected, _, _ = multipletests(p_vals, method='fdr_bh')

results = pd.DataFrame({
    "feature": np.arange(1, 14),
    "wilcoxon_stat": stats,
    "p_value": p_vals,
    "p_corrected": p_corrected,
    "significant": reject,
    "mean_diff": diff_means
})


def direction_label(x):
    if x > 0:
        return "HS > CS"
    elif x < 0:
        return "CS > HS"
    else:
        return "no difference"


results["direction"] = results["mean_diff"].apply(direction_label)

print(results)
results.to_csv('stats analysis.csv')

colors = ['red' if x < 0 else 'blue' for x in diff_means]
fig, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)
hbars = ax.barh(np.arange(1, 14), diff_means, color=colors)
ax.margins(x=0.3)
ax.axvline(0, color='black', linewidth=0.8)
ax.bar_label(hbars, [f'p value: {i:.2e}' for i in p_corrected])
ax.set_title('Average Difference in Importance Values per MFCC: HS - CS')
ax.set_xlabel('Average Difference')
ax.set_ylabel('MFCC')
plt.savefig('diff.png')
plt.show()


