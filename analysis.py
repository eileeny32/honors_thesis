import pandas as pd
import scipy
import numpy as np
import matplotlib.pyplot as plt
import os
import librosa
from sklearn.model_selection import train_test_split
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests

"""class_labels = {'heritage': 0, 'control': 1}
info_df = pd.read_csv('LOS_proficiency_Eileen.csv')
info_df = info_df.set_index('Unnamed: 0')
info_df['group2'] = info_df['group2'].map(class_labels)
info_df['Participant'] = info_df['Participant'].str[-2:]
X_train, X_test, y_train, y_test = train_test_split(info_df['Participant'], info_df['group2'], test_size=0.2, random_state=32)

lens = []
X_train_list = []
X_test_list = []
for i in os.listdir('./wav audio trimmed'):
    y, sr = librosa.load(f'./wav audio trimmed/{i}', sr=16000)
    if y.shape[0] < 2336:
        continue
    length = librosa.get_duration(path=f'./wav audio trimmed/{i}')
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=14)
    mfcc = mfcc[1:]
    num = i[-14:-12]
    if num[0] == "_":
        num = '0' + num[1]
    mfcc = np.mean(mfcc, axis=1)
    if int(num) in X_test:
        X_test_list.append(mfcc)
        lens.append(length)
        #y_test_list.append(info_df.loc[info_df['Participant'] == num, 'group2'].iloc[0])
    elif int(num) in X_train:
        X_train_list.append(mfcc)
        lens.append(length)
        #y_train_list.append(info_df.loc[info_df['Participant'] == num, 'group2'].iloc[0])
        #groups_train_list.append(num)

print(len(X_train_list))
print(len(X_test_list))
lens_df = pd.Series(lens)
print(lens_df.info())
print(lens_df.describe())"""

"""roc_auc = [0.5767392113095238, 0.574052039809255, 0.5102261609091593, 0.6166801619433199, 0.45668016194331984,
           0.5766737981374276, 0.5819333436796772, 0.599876002634944, 0.5716478093403948, 0.5099830795262268,
           0.5490443502254526, 0.5727437166793602, 0.5193973135383528, 0.5935641110376735, 0.6006899287012185,
           0.568060844847833, 0.5680885780885782, 0.5323174208262065, 0.5743431705406111, 0.5323174208262065,
           0.5537657764770441, 0.5434158454256227, 0.5353481795687521, 0.5665255680435946, 0.5856568294593889,
           0.5051366543903857, 0.5499437540412943, 0.5210084033613446]

val_scores = [[0.60748857, 0.6000209,  0.61516767, 0.6524502,  0.64092957],
              [0.56427759, 0.60585162, 0.63865387, 0.51653835, 0.54295659],
              [0.57579386, 0.65337513, 0.65808745, 0.60350727, 0.60323505],
              [0.59575275, 0.55458725, 0.62822609, 0.53042894, 0.62870598],
              [0.61259805, 0.61368861, 0.65770826, 0.56924926, 0.62411957],
              [0.57319921, 0.58443051, 0.63841687, 0.62077885, 0.59095004],
              [0.5072351, 0.60177638, 0.67839791, 0.63481201, 0.61871417],
              [0.56988603, 0.56244514, 0.63159142, 0.56030796, 0.58286241],
              [0.60048301, 0.62081505, 0.64960303, 0.52792374, 0.55993038],
              [0.52426002, 0.5799791, 0.58817395, 0.56454438, 0.604095],
              [0.58413668, 0.55373041, 0.64339377, 0.55474765, 0.5953317],
              [0.6017005, 0.54334378, 0.60812893, 0.49894089, 0.62002457],
              [0.6095643,  0.59205852, 0.69536675, 0.60479042, 0.64191237],
              [0.51284354, 0.53951933, 0.66903662, 0.53008269, 0.61359541],
              [0.56264096, 0.60321839, 0.64400995, 0.56442218, 0.56205979],
              [0.57968585, 0.6616092, 0.73188766, 0.4973115, 0.58497133],
              [0.54054647, 0.54578892, 0.6186041, 0.49859465, 0.61058559],
              [0.54984731, 0.61483804, 0.63545444, 0.59073689, 0.61425061],
              [0.56020598, 0.58965517, 0.59488091, 0.61265225, 0.58693694],
              [0.55916811, 0.5923093, 0.67273374, 0.60707157, 0.63024161],
              [0.58355787, 0.66505747, 0.63974405, 0.49107907, 0.60487305],
              [0.63778616, 0.61210031, 0.60777343, 0.63532119, 0.58458231],
              [0.53182444, 0.59648903, 0.73999289, 0.66369302, 0.6260647],
              [0.5522823, 0.58219436, 0.6416874, 0.48720925, 0.53896396],
              [0.60128136, 0.59473354, 0.66595568, 0.63536193, 0.60642916],
              [0.53641499, 0.61880878, 0.683517, 0.57446332, 0.61246929],
              [0.57976568, 0.58271682, 0.63040645, 0.52780154, 0.60804668],
              [0.5486897, 0.61939394, 0.66150018, 0.58562467, 0.61056511],
              [0.55745165, 0.59912226, 0.65384524, 0.59725447, 0.59357084]]

hs_df = pd.read_csv('hs_cam.csv')
cs_df = pd.read_csv('cs_cam.csv')

hs = pd.read_csv('hs_results.csv')
cs = pd.read_csv('cs_results.csv')

hs = pd.concat([hs, hs_df[hs_df.columns[1]]], axis=1)
cs = pd.concat([cs, cs_df[hs_df.columns[1]]], axis=1)

hs.to_csv('hs_results.csv')
cs.to_csv('cs_results.csv')"""

"""hs_df = pd.read_csv('hs_results.csv')
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
#results.to_csv('stats analysis.csv')

colors = ['red' if x < 0 else 'blue' for x in diff_means]
fig, ax = plt.subplots()
hbars = ax.barh(np.arange(1, 14), diff_means, color=colors)
ax.axvline(0, color='black', linewidth=0.8)
ax.bar_label(hbars, [f'p value: {i:.2e}' for i in p_corrected])

plt.tight_layout()
plt.show()

plt.savefig('diff.png')"""

test_scores = pd.read_csv('test_scores.csv')
val_scores = pd.read_csv('val_scores.csv')

print(test_scores.describe())
for i in range(5):
    print(val_scores[val_scores.columns[i]].describe())
print(val_scores.to_numpy().mean())
print(val_scores.to_numpy().std())

