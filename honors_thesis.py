import shutil
import os
import librosa
from librosa import effects
import numpy as np
import soundfile as sf
import pandas as pd
import pingouin as pg
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, BatchNormalization, GlobalAveragePooling1D, Dropout
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedGroupKFold, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from scikeras.wrappers import KerasClassifier
from pydub import AudioSegment
from pydub.silence import split_on_silence


pd.set_option('display.max_columns', None)
np.random.seed(32)


def integrated_gradients_full(model, X, class_idx, steps=100):
    X = tf.cast(X, tf.float32)
    baseline = tf.zeros_like(X)
    N = X.shape[0]

    alphas = tf.linspace(0.0, 1.0, steps)[:, None, None, None]  # (steps,1,1,1)

    X_exp = tf.expand_dims(X, axis=0)        # (1, N, 13, 1)
    base_exp = tf.expand_dims(baseline, axis=0)

    interpolated = base_exp + alphas * (X_exp - base_exp)
    interpolated = tf.reshape(interpolated, (-1, 13, 1))

    with tf.GradientTape() as tape:
        tape.watch(interpolated)
        preds = model(interpolated, training=False)
        loss = preds[:, class_idx]

    grads = tape.gradient(loss, interpolated)
    grads = tf.reshape(grads, (steps, N, 13, 1))
    avg_grads = tf.reduce_mean(grads, axis=0)  # (N, 13, 1)

    ig = (X - baseline) * avg_grads
    return ig.numpy().squeeze()


def receptive_field_map(x, mapping):
    sums = np.zeros(x.shape[1])
    counts = np.zeros(x.shape[1])
    for i in range(mapping.shape[0]):
        start = i
        end = i + 7
        sums[start:end] += mapping[i]
        counts[start:end] += 1
    rf_map = sums/counts
    rf_map = rf_map.reshape(x.shape[1:])
    return rf_map


def class_activation_map(cnn, cam, x, label):
    cnn_weight = cam(x)
    cnn_weight = cnn_weight.numpy()[0]
    label_weight = cnn.layers[-1].get_weights()[0][:, label]
    mapping = np.dot(cnn_weight, label_weight).reshape((-1, 1))
    mapping = receptive_field_map(x, mapping)
    mapping = np.maximum(mapping, 0)
    return mapping


"""i = 1
file_names = ['frutas_1.wav', 'verduras_2.wav', 'instrumentos musicales_3.wav', 'artículos de ropa_5.wav']
excl = ['wav audio', 'honors_thesis.py']
for item in os.listdir('.'):
    if item in excl:
        continue
    else:
        for file in os.listdir(f'./{item}/recorded_audio'):
            if file in file_names:
                file_new = file[:-4]
                file_new = file_new + '_' + str(i) + '.wav'
                shutil.move(f'./{item}/recorded_audio/{file}', f'./wav audio/{file_new}')
        i += 1"""

"""for item in os.listdir('./wav audio'):
    audio = AudioSegment.from_wav(f'./wav audio/{item}')
    chunks = split_on_silence(
        audio,
        min_silence_len=400,
        silence_thresh=-45
    )
    item = item[:-4]
    for i, chunk in enumerate(chunks):
        chunk.export(f'./wav audio trimmed/{i}_{item}_trimmed.wav', format="wav")"""

class_labels = {'heritage': 1, 'control': 0}
# balanced accuracy, roc_auc_score 
info_df = pd.read_csv('LOS_proficiency_Eileen.csv')
info_df = info_df.set_index('Unnamed: 0')
info_df['group2'] = info_df['group2'].map(class_labels)
info_df['Participant'] = info_df['Participant'].str[-2:]
X_train, X_test, y_train, y_test = train_test_split(info_df['Participant'], info_df['group2'], test_size=0.2, random_state=32)

X_test_list = [] # list of MFCCs that correspond to participants in X_test from train_test_split
y_test_list = [] # matched list with X_test_list indicating control/heritage speaker
X_train_list = [] # list of MFCCs that correspond to participants in X_train from train_test_split (data)
y_train_list = [] # matched list with X_train_list indicating control/heritage speaker (labels)
groups_train_list = [] # matched list with X_train_list indicating participant number (groups)
for i in os.listdir('./wav audio trimmed'):
    y, sr = librosa.load(f'./wav audio trimmed/{i}', sr=16000)
    if y.shape[0] < 2336:
        continue
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=14)
    mfcc = mfcc[1:]
    num = i[-14:-12]
    if num[0] == "_":
        num = '0' + num[1]
    mfcc = np.mean(mfcc, axis=1)
    if int(num) in X_test:
        X_test_list.append(mfcc)
        y_test_list.append(info_df.loc[info_df['Participant'] == num, 'group2'].iloc[0])
    elif int(num) in X_train:
        X_train_list.append(mfcc)
        y_train_list.append(info_df.loc[info_df['Participant'] == num, 'group2'].iloc[0])
        groups_train_list.append(num)
X_train_list = np.array(X_train_list)
X_test_list = np.array(X_test_list)
scale = StandardScaler()
X_train_list = scale.fit_transform(X_train_list)
X_test_list = scale.transform(X_test_list)
X_test_df = pd.DataFrame(X_test_list)
X_test_df.to_csv('test.csv')
y_train_list = np.array(y_train_list).reshape(-1,1)

test_scores = []
val_scores = []

for i in range(30):
    skf = StratifiedGroupKFold(n_splits=5)
    """for i, (train_index, test_index) in enumerate(skf.split(X_train_list, y_train_list, groups_train_list)):
        print(f"Fold {i}:")
        print(f"  Train: index={train_index}")
        print(f"  Test:  index={test_index}")"""

    model = Sequential([
        Conv1D(4, 3, activation='relu', input_shape=(13, 1)),
        BatchNormalization(),
        Conv1D(8, 3, activation='relu'),
        BatchNormalization(),
        Conv1D(16, 3, activation='relu', name='last_conv'),
        BatchNormalization(),
        GlobalAveragePooling1D(name='gap'),
        Dense(2, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=[tf.keras.metrics.AUC(multi_label=True)])

    skmodel = KerasClassifier(model=model, epochs=64, batch_size=4, verbose=0)
    scores = cross_val_score(skmodel, X_train_list, y=y_train_list, groups=groups_train_list, scoring='roc_auc', cv=skf)
    print(scores)
    val_scores.append(scores)
    # scores2 = cross_val_score(skmodel, X_train_list, y=y_train_list, groups=groups_train_list, scoring='balanced_accuracy', cv=skf)
    # print(scores2)

    """cohens_list = []
    for i in hs_df.columns:
        hs_group = hs_df[i]
        cs_group = cs_df[i]
        cohens_d = pg.compute_effsize(x=hs_group, y=cs_group, eftype='cohen')
        cohens_list.append(cohens_d)"""

    """X_train, X_test, y_train, y_test = train_test_split(mfcc_df, participant_list_new, test_size=0.2, random_state=32, stratify=participant_list_new)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=32, stratify=y_train)
    X_train_list = X_train_list.to_numpy()[..., np.newaxis].astype(np.float32)
    X_test_list = X_test_list.to_numpy()[..., np.newaxis].astype(np.float32)
    X_train_list = tf.convert_to_tensor(X_train_list, dtype=tf.float32)
    y_train_list = tf.convert_to_tensor(y_train_list, dtype=tf.int32)
    X_test_list = tf.convert_to_tensor(X_test_list, dtype=tf.float32)
    y_test_list = tf.convert_to_tensor(y_test_list, dtype=tf.int32)"""

    hs_cam_list = []
    cs_cam_list = []

    model = Sequential([
        Conv1D(4, 3, activation='relu', input_shape=(13, 1)),
        BatchNormalization(),
        Conv1D(8, 3, activation='relu'),
        BatchNormalization(),
        Conv1D(16, 3, activation='relu', name='last_conv'),
        BatchNormalization(name='last_bn'),
        GlobalAveragePooling1D(name='gap'),
        Dense(2, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=[tf.keras.metrics.AUC(multi_label=True)])

    model.fit(X_train_list, y_train_list, epochs=64, batch_size=4, verbose=0)

    int_to_labels = {1: 'HS', 0: 'CS'}
    y_test_pred = model.predict(X_test_list)
    y_test_pred_class = [np.argmax(i) for i in y_test_pred]
    y_test_pred_labels = [int_to_labels[int(i)] for i in y_test_pred_class]
    print(roc_auc_score(y_test_pred_class, y_test_list))
    test_scores.append(roc_auc_score(y_test_pred_class, y_test_list))

    hs_mask = y_test_list == np.int64(1)
    cs_mask = y_test_list == np.int64(0)
    hs_x = tf.boolean_mask(X_test_list, hs_mask, axis=0)
    cs_x = tf.boolean_mask(X_test_list, cs_mask, axis=0)
    hs_x = np.array(hs_x)
    cs_x = np.array(cs_x)
    hs_x = hs_x.reshape(hs_x.shape[0], hs_x.shape[1], 1)
    cs_x = cs_x.reshape(cs_x.shape[0], cs_x.shape[1], 1)
    hs_ig = integrated_gradients_full(model, hs_x, 1)
    cs_ig = integrated_gradients_full(model, cs_x, 0)
    hs_mean = np.mean(hs_ig, axis=0)
    cs_mean = np.mean(cs_ig, axis=0)

    """hs_mapping = []
    cs_mapping = []
    for i in range(hs_x.shape[0]):
        sample = hs_x[0]
        if i == hs_x.shape[0] - 1:
            sample = hs_x[i:]
        else:
            sample = hs_x[i:i + 1]
        sample_mapping = class_activation_map(model, cam_model, sample, 1)
        hs_mapping.append(sample_mapping)
    hs_mapping = np.array(hs_mapping)
    hs_mapping = np.mean(hs_mapping, axis=0)
    hs_cam_list.append(list(hs_mapping))
    for i in range(cs_x.shape[0]):
        sample = cs_x[0]
        if i == cs_x.shape[0] - 1:
            sample = cs_x[i:]
        else:
            sample = cs_x[i:i + 1]
        sample_mapping = class_activation_map(model, cam_model, sample, 0)
        cs_mapping.append(sample_mapping)
    cs_mapping = np.array(cs_mapping)
    cs_mapping = np.mean(cs_mapping, axis=0)
    cs_cam_list.append(list(cs_mapping))"""
    
    hs_ig_df = pd.DataFrame(hs_mean)
    cs_ig_df = pd.DataFrame(cs_mean)
    hs = pd.read_csv('hs_results.csv')
    cs = pd.read_csv('cs_results.csv')
    hs = pd.concat([hs, hs_ig_df[hs_ig_df.columns[0]]], axis=1)
    cs = pd.concat([cs, cs_ig_df[cs_ig_df.columns[0]]], axis=1)
    hs.to_csv('hs_results.csv')
    cs.to_csv('cs_results.csv')

    conf_matrix = confusion_matrix(y_test_list, y_test_pred_class)
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(f'./confusion matrices/confusion_matrix_{i}.png')

    """cam_model = Model(inputs=model.inputs, outputs=model.get_layer('last_conv').output)
    hs_mask = y_test == 0
    cs_mask = y_test == 1
    hs_x = tf.boolean_mask(X_test, hs_mask, axis=0)
    cs_x = tf.boolean_mask(X_test, cs_mask, axis=0)
    hs_mapping = []
    cs_mapping = []
    for i in range(hs_x.shape[0]):
        sample = hs_x[0]
        if i == hs_x.shape[0] - 1:
            sample = hs_x[i:]
        else:
            sample = hs_x[i:i+1]
        sample_mapping = class_activation_map(model, cam_model, sample, 0)
        hs_mapping.append(sample_mapping)
    hs_mapping = np.array(hs_mapping)
    hs_mapping = np.mean(hs_mapping, axis=0)
    for i in range(cs_x.shape[0]):
        sample = cs_x[0]
        if i == cs_x.shape[0] - 1:
            sample = cs_x[i:]
        else:
            sample = cs_x[i:i+1]
        sample_mapping = class_activation_map(model, cam_model, sample, 1)
        cs_mapping.append(sample_mapping)
    cs_mapping = np.array(cs_mapping)
    cs_mapping = np.mean(cs_mapping, axis=0)"""

    plt.figure(figsize=(8, 3))
    plt.plot(hs_mean, label='HS')
    plt.plot(cs_mean, label='CS')
    plt.xlabel('MFCC Index (0-Based Indexing)')
    plt.ylabel('Importance')
    plt.title('Average Importance of MFCCs')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'./line charts/mfcc_importance_line_plot_{i}.png')

    plt.figure(figsize=(10, 8))
    sns.heatmap(hs_mean.reshape(-1,1), annot=True, cbar_kws={'label': 'Importance'}, yticklabels=range(1, 14))
    plt.ylabel('MFCC Index (0-Based Indexing)')
    plt.title('Relative Importance of MFCCs in HS Heatmap')
    plt.tight_layout()
    plt.savefig(f'./hs heatmaps/mfcc_hs_heatmap_{i}.png')

    plt.figure(figsize=(10, 8))
    sns.heatmap(cs_mean.reshape(-1,1), annot=True, cbar_kws={'label': 'Importance'})
    plt.ylabel('MFCC Index (0-Based Indexing)')
    plt.title('Relative Average Importance of MFCCs in CS Heatmap')
    plt.tight_layout()
    plt.savefig(f'./cs heatmaps/mfcc_cs_heatmap_{i}.png')

test_scores_df = pd.DataFrame(test_scores)
val_scores_df = pd.DataFrame(val_scores)
test_scores_df.to_csv('test_scores.csv')
val_scores_df.to_csv('val_scores.csv')
