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
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, BatchNormalization, GlobalAveragePooling1D
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, StratifiedGroupKFold, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from scikeras.wrappers import KerasClassifier


pd.set_option('display.max_columns', None)
np.random.seed(32)


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
    y, sr = librosa.load(f'./wav audio/{item}')
    intervals = effects.split(y, top_db=30)
    item = item[:-4]
    i = 1
    for interval in intervals:
        non_silent_audio = y[interval[0]:interval[1]]
        sf.write(f'./wav audio trimmed/{i}_{item}_trimmed.wav', non_silent_audio, sr)
        i += 1"""

class_labels = {'HS': 0, 'CS': 1}
info_df = pd.read_csv('LOS_participant_info.csv')
info_df['group'] = info_df['group'].map(class_labels)
info_df['participant'] = info_df['participant'].str[-2:]
X_train, X_test, y_train, y_test = train_test_split(info_df['participant'], info_df['group'], test_size=0.2, random_state=32)

X_test_list = [] # list of MFCCs that correspond to participants in X_test from train_test_split
y_test_list = [] # matched list with X_test_list indicating control/heritage speaker
X_train_list = [] # list of MFCCs that correspond to participants in X_train from train_test_split (data)
y_train_list = [] # matched list with X_train_list indicating control/heritage speaker (labels)
groups_train_list = [] # matched list with X_train_list indicating participant number (groups)
for i in os.listdir('./wav audio trimmed'):
    y, sr = librosa.load(f'./wav audio trimmed/{i}')
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    num = i[-14:-12]
    if num[0] == "_":
        num = '0' + num[1]
    if num == '53':
        continue
    mfcc = np.mean(mfcc, axis=1)
    if num in X_test:
        X_test_list.append(mfcc)
        y_test_list.append(info_df.loc[info_df['participant'] == num, 'group'])
    else:
        X_train_list.append(mfcc)
        y_train_list.append(info_df.loc[info_df['participant'] == num, 'group'])
        groups_train_list.append(num)

skf = StratifiedGroupKFold(n_splits=5)
"""for i, (train_index, test_index) in enumerate(skf.split(X_train_list, y_train_list, groups_train_list)):
    print(f"Fold {i}:")
    print(f"  Train: index={train_index}")
    print(f"  Test:  index={test_index}")"""

model = Sequential([
    Conv1D(8, 3, activation='relu', input_shape=(13, 1)),
    BatchNormalization(),
    Conv1D(16, 3, activation='relu'),
    BatchNormalization(),
    GlobalAveragePooling1D(name='gap'),
    Dense(2, activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['f1_score'])

skmodel = KerasClassifier(model=model, epochs=20, batch_size=8, verbose=0)
scores = cross_val_score(skmodel, X_train_list, y=y_train_list, groups=groups_train_list, scoring='f1', cv=skf)
print(scores)

"""cohens_list = []
for i in hs_df.columns:
    hs_group = hs_df[i]
    cs_group = cs_df[i]
    cohens_d = pg.compute_effsize(x=hs_group, y=cs_group, eftype='cohen')
    cohens_list.append(cohens_d)"""

"""X_train, X_test, y_train, y_test = train_test_split(mfcc_df, participant_list_new, test_size=0.2, random_state=32, stratify=participant_list_new)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=32, stratify=y_train)
X_train = X_train.to_numpy()[..., np.newaxis].astype(np.float32)
X_val = X_val.to_numpy()[..., np.newaxis].astype(np.float32)
X_test = X_test.to_numpy()[..., np.newaxis].astype(np.float32)
X_train = tf.convert_to_tensor(X_train, dtype=tf.float32)
y_train = tf.convert_to_tensor(y_train, dtype=tf.int32)
X_val = tf.convert_to_tensor(X_val, dtype=tf.float32)
y_val = tf.convert_to_tensor(y_val, dtype=tf.int32)
X_test = tf.convert_to_tensor(X_test, dtype=tf.float32)
y_test = tf.convert_to_tensor(y_test, dtype=tf.int32)

# hs_cam_list = []
# cs_cam_list = []

model = Sequential([
    Conv1D(16, 3, activation='relu', input_shape=(13, 1)),
    BatchNormalization(),
    Conv1D(32, 3, activation='relu'),
    BatchNormalization(),
    Conv1D(64, 3, activation='relu', name='last_conv'),
    BatchNormalization(name='last_bn'),
    GlobalAveragePooling1D(name='gap'),
    Dense(2, activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=30, batch_size=16, verbose=0, validation_data=(X_val, y_val))

int_to_labels = {0: 'HS', 1: 'CS'}
y_test_pred = model.predict(X_test)
y_test_pred_class = [np.argmax(i) for i in y_test_pred]
y_test_pred_labels = [int_to_labels[i] for i in y_test_pred_class]"""

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
        sample = hs_x[i:i + 1]
    sample_mapping = class_activation_map(model, cam_model, sample, 0)
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
    sample_mapping = class_activation_map(model, cam_model, sample, 1)
    cs_mapping.append(sample_mapping)
cs_mapping = np.array(cs_mapping)
cs_mapping = np.mean(cs_mapping, axis=0)
cs_cam_list.append(list(cs_mapping))

hs_cam_arr = np.array(hs_cam_list).squeeze()
hs_cam_df = pd.DataFrame(hs_cam_arr)
hs_cam_df.to_csv("hs_cam.csv")
cs_cam_arr = np.array(cs_cam_list).squeeze()
cs_cam_df = pd.DataFrame(cs_cam_arr)
cs_cam_df.to_csv("cs_cam.csv")


conf_matrix = confusion_matrix(y_test, y_test_pred_class)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

cam_model = Model(inputs=model.inputs, outputs=model.get_layer('last_conv').output)
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
cs_mapping = np.mean(cs_mapping, axis=0)

plt.figure(figsize=(8, 3))
plt.plot(hs_mapping, label='HS')
plt.plot(cs_mapping, label='CS')
plt.xlabel('MFCC Index (0-Based Indexing)')
plt.ylabel('Importance')
plt.title('Average Importance of MFCCs')
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 8))
sns.heatmap(hs_mapping, annot=True, cbar_kws={'label': 'Importance'})
plt.ylabel('MFCC Index (0-Based Indexing)')
plt.title('Relative Importance of MFCCs in HS Heatmap')
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 8))
sns.heatmap(cs_mapping, annot=True, cbar_kws={'label': 'Importance'})
plt.ylabel('MFCC Index (0-Based Indexing)')
plt.title('Relative Average Importance of MFCCs in CS Heatmap')
plt.tight_layout()
plt.show()
"""
