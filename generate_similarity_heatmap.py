import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from natsort import natsorted

# CSVファイルからデータを読み込む
data = pd.read_csv('similarity_results.csv')

# ファイル名で自然順ソート
data['File1'] = pd.Categorical(data['File1'], categories=natsorted(data['File1'].unique()), ordered=True)
data['File2'] = pd.Categorical(data['File2'], categories=natsorted(data['File2'].unique()), ordered=True)

# ファイル名でソート
sorted_data = data.sort_values(by=['File1', 'File2'], key=lambda col: natsorted(col))

# ピボットテーブルを作成して類似度スコアを行列形式にする
data_pivot = sorted_data.pivot(index='File1', columns='File2', values='Similarity Score')

# 対称行列にするために、データを補完
data_pivot = data_pivot.combine_first(data_pivot.T).fillna(0)

# 右上を0で埋める
for i in range(len(data_pivot)):
    for j in range(i + 1, len(data_pivot)):
        data_pivot.iloc[i, j] = 0

# ヒートマップの作成
plt.figure(figsize=(12, 10))
sns.heatmap(data_pivot, annot=True, cmap='viridis', linewidths=0.5)
plt.title('HTML File Similarity Heatmap')
plt.xlabel('File2')
plt.ylabel('File1')
plt.show()
