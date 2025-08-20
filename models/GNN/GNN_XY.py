
import os
import pandas as pd
import sys
import re
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from utils.metric_gnn import *
from utils.feature_gnn import *
from utils.plot import *
from sklearn.decomposition import PCA

##——————————————————————————————Data load and oringinal process————————————————————————————##
# 数据加载和初始处理
case_name = "case14"
experiment_name = "solve_time"
experiment_dir = os.path.join("data", experiment_name, case_name)
print(f"Loading data from: {experiment_dir}")

df = aggregate_df(experiment_dir)
sol = solve_times_per_profile_df(df)
print(f"Solution DataFrame: {sol.shape[0]} rows × {sol.shape[1]} columns")
print(f"Columns: {list(sol.columns)}")

print(sol.columns)
print(sol.shape)
print("pivoting solve times")
pivot = pivot_solve_times(sol)
print(pivot.columns)
print(pivot.shape)

##——————————————————————————————network feature process————————————————————————————————————##
# 查看文件夹中有多少个文件
load_dir = f"/home/goatoine/Documents/Lanyue/data/load_profiles/{case_name}"
df_2 = features_from_df_Y(pivot, case_name, load_dir)
#print(df_2.head())
print(df_2.shape)
print(df_2.columns)
#save the DataFrame to a pkl file
output_file = "data/feature/GNN/feature_alpha_profiles.pkl.gz"
df_2.to_pickle(output_file, compression="gzip")
print(f"DataFrame saved to {output_file}")
#这步其实没起到作用，因为case本来就没重复 
print(df_2["case"].unique())
print(df_2.shape)
counts = df_2.groupby('case').size()
print("Samples per profile:\n", counts)




##——————————————————————————————draw box picture————————————————————————————————————##
# print(sort_by_plot(stats, 'p_mean', 4))

df_regret = compute_regret(df_2)

##——————————————————————————————generate X and Y————————————————————————————————————##
#Create X Matrix and a regret vector Y with features from df_2 (except case column)
gnn_X = df_2[["gnn_A_hat", "gnn_X"]]
#case disappears as the index


# 选出含有 'chordal' 的列
chordal_cols = [c for c in df_regret.columns if 'Chordal_' in c]
if not chordal_cols:
    raise ValueError("没找到包含 'chordal' 的列。")
# Y（保留 case 为索引）
Y = df_regret.set_index("case")[chordal_cols]


# Save X_PCA and Y_A as PKL
output_file_X = "/home/goatoine/Documents/Lanyue/data/feature/GNN/gnn_X.pkl"
output_file_Y = "/home/goatoine/Documents/Lanyue/data/feature/GNN/gnn_Y.pkl"

pd.DataFrame(gnn_X).to_pickle(output_file_X)
pd.DataFrame(Y).to_pickle(output_file_Y)

print(f"gnn_X saved to {output_file_X}")
print(f"gnn_Y saved to {output_file_Y}")