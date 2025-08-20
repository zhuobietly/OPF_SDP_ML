
import os
import pandas as pd
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from utils.metric import *
from utils.feature import *
from utils.plot import *
from sklearn.decomposition import PCA

##——————————————————————————————Data load and oringinal process————————————————————————————##
experiment_name = "Alpha"  # Example experiment name, adjust as needed
df = aggregate_df("data/Alpha")  # Adjust the path to your directory
sol = solve_times_per_profile_df(df)
print(sol.columns)
print(sol.shape)
print("pivoting solve times")
pivot = pivot_solve_times(sol)
print(pivot.columns)
print(pivot.shape)

##——————————————————————————————network feature process————————————————————————————————————##
# 查看文件夹中有多少个文件
load_dir = "data/network_demand/pglib_opf_case2746wop_k_A"
df_2 = features_from_df_Y(pivot, load_dir)
#print(df_2.head())
print(df_2.shape)
print(df_2.columns)
#save the DataFrame to a CSV file
output_file = "data/feature/XGboost/feature_alpha_profiles.csv"
df_2.to_csv(output_file, index=False)
print(f"DataFrame saved to {output_file}")
#这步其实没起到作用，因为case本来就没重复 
print(df_2["case"].unique())
print(df_2.shape)
counts = df_2.groupby('case').size()
print("Samples per profile:\n", counts)
stats = compute_profile_stat(df_2)
print(stats.head())
print(stats.columns)
import matplotlib.pyplot as plt



##——————————————————————————————draw box picture————————————————————————————————————##
# print(sort_by_plot(stats, 'p_mean', 4))

df_regret = compute_regret(stats)

for s in ["p_mean", "p_std", "p_min", "p_max",
          "q_mean", "q_std", "q_min", "q_max"]:
    sort_by_plot_subplots(df_regret, s, 10, "Regret")


counts_df, prop_df = summarize_best_strategies(df_regret, 'p_mean', 4)
print("Counts of best strategies per chunk:\n", counts_df)
print("Proportions of best strategies per chunk:\n", prop_df)
variance_dict = {}
for s in ["p_mean", "p_std", "p_min", "p_max",
          "q_mean", "q_std", "q_min", "q_max"]:
    counts_df, prop_df = summarize_best_strategies(df_regret, s, 10)
    print(f"Counts of best strategies per chunk for {s}:\n", counts_df)
    print(f"Proportions of best strategies per chunk for {s}:\n", prop_df)
    var_series = prop_df.var(axis=0)
    
    variance_dict[s] = var_series

# Build a DataFrame out of it
var_df = pd.DataFrame(variance_dict).T
var_df.index.name = "sort_key"
#Make sure the values are floats
var_df = var_df.astype(float)

# Also record which strategy had the highest variance for each sort_key
var_df["top_strategy"]   = var_df.idxmax(axis=1)
var_df["max_variance"] = var_df.max(axis=1, numeric_only=True)

# Sort descending by max_variance to surface the most “interesting” stats
var_df = var_df.sort_values("max_variance", ascending=False)
print(var_df)

print(df_2.head())
strat_cols = [c for c in df_2.columns if c.startswith("Chordal_")]
df_regret = df_2[strat_cols + ["case"]]
df_regret = compute_regret(df_regret)
print(df_regret.head())


#Print all columns starting by "Chordal_" in df_2
print(df_2.columns)
print(df_2.columns[df_2.columns.str.startswith("Chordal_")])
#df_2.drop(columns=["Chordal_MFI_false", "Chordal_AMD_true", "Chordal_MFI_true", "Chordal_AMD_false"], inplace=True)

#Make sure the rows are ordered the same by the case column in df_2 and df_regret
#let X and Y in the same order
df_2 = df_2.set_index("case").reindex(df_regret["case"]).reset_index()
print(df_2.head())
print(df_regret.head())

##——————————————————————————————generate X and Y————————————————————————————————————##
#Create X Matrix and a regret vector Y with features from df_2 (except case column)
#df_2.columns[df_2.columns.str.startswith("Chordal_")]
# 找出要保留的列（不是case且不包含Chordal_）

cols_to_keep = [col for col in df_2.columns 
                if col != "case" and not col.startswith("Chordal_") and "bus" not in col]
cols_to_keep = [col for col in stats.columns 
                if col != "case" ]
# 只选择要保留的列
X = df_2[cols_to_keep].values
#case disappears as the index
Y = df_regret.set_index("case").values
print(X.shape)
print(Y.shape)


#Save X_PCA and Y_A as csv
output_file_X = "//home/goatoine/Documents/Lanyue/data/feature/XGboost/new_X_PCA.csv"
output_file_Y = "//home/goatoine/Documents/Lanyue/data/feature/XGboost/new_Y_A.csv"

pd.DataFrame(X).to_csv(output_file_X, index=False)
pd.DataFrame(Y).to_csv(output_file_Y, index=False)

print(f"X_PCA saved to {output_file_X}")
print(f"Y_A saved to {output_file_Y}")