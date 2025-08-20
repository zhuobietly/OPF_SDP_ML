import os
import pandas as pd


def features_from_df_Y(df_Y, load_dir):
    """df_Y : pd.DataFrame a dataframe with the cases and their labels
    load_dir: str, the directory where the load profiles are stored
    …
    Returns a DataFrame with the features for each case and their labels.
    如果你要改y和x的配对估计是在这里
    """

    
    features = []
    cols = df_Y.columns.tolist()
    # identify all the 'Chordal_*' columns once
    chordal_cols = [c for c in cols if c.startswith("Chordal_")]
   
    for _, row in df_Y.iterrows():
        case = row['profile']

        # if you have a 'best_decomp' label column, handle it
        if "best_decomp" in cols:
            label = row['best_decomp']

        # Load the corresponding load profile
        load_file = os.path.join(load_dir, case)
        if not os.path.exists(load_file):
            print(f"Load file {load_file} does not exist. Skipping case {case}.")
            continue

        with open(load_file, 'r') as f:
            load_data = eval(f.read())
        #print(f"Processing case {case}")

        # build the feature dict for this case
        case_features = {}

        # extract pd, qd for each bus
        for bus_id, bus_data in load_data.items():
            case_features[f"pd_bus_{bus_id}"] = bus_data['pd']
            case_features[f"qd_bus_{bus_id}"] = bus_data['qd']

        # always include the case name
        case_features['case'] = case

        # add either the single 'best_decomp' label...
        if "best_decomp" in cols:
            case_features['label'] = label
        # ...or all the 'Chordal_*' columns dynamically
        else:
            for c in chordal_cols:
                case_features[c] = row[c]

        features.append(case_features)

    return pd.DataFrame(features)

def compute_profile_stat(df):
    # 1) Identify your feature‐columns
    p_cols       = [c for c in df.columns if c.startswith('pd_')]
    q_cols       = [c for c in df.columns if c.startswith('qd_')]
    chordal_cols = [c for c in df.columns if c.startswith('Chordal_')]
    print(f"p_cols: {p_cols}")
    print(f"q_cols: {q_cols}")
    print(f"chordal_cols: {chordal_cols}")

    # 2) Compute the row‐wise statistics
    stats = pd.DataFrame({
        'case'   : df['case'],
        'p_mean' : df[p_cols].mean(axis=1),
        'p_std'  : df[p_cols].std(axis=1, ddof=0),
        'p_min'  : df[p_cols].min(axis=1),
        'p_max'  : df[p_cols].max(axis=1),
        'q_mean' : df[q_cols].mean(axis=1),
        'q_std'  : df[q_cols].std(axis=1, ddof=0),
        'q_min'  : df[q_cols].min(axis=1),
        'q_max'  : df[q_cols].max(axis=1),
    })

    # 3) Append all your Chordal_* columns dynamically
    for col in chordal_cols:
        stats[col] = pd.to_numeric(df[col], errors='coerce').values

    print(stats.head())
    return stats