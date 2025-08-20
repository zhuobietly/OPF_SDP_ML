import os
import pandas as pd
#processing the original data
def aggregate_df(dir_path):
    """dir_path: directory with different results csv files.
    Returns a single df where it aggregates all the files but only keeps One header row."""
    import os
    import pandas as pd

    all_files = [f for f in os.listdir(dir_path) if f.endswith('.csv')]
    dfs = []
    for file in all_files:
        file_path = os.path.join(dir_path, file)
        df = pd.read_csv(file_path)
        dfs.append(df)

    # Concatenate all DataFrames
    aggregated_df = pd.concat(dfs, ignore_index=True)
    #Drop the AC and SOC Formulations
    print("Before dropping AC and SOC Formulations:", aggregated_df.shape)
    #Find the amount of AC formulation in the aggregated_df
    ac_count = aggregated_df[aggregated_df['Formulation'] == 'AC'].shape[0]
    print(f"Number of AC entries: {ac_count}")
    
    #Find the amount of INFEASIBLE in the aggregated_df
    infeasible_count = aggregated_df[aggregated_df['Status'] == 'LOCALLY_INFEASIBLE'].shape[0]
    print(f"Number of INFEASIBLE entries: {infeasible_count}")
    print(ac_count - infeasible_count)
    aggregated_df = aggregated_df[~aggregated_df['Formulation'].str.contains('AC|SOC', regex=True)]
    #Drop the lines where there is not 4 entry for the given perturbation (group by perturbation and keep only those with 4 entries)
    aggregated_df = aggregated_df.groupby('perturbation').filter(lambda x: len(x) == 12)
    #Drop the perturbation column
    # Reset index and return
    return aggregated_df.reset_index(drop=True)

def to_decomp(Bool):
    """convert a boolean into true or false,weather merging the cliques or not."""
    if isinstance(Bool, pd.Series):
        return Bool.apply(lambda x: "true" if x else "false")
    else:
        if Bool:
            return "true"
        else:
            return "false"
        
def solve_times_per_profile_df(solver_df):
    """df: DataFrame with teh SolveTime for each Formulation and Merge for each perturbation.
    1. If there is no column "profile" : Creates a profile column with the name
    2. Creates a "decomp" column with the name of the decompositon
    3. Takes Each profile and create a New DataFrame with columns profile, d in decomp.unique and each case is the SolveTime for the given profile and decomp
    4. Returns the new DataFrame with the profile, decomp and the SolveTime
    """
    
    print(solver_df.columns)
    # --- C) Parse perturbation and build profile ID ---
    solver_df[["sigma", "raw_i"]] = (
        solver_df["perturbation"]
        .str.strip("()")
        .str.split(",", expand=True)
    )
    solver_df["sigma"] = solver_df["sigma"].astype(float)
    solver_df["raw_i"] = solver_df["raw_i"].astype(int)

    # Assuming your load‐profile filenames use i = raw_i // 2
    #Check if the column "load_id" exists, if not create it
    if "load_id" not in solver_df.columns:
        solver_df["profile"] = solver_df.apply(
            lambda r: f"{r['Case']}_{r['sigma']}_perturbation_{r['raw_i']}", axis=1
    )
    else:
        solver_df["profile"] = solver_df["load_id"]
    #Here, solver_df["A_parameter"], is a Float and we want to convert it to a decomp with formulation + A°_parameter
    solver_df["decomp"] = solver_df["Formulation"]+"_"+solver_df["A_parameter"].astype(str) 
    return solver_df



def pivot_solve_times(df: pd.DataFrame,
                      profile_col: str = "profile",
                      decomp_col: str = "decomp",
                      value_col: str = "SolveTime") -> pd.DataFrame:
    """
    Pivot df so that each profile is one row, and each decomp strategy
    becomes its own column of solve times.

    Parameters
    ----------
    df
        Original DataFrame with one row per (profile, decomp) pair.
    profile_col
        Name of the column identifying each load profile.
    decomp_col
        Name of the column identifying the decomposition strategy (must have exactly 4 unique values per profile).
    value_col
        Name of the column to fill into the pivot (here SolveTime).

    Returns
    -------
    pd.DataFrame
        index: auto-generated, columns = [profile_col, decomp_1, decomp_2, …].
    """
    # pivot so that decomp values become columns
    pivot = df.pivot(index=profile_col,
                     columns=decomp_col,
                     values=value_col)

    # optional: ensure columns are sorted by decomp strategy
    pivot = pivot.reindex(sorted(pivot.columns), axis=1)

    # rename columns to something more explicit
    pivot.columns = [col for col in pivot.columns]

    # bring profile back as a column
    pivot = pivot.reset_index()

    return pivot

def compute_regret(df: pd.DataFrame,
                   profile_col: str = "profile") -> pd.DataFrame:
    """
    Given a DataFrame with one row per profile and columns
    like 'AMD_true', 'AMD_false', 'MFI_true', 'MFI_false' (plus any
    other cols), returns a new DataFrame where those four strategy
    columns have been replaced by their regret:

        regret_s = (t_solve_s − t_best) / t_best

    where t_best = min_s(t_solve_s) for that row.

    The profile column (and any other cols) are left untouched.
    处理时间变成比例的
    """
    # 1) copy to avoid mutating the original
    df_reg = df.copy()

    # 2) detect the four strategy cols (exactly those ending in _true or _false)
    strat_cols = [c for c in df.columns if c.startswith("Chordal_")]


    # 3) compute the best solve‐time per row
    for col in strat_cols:
        df_reg[col] = pd.to_numeric(df[col], errors='coerce')
    best = df_reg[strat_cols].min(axis=1)
     
    # 4) replace each strategy col by its regret
    df_reg[strat_cols] = (
        df_reg[strat_cols]
        .sub(best, axis=0)
        .div(best, axis=0)
    )

    return df_reg



def summarize_best_strategies(df: pd.DataFrame,
                              sort_by: str,
                              number_of_chunks: int):
    """
    Splits `df` into `number_of_chunks` chunks sorted by `sort_by`, then for each chunk:
      - finds which strategy column has the minimum value on each row
      - counts how many times each strategy is best
      - computes the proportion of wins per strategy

    Returns
    -------
    counts_df : DataFrame, shape (number_of_chunks, S)
        counts_df.loc[chunk, strat] = number of profiles in `chunk` where `strat` was best
    prop_df   : DataFrame, same shape
        prop_df = counts_df / chunk_size  (float)
    """
    # 1) Sort & split
    sorted_df = df.sort_values(by=sort_by, ascending=False)
    N = len(sorted_df)
    chunk_size = N // number_of_chunks
    chunks = [
        sorted_df.iloc[i*chunk_size : (i+1)*chunk_size if i < number_of_chunks-1 else N]
        for i in range(number_of_chunks)
    ]

    # 2) Identify your strategy columns
    strat_cols = [c for c in df.columns if c.startswith("Chordal_")]


    count_rows = []
    prop_rows  = []
    for i, chunk in enumerate(chunks, start=1):
        # which strategy is best on each row?
        best = chunk[strat_cols].idxmin(axis=1)
        counts = best.value_counts().to_dict()
        total = len(chunk)

        # build row dicts
        count_row = { strat: counts.get(strat, 0) for strat in strat_cols }
        prop_row  = { strat: count_row[strat] / total for strat in strat_cols }

        # label by chunk name
        chunk_name = f"{sort_by}_{i}"
        count_row["chunk"] = chunk_name
        prop_row["chunk"]  = chunk_name

        count_rows.append(count_row)
        prop_rows.append(prop_row)

    # 3) Build DataFrames
    counts_df = pd.DataFrame(count_rows).set_index("chunk")[strat_cols]
    prop_df   = pd.DataFrame(prop_rows).set_index("chunk")[strat_cols]

    return counts_df, prop_df