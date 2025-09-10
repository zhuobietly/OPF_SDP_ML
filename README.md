# OPF_SDP_ML
# Project Environment Setup

This project requires **Julia** and **Python** with the following
versions and dependencies.

------------------------------------------------------------------------

## Julia Environment

-   **Julia version:** 1.11.6\
-   Dependencies are fully specified in `Project.toml` and
    `Manifest.toml`.

### Setup

From the project root directory:

``` bash
julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'
```

This will install the exact versions of all Julia packages as locked in
`Manifest.toml`.

To check your Julia setup:

``` bash
julia --project=. -e 'using InteractiveUtils; versioninfo()'
```

------------------------------------------------------------------------

## Python Environment

-   **Python version:** 3.11.6\
-   Dependencies are listed in `requirements.txt`.

### Setup

We recommend using a virtual environment:

``` bash
# create and activate venv
python -m venv .venv
source .venv/bin/activate

# install dependencies
pip install -r requirements.txt
```

To check Python version:

``` bash
python -V
```

------------------------------------------------------------------------

## Quick Start

Once both environments are set up:

``` bash
# Run Julia script with project environment
julia --project=. src_jl/loaddate_gen_test.jl
```

If you need Python support inside Julia (via `PyCall` or `PythonCall`),
ensure the correct Python environment is active or configured.

------------------------------------------------------------------------

## Notes

-   Julia and Python versions are pinned to ensure reproducibility.\
-   If you encounter issues with package resolution, delete
    `Manifest.toml` and re-run `Pkg.instantiate()`, though this may
    update to newer versions.\
-   For GPU solvers (e.g., Gurobi), make sure the environment variables
    (like `LD_LIBRARY_PATH`) are properly set, as in the example above.
