# CLAUDE.md

## Coding conventions

- Keep comments and docstrings brief: about one line, max one sentence.
- No sentence in a comment should run past one line.

## Resource usage when running code locally

- Before running code, check current memory usage.
- On a CPU machine, don't let the code use more than half the available cores.
- These limits don't apply to cluster job submissions (e.g. SLURM).

## Code vs data locations

- All analysis and plotting code lives under /pollard/home/szhang20/, in the repo it belongs to (e.g. ~/arsenal-chrombpnet/analysis/).
- /pollard/data/ is for data: inputs, model runs, scores and other outputs.
- Exception: SLURM job scripts, local run wrappers and smoke tests may stay next to the runs they launch in /pollard/data/.

## Plots and figures

- Save plots under the home directory, in the repo they belong to (e.g. ~/arsenal-chrombpnet/figs/), not /pollard/data.
- Start every plot or figure filename with the date it was made and an underscore, e.g. 20261014_summary.png.
- Use Helvetica (Arial is fine); on the pollard server Nimbus Sans is the installed fallback.
