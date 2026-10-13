# CLAUDE.md

## Coding conventions

- Keep comments and docstrings brief: about one line, max one sentence.
- No sentence in a comment should run past one line.

## Resource usage when running code locally

- Before running code, check current memory usage.
- On a CPU machine, don't let the code use more than half the available cores.
- These limits don't apply to cluster job submissions (e.g. SLURM).
