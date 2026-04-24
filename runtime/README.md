# Runtime Outputs

This directory is reserved for generated experiment outputs.

The training and summary scripts create:

- `runtime/runs/`
- `runtime/summary/`
- `runtime/cache/`

These generated paths are ignored by Git. Do not publish machine-specific
config snapshots, trained weights, prediction arrays, or cache files in the
source repository unless they are intentionally released as a separate artifact.
