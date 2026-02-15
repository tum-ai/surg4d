# Dev Ops
- This is a pixi project. Run all python commands with `pixi run python ...`
- We are developing on the endpoint node of a slurm cluster without GPU. If you want me to run anything that does mid to heavy compute or needs a GPU just tell me the command and I'll run it on the cluster for you.

# Code Style
- This is an ML project, so never code defensively. I want stuff to fail and fix the bug rather than fallbacks (etc. for argument types)
- Never introduce default / fallback values for hydra config entries. Hydra must be single source of truth. Always assume config values you need exist and create them if possible.
- We have a multistep pipeline in which each step only accesses config values from `clips` or its own step! If you need to access values from previous steps "wire them through" using hydra variable resolution.
- Never import modules anywhere but at the top of the file!

# Agent Rules
- Don't write markdown reports if not explicitly asked for it.