# Dev Ops
- Because of cluster infra, you unfortunately don't have access to our pixi environment. If you need me to run shell commands for checking formats etc. just give me a shell command and ask me to run it for you and paste you the results.

# Code Style
- This is an ML project, so never code defensively. I want stuff to fail and fix the bug rather than fallbacks (etc. for argument types)
- Never introduce default / fallback values for hydra config entries. Hydra must be single source of truth. Always assume config values you need exist and create them if possible.
- We have a multistep pipeline in which each step only accesses config values from `clips` or its own step! If you need to access values from previous steps "wire them through" using hydra variable resolution.
- Never import modules anywhere but at the top of the file!

# Agent Rules
- Don't write markdown reports if not explicitly asked for it.