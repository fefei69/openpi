export XLA_PYTHON_CLIENT_PREALLOCATE=false
python scripts/serve_policy.py policy:checkpoint \
       --policy.config=pi05_rrl_cabinet \
       --policy.dir=checkpoints/25000 # ;(
