"""What a learning agent needs from the engine: a shared vocabulary and an observation encoder.

Nothing here imports torch. The trainer in `training/` does; the bot and the evaluation harness
only ever need these arrays and an onnxruntime session.
"""
