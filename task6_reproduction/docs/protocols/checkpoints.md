# Checkpoint protocol

Each scheduled checkpoint is committed atomically only after all payload files are written. `model_state.pt` contains the full wrapped T5 state, router parameters, fixed centroid buffers and G4 beta. `training_state.pt` contains Adam, scheduler, shuffle generator and global Python/NumPy/Torch/CUDA RNG states.

The metadata pins condition, protocol, prepared-input identity, optimizer step, epoch, scheduler dimensions and exact backbone/router parameter-group manifest. Restore uses strict state-dict loading and rejects changed metadata. G4 pending counts must be empty at save and after restore.
