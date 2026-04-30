# Checkpoints

Canonical checkpoints for the project:

```text
checkpoints/checkpoint.pt
checkpoints/stage1_decoder.pt
checkpoints/stage2_decoder_best.pt
```

- `checkpoints/checkpoint.pt`
  Pretrained BioPM / 50MR encoder checkpoint for `encoder_acc`.
- `checkpoints/stage1_decoder.pt`
  Team Stage 1 decoder checkpoint for token-to-`x_acc_filt` reconstruction.
- `checkpoints/stage2_decoder_best.pt`
  Chosen Stage 2 decoder checkpoint for token-to-window reconstruction.

The repo should treat the three files above as the primary checkpoints.
