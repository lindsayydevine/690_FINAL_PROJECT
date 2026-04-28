# Checkpoints

The pretrained BioPM / 50MR checkpoint is included:

```
checkpoints/checkpoint.pt    (5.6 MB)
checkpoints/stage2_decoder_best.pt
checkpoints/stage2_decoder_hierarchical.pt
checkpoints/stage1_prior_decoder_bestbody.pt
```

This file contains the pretrained weights for `encoder_acc` (the
movement-element transformer, trained with 50% masking rate).

It is a PyTorch state dict saved with `torch.save()` containing only
the `encoder_acc` weights (not the classifier head or gravity CNN).

The scripts in this package default to this path.  For example:

```bash
python scripts/extract_features.py \
    --checkpoint checkpoints/checkpoint.pt \
    ...
```

Project checkpoints added during the synthetic-generation work:

- `checkpoints/stage2_decoder_best.pt`
  Best overall Stage 2 decoder checkpoint from the medium token-to-window run.
- `checkpoints/stage2_decoder_hierarchical.pt`
  Latest hierarchical Stage 2 experiment using a Stage 1 body prior plus a
  dedicated gravity branch.
- `checkpoints/stage1_prior_decoder_bestbody.pt`
  Best Stage 1 prior-aware decoder checkpoint used to build the hierarchical
  Stage 2 body scaffold.
