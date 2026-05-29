# Architecture

The model adapts an open-world newness detector to 1-D Mill tool-wear sensor
windows.

## Pipeline

1. `src/data/dataset.py` windows multi-channel sensor signals and builds the
   known/unknown split.
2. `src/models/transformer.py` converts each window into 1-D patches and passes
   them through a transformer encoder/decoder with learnable object queries.
3. `src/models/pam.py` compares query features against class prototypes through
   Prototype-Attention Memory.
4. `src/models/newness_model.py` combines objectness, class logits, free energy,
   prototype distance, and max-softmax probability into per-query predictions.
5. `src/evaluation/metrics.py` reports known recall, unknown precision/recall,
   AUROC/AUPR, and threshold-sweep operating points.

## Code Map

| Concept | Implementation |
| --- | --- |
| 1-D patch embedding | `PatchEmbed1D` |
| Transformer encoder/decoder | `TransformerEncoderDecoder` |
| Learnable object queries | `TransformerEncoderDecoder.query_embed` |
| Prototype memory | `PrototypeAttentionMemory` |
| Free-energy score | `energy_score` |
| Fused newness score | `NewnessTransformer.forward` |
| Open-world prediction gate | `NewnessTransformer.predict` |
| Hungarian matching loss | `HungarianMatcher` |
| Training loop and checkpointing | `Trainer` |

## Open-World Split

The default experiment trains on known classes `[0, 1]` and holds out class `2`
as the unknown class for validation and test. Unknown samples are mapped to
label `-1` for reporting.
