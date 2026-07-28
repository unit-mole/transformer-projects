# Longformer Architecture Overview

Standard Transformer self-attention compares every token with every other token. The computational and memory cost therefore grows quadratically as the sequence becomes longer.

Longformer replaces full self-attention with a sparse pattern. Most tokens use sliding-window local attention, which focuses on nearby tokens. Selected tokens can receive global attention so that they exchange information with the full sequence. For question answering, question tokens can be assigned global attention while document tokens use local attention.

This combination reduces the cost of processing long documents and allows Longformer checkpoints to handle substantially longer contexts than standard BERT-style models. Extremely long documents may still require overlapping windows and answer aggregation.
