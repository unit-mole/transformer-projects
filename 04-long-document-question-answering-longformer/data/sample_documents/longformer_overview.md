# Longformer Demonstration Note

Standard full self-attention becomes expensive as sequence length grows because
its computation increases quadratically with the number of tokens. Longformer
uses local sliding-window attention and selected global-attention tokens to
process substantially longer sequences.

For extractive question answering, the question tokens receive global attention
so they can interact with the complete available context. The selected
checkpoint in this project supports sequences up to 4,096 tokens, although the
public CPU demo defaults to a smaller runtime window for responsiveness.

Documents longer than one model window are processed with overlapping token
windows. The application scores answer spans in every window, selects the best
valid span, maps it back to character offsets in the original document, and
displays the paragraph containing that span.

The confidence value shown by the application is a model-based proxy derived
from start-token and end-token probabilities. It is not a calibrated guarantee
that an answer is correct.
