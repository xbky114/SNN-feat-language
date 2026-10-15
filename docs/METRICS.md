## Semantic depth

Restrict logits to the image's path and apply softmax:

\[
D = \sum_{k=0}^{N} P_k\alpha_k, \qquad \alpha_k = \frac{k}{L}.
\]

Here (k=0) is the root and (k=L+1) is the leaf. Report the dataset mean
of (D).

## Classification accuracy

For each granularity (k=1,\ldots,N), classify among all concepts in level
(k). (A_k) is the fraction whose prediction equals the ground-truth
representative at that level.

## Semantic selectivity

For each image, find the largest logit among the (L+1) path concepts. For
each (k=1,\ldots,Nc), (S_k) is the fraction of images whose winning
concept is in bin (k). The root (bin 0) is excluded from reported
selectivity values.

All metrics are averaged over the evaluation set and may be reported per
timestep for SNN evaluation.
