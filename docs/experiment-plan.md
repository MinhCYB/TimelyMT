# Experiment Plan

The active experiment groups are:

1. Streaming baselines: Fixed-N, Fixed-Time, and LocalAgreement.
2. Policy model comparison: logistic regression, MiniLM, DistilBERT, and XLM-RoBERTa.
3. Policy ablation: no history, source history, source-plus-target history, and technical context.

Talk IDs, rather than randomly sampled sentences, define train/dev/test splits. The learned policy will be compared against heuristic streaming strategies using the shared streaming evaluation workflow.
