package com.nexa.ai.ai.transformer;

public class TransformerBlock {

    private final int embeddingSize;
    private final int feedForwardSize;

    private final SelfAttention selfAttention;
    private final FeedForward feedForward;

    private final LayerNormalization attentionNormalization;
    private final LayerNormalization feedForwardNormalization;

    public TransformerBlock(
            int embeddingSize,
            int feedForwardSize
    ) {

        if (embeddingSize <= 0) {
            throw new IllegalArgumentException(
                    "Embedding size must be greater than zero"
            );
        }

        if (feedForwardSize <= 0) {
            throw new IllegalArgumentException(
                    "Feed-forward size must be greater than zero"
            );
        }

        this.embeddingSize = embeddingSize;
        this.feedForwardSize = feedForwardSize;

        this.selfAttention =
                new SelfAttention(embeddingSize);

        this.feedForward =
                new FeedForward(
                        embeddingSize,
                        feedForwardSize
                );

        this.attentionNormalization =
                new LayerNormalization(
                        embeddingSize
                );

        this.feedForwardNormalization =
                new LayerNormalization(
                        embeddingSize
                );
    }

    public double[][] forward(
            double[][] input
    ) {
        return forward(input, true);
    }

    public double[][] forward(
            double[][] input,
            boolean causal
    ) {

        validateInput(input);

        /*
         * First sub-layer:
         *
         * Attention
         *     ↓
         * Add original input
         *     ↓
         * LayerNorm
         */

        double[][] attentionOutput =
                selfAttention.forward(
                        input,
                        causal
                );

        double[][] attentionResidual =
                add(
                        input,
                        attentionOutput
                );

        double[][] normalizedAttention =
                attentionNormalization.forward(
                        attentionResidual
                );

        /*
         * Second sub-layer:
         *
         * FeedForward
         *     ↓
         * Add previous representation
         *     ↓
         * LayerNorm
         */

        double[][] feedForwardOutput =
                feedForward.forward(
                        normalizedAttention
                );

        double[][] feedForwardResidual =
                add(
                        normalizedAttention,
                        feedForwardOutput
                );

        return feedForwardNormalization.forward(
                feedForwardResidual
        );
    }

    private double[][] add(
            double[][] a,
            double[][] b
    ) {

        if (a.length != b.length) {
            throw new IllegalArgumentException(
                    "Matrix row counts do not match"
            );
        }

        double[][] result =
                new double[a.length][embeddingSize];

        for (int i = 0; i < a.length; i++) {

            if (a[i].length != embeddingSize ||
                    b[i].length != embeddingSize) {

                throw new IllegalArgumentException(
                        "Matrix dimensions do not match"
                );
            }

            for (int j = 0; j < embeddingSize; j++) {

                result[i][j] =
                        a[i][j] + b[i][j];
            }
        }

        return result;
    }

    private void validateInput(
            double[][] input
    ) {

        if (input == null ||
                input.length == 0) {

            throw new IllegalArgumentException(
                    "Input cannot be null or empty"
            );
        }

        for (double[] row : input) {

            if (row == null ||
                    row.length != embeddingSize) {

                throw new IllegalArgumentException(
                        "Input dimensions do not match embedding size"
                );
            }
        }
    }

    public int getEmbeddingSize() {
        return embeddingSize;
    }

    public int getFeedForwardSize() {
        return feedForwardSize;
    }

    public SelfAttention getSelfAttention() {
        return selfAttention;
    }
}