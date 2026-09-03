package com.nexa.ai.ai.transformer;

public class PositionalEncoding {

    private final int maxSequenceLength;
    private final int embeddingSize;

    private final double[][] encoding;

    public PositionalEncoding(
            int maxSequenceLength,
            int embeddingSize
    ) {

        if (maxSequenceLength <= 0) {
            throw new IllegalArgumentException(
                    "Maximum sequence length must be greater than zero"
            );
        }

        if (embeddingSize <= 0) {
            throw new IllegalArgumentException(
                    "Embedding size must be greater than zero"
            );
        }

        this.maxSequenceLength = maxSequenceLength;
        this.embeddingSize = embeddingSize;

        this.encoding =
                new double[maxSequenceLength][embeddingSize];

        initialize();
    }

    private void initialize() {

        for (int position = 0;
             position < maxSequenceLength;
             position++) {

            for (int dimension = 0;
                 dimension < embeddingSize;
                 dimension++) {

                double angle =
                        position /
                                Math.pow(
                                        10000.0,
                                        (2.0 * (dimension / 2))
                                                / embeddingSize
                                );

                if (dimension % 2 == 0) {
                    encoding[position][dimension] =
                            Math.sin(angle);
                } else {
                    encoding[position][dimension] =
                            Math.cos(angle);
                }
            }
        }
    }

    public double[][] add(double[][] embeddings) {

        if (embeddings == null) {
            throw new IllegalArgumentException(
                    "Embeddings cannot be null"
            );
        }

        if (embeddings.length > maxSequenceLength) {
            throw new IllegalArgumentException(
                    "Sequence length exceeds maximum sequence length"
            );
        }

        for (double[] row : embeddings) {

            if (row == null ||
                    row.length != embeddingSize) {

                throw new IllegalArgumentException(
                        "Embedding dimensions do not match"
                );
            }
        }

        double[][] result =
                new double[embeddings.length][embeddingSize];

        for (int position = 0;
             position < embeddings.length;
             position++) {

            for (int dimension = 0;
                 dimension < embeddingSize;
                 dimension++) {

                result[position][dimension] =
                        embeddings[position][dimension]
                                + encoding[position][dimension];
            }
        }

        return result;
    }

    public double[][] getEncoding() {

        double[][] copy =
                new double[maxSequenceLength][embeddingSize];

        for (int i = 0;
             i < maxSequenceLength;
             i++) {

            System.arraycopy(
                    encoding[i],
                    0,
                    copy[i],
                    0,
                    embeddingSize
            );
        }

        return copy;
    }

    public int getMaxSequenceLength() {
        return maxSequenceLength;
    }

    public int getEmbeddingSize() {
        return embeddingSize;
    }
}