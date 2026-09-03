package com.nexa.ai.ai.transformer;

public class LayerNormalization {

    private final int embeddingSize;

    private final double[] gamma;
    private final double[] beta;

    private static final double EPSILON = 1e-5;

    public LayerNormalization(int embeddingSize) {

        if (embeddingSize <= 0) {
            throw new IllegalArgumentException(
                    "Embedding size must be greater than zero"
            );
        }

        this.embeddingSize = embeddingSize;

        this.gamma = new double[embeddingSize];
        this.beta = new double[embeddingSize];

        initialize();
    }

    private void initialize() {

        // Initially:
        //
        // gamma = 1
        // beta  = 0
        //
        // This means normalization does not
        // initially change the overall scale
        // or shift of the representation.

        for (int i = 0; i < embeddingSize; i++) {
            gamma[i] = 1.0;
            beta[i] = 0.0;
        }
    }

    public double[][] forward(double[][] input) {

        validateInput(input);

        double[][] output =
                new double[input.length][embeddingSize];

        for (int i = 0; i < input.length; i++) {

            double mean =
                    calculateMean(input[i]);

            double variance =
                    calculateVariance(
                            input[i],
                            mean
                    );

            double denominator =
                    Math.sqrt(
                            variance + EPSILON
                    );

            for (int j = 0; j < embeddingSize; j++) {

                double normalized =
                        (input[i][j] - mean)
                                / denominator;

                output[i][j] =
                        gamma[j] * normalized
                                + beta[j];
            }
        }

        return output;
    }

    private double calculateMean(
            double[] row
    ) {

        double sum = 0.0;

        for (double value : row) {
            sum += value;
        }

        return sum / embeddingSize;
    }

    private double calculateVariance(
            double[] row,
            double mean
    ) {

        double sum = 0.0;

        for (double value : row) {

            double difference =
                    value - mean;

            sum +=
                    difference * difference;
        }

        return sum / embeddingSize;
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
}