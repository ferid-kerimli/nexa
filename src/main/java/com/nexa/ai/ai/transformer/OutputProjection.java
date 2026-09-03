package com.nexa.ai.ai.transformer;

import java.util.Random;

public class OutputProjection {

    private final int embeddingSize;
    private final int vocabularySize;

    private final double[][] weights;
    private final double[] bias;

    private final Random random;

    public OutputProjection(
            int embeddingSize,
            int vocabularySize
    ) {

        if (embeddingSize <= 0) {
            throw new IllegalArgumentException(
                    "Embedding size must be greater than zero"
            );
        }

        if (vocabularySize <= 0) {
            throw new IllegalArgumentException(
                    "Vocabulary size must be greater than zero"
            );
        }

        this.embeddingSize = embeddingSize;
        this.vocabularySize = vocabularySize;

        this.weights =
                new double[embeddingSize][vocabularySize];

        this.bias =
                new double[vocabularySize];

        this.random = new Random();

        initialize();
    }

    private void initialize() {

        double limit =
                Math.sqrt(
                        6.0 /
                                (embeddingSize + vocabularySize)
                );

        for (int i = 0; i < embeddingSize; i++) {

            for (int j = 0; j < vocabularySize; j++) {

                weights[i][j] =
                        -limit
                                + random.nextDouble()
                                * (2.0 * limit);
            }
        }
    }

    public double[][] forward(
            double[][] input
    ) {

        validateInput(input);

        double[][] logits =
                multiply(
                        input,
                        weights
                );

        addBias(
                logits,
                bias
        );

        return logits;
    }

    private double[][] multiply(
            double[][] a,
            double[][] b
    ) {

        if (a[0].length != b.length) {
            throw new IllegalArgumentException(
                    "Matrix dimensions do not match"
            );
        }

        double[][] result =
                new double[a.length][b[0].length];

        for (int i = 0; i < a.length; i++) {

            for (int j = 0; j < b[0].length; j++) {

                double sum = 0.0;

                for (int k = 0; k < b.length; k++) {

                    sum +=
                            a[i][k]
                                    * b[k][j];
                }

                result[i][j] = sum;
            }
        }

        return result;
    }

    private void addBias(
            double[][] matrix,
            double[] bias
    ) {

        for (int i = 0; i < matrix.length; i++) {

            for (int j = 0; j < matrix[i].length; j++) {

                matrix[i][j] += bias[j];
            }
        }
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

    public int getVocabularySize() {
        return vocabularySize;
    }
}