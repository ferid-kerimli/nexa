package com.nexa.ai.ai.transformer;

import java.util.Random;

public class SelfAttention {

    private final int embeddingSize;

    private final double[][] queryWeights;
    private final double[][] keyWeights;
    private final double[][] valueWeights;
    private double[][] lastAttentionWeights;

    public SelfAttention(int embeddingSize) {

        if (embeddingSize <= 0) {
            throw new IllegalArgumentException(
                    "Embedding size must be greater than zero"
            );
        }

        this.embeddingSize = embeddingSize;

        this.queryWeights =
                createWeights(embeddingSize);

        this.keyWeights =
                createWeights(embeddingSize);

        this.valueWeights =
                createWeights(embeddingSize);
    }

    public double[][] forward(double[][] input) {
        return forward(input, false);
    }

    public double[][] forward(
            double[][] input,
            boolean causal
    ) {

        validateInput(input);

        double[][] queries =
                multiply(input, queryWeights);

        double[][] keys =
                multiply(input, keyWeights);

        double[][] values =
                multiply(input, valueWeights);

        double[][] scores =
                multiply(
                        queries,
                        transpose(keys)
                );

        double scale =
                Math.sqrt(embeddingSize);

        for (int i = 0; i < scores.length; i++) {

            for (int j = 0; j < scores[i].length; j++) {

                scores[i][j] /= scale;

                if (causal && j > i) {
                    scores[i][j] = Double.NEGATIVE_INFINITY;
                }
            }
        }

        lastAttentionWeights =
                softmaxRows(scores);

        return multiply(
                lastAttentionWeights,
                values
        );
    }

    private double[][] createWeights(int size) {

        double[][] weights =
                new double[size][size];

        Random random = new Random();

        double limit =
                Math.sqrt(6.0 / (size + size));

        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {

                weights[i][j] =
                        -limit
                                + random.nextDouble()
                                * (2 * limit);
            }
        }

        return weights;
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
                    sum += a[i][k] * b[k][j];
                }

                result[i][j] = sum;
            }
        }

        return result;
    }

    private double[][] transpose(double[][] matrix) {

        double[][] result =
                new double[matrix[0].length][matrix.length];

        for (int i = 0; i < matrix.length; i++) {
            for (int j = 0; j < matrix[i].length; j++) {
                result[j][i] = matrix[i][j];
            }
        }

        return result;
    }

    private double[][] softmaxRows(double[][] input) {

        double[][] result =
                new double[input.length][input[0].length];

        for (int i = 0; i < input.length; i++) {

            double max = input[i][0];

            for (int j = 1; j < input[i].length; j++) {
                max = Math.max(max, input[i][j]);
            }

            double sum = 0.0;

            for (int j = 0; j < input[i].length; j++) {

                result[i][j] =
                        Math.exp(input[i][j] - max);

                sum += result[i][j];
            }

            for (int j = 0; j < input[i].length; j++) {
                result[i][j] /= sum;
            }
        }

        return result;
    }

    private void validateInput(double[][] input) {

        if (input == null || input.length == 0) {
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

    public double[][] getLastAttentionWeights() {

        if (lastAttentionWeights == null) {
            throw new IllegalStateException(
                    "forward() must be called before accessing attention weights"
            );
        }

        double[][] copy =
                new double[lastAttentionWeights.length][];

        for (int i = 0; i < lastAttentionWeights.length; i++) {
            copy[i] =
                    lastAttentionWeights[i].clone();
        }

        return copy;
    }
}