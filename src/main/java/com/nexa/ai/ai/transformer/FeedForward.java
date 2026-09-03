package com.nexa.ai.ai.transformer;

public class FeedForward {

    private final int embeddingSize;
    private final int hiddenSize;

    private final double[][] firstWeights;
    private final double[] firstBias;

    private final double[][] secondWeights;
    private final double[] secondBias;

    public FeedForward(
            int embeddingSize,
            int hiddenSize
    ) {

        if (embeddingSize <= 0) {
            throw new IllegalArgumentException(
                    "Embedding size must be greater than zero"
            );
        }

        if (hiddenSize <= 0) {
            throw new IllegalArgumentException(
                    "Hidden size must be greater than zero"
            );
        }

        this.embeddingSize = embeddingSize;
        this.hiddenSize = hiddenSize;

        this.firstWeights =
                createWeights(
                        embeddingSize,
                        hiddenSize
                );

        this.firstBias =
                new double[hiddenSize];

        this.secondWeights =
                createWeights(
                        hiddenSize,
                        embeddingSize
                );

        this.secondBias =
                new double[embeddingSize];
    }

    public double[][] forward(double[][] input) {

        validateInput(input);

        double[][] hidden =
                multiply(
                        input,
                        firstWeights
                );

        addBias(
                hidden,
                firstBias
        );

        relu(hidden);

        double[][] output =
                multiply(
                        hidden,
                        secondWeights
                );

        addBias(
                output,
                secondBias
        );

        return output;
    }

    private double[][] createWeights(
            int inputSize,
            int outputSize
    ) {

        double[][] weights =
                new double[inputSize][outputSize];

        double limit =
                Math.sqrt(
                        6.0 /
                                (inputSize + outputSize)
                );

        for (int i = 0; i < inputSize; i++) {

            for (int j = 0; j < outputSize; j++) {

                weights[i][j] =
                        -limit
                                + Math.random()
                                * (2.0 * limit);
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

    private void relu(double[][] matrix) {

        for (int i = 0; i < matrix.length; i++) {

            for (int j = 0; j < matrix[i].length; j++) {

                matrix[i][j] =
                        Math.max(
                                0.0,
                                matrix[i][j]
                        );
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

    public int getHiddenSize() {
        return hiddenSize;
    }
}