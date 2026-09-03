package com.nexa.ai.ai.transformer;

public class Softmax {

    public double[][] forward(
            double[][] input
    ) {

        validateInput(input);

        double[][] output =
                new double[input.length][];

        for (int i = 0; i < input.length; i++) {

            output[i] =
                    softmaxRow(input[i]);
        }

        return output;
    }

    private double[] softmaxRow(
            double[] row
    ) {

        double max = row[0];

        for (int i = 1; i < row.length; i++) {

            max =
                    Math.max(
                            max,
                            row[i]
                    );
        }

        double[] result =
                new double[row.length];

        double sum = 0.0;

        for (int i = 0; i < row.length; i++) {

            result[i] =
                    Math.exp(
                            row[i] - max
                    );

            sum += result[i];
        }

        for (int i = 0; i < result.length; i++) {

            result[i] /= sum;
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

        int columns =
                input[0].length;

        if (columns == 0) {
            throw new IllegalArgumentException(
                    "Input rows cannot be empty"
            );
        }

        for (double[] row : input) {

            if (row == null ||
                    row.length != columns) {

                throw new IllegalArgumentException(
                        "Input must be rectangular"
                );
            }
        }
    }
}