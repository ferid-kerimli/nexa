package com.nexa.ai.ai.layers;

import com.nexa.ai.ai.tensor.Tensor;

import java.util.Random;

public class DenseLayer implements Layer {

    private final int inputSize;
    private final int outputSize;

    private final Tensor weights;
    private final Tensor bias;

    private Tensor lastInput;

    private Tensor weightGradient;
    private Tensor biasGradient;

    public DenseLayer(
            int inputSize,
            int outputSize
    ) {

        if (inputSize <= 0 || outputSize <= 0) {
            throw new IllegalArgumentException(
                    "Layer dimensions must be greater than zero"
            );
        }

        this.inputSize = inputSize;
        this.outputSize = outputSize;

        this.weights =
                initializeWeights(
                        inputSize,
                        outputSize
                );

        this.bias =
                new Tensor(
                        1,
                        outputSize
                );
    }

    @Override
    public Tensor forward(Tensor input) {

        if (input.columns() != inputSize) {
            throw new IllegalArgumentException(
                    "Expected input with "
                            + inputSize
                            + " features but received "
                            + input.columns()
            );
        }

        /*
         * Save input because backward()
         * needs it to calculate dL/dW.
         */
        this.lastInput = input;

        return input
                .matrixMultiply(weights)
                .add(bias);
    }

    @Override
    public Tensor backward(Tensor gradient) {

        if (lastInput == null) {
            throw new IllegalStateException(
                    "forward() must be called before backward()"
            );
        }

        /*
         * dL/dW = X^T × dL/dY
         */
        weightGradient =
                lastInput
                        .transpose()
                        .matrixMultiply(gradient)
                        .divide(lastInput.rows());

        /*
         * Calculate bias gradient.
         */
        biasGradient =
                new Tensor(
                        1,
                        outputSize
                );

        for (int j = 0; j < outputSize; j++) {

            double sum = 0.0;

            for (int i = 0; i < gradient.rows(); i++) {

                sum += gradient.get(i, j);
            }

            biasGradient.set(
                    0,
                    j,
                    sum / gradient.rows()
            );
        }

        /*
         * dL/dX = dL/dY × W^T
         */
        return gradient
                .matrixMultiply(weights.transpose());
    }

    private Tensor initializeWeights(
            int rows,
            int columns
    ) {

        Tensor tensor =
                new Tensor(rows, columns);

        Random random =
                new Random();

        double limit =
                Math.sqrt(
                        6.0 / (rows + columns)
                );

        for (int i = 0; i < rows; i++) {

            for (int j = 0; j < columns; j++) {

                double value =
                        -limit
                                + random.nextDouble()
                                * (2 * limit);

                tensor.set(
                        i,
                        j,
                        value
                );
            }
        }

        return tensor;
    }

    public int getInputSize() {
        return inputSize;
    }

    public int getOutputSize() {
        return outputSize;
    }

    public Tensor getWeights() {
        return weights;
    }

    public Tensor getBias() {
        return bias;
    }

    public Tensor getWeightGradient() {
        return weightGradient;
    }

    public Tensor getBiasGradient() {
        return biasGradient;
    }
}