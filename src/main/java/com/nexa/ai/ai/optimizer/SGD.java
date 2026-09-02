package com.nexa.ai.ai.optimizer;

import com.nexa.ai.ai.layers.DenseLayer;
import com.nexa.ai.ai.tensor.Tensor;

public class SGD implements Optimizer {

    private final double learningRate;

    public SGD(double learningRate) {

        if (learningRate <= 0) {
            throw new IllegalArgumentException(
                    "Learning rate must be greater than zero"
            );
        }

        this.learningRate = learningRate;
    }

    @Override
    public void update(DenseLayer layer) {

        Tensor weights =
                layer.getWeights();

        Tensor weightGradient =
                layer.getWeightGradient();

        Tensor bias =
                layer.getBias();

        Tensor biasGradient =
                layer.getBiasGradient();

        if (weightGradient == null
                || biasGradient == null) {

            throw new IllegalStateException(
                    "backward() must be called before update()"
            );
        }

        /*
         * Update weights.
         *
         * W = W - learningRate * dL/dW
         */
        for (int i = 0; i < weights.rows(); i++) {

            for (int j = 0; j < weights.columns(); j++) {

                double updatedValue =
                        weights.get(i, j)
                                - learningRate
                                * weightGradient.get(i, j);

                weights.set(
                        i,
                        j,
                        updatedValue
                );
            }
        }

        /*
         * Update biases.
         *
         * B = B - learningRate * dL/dB
         */
        for (int j = 0; j < bias.columns(); j++) {

            double updatedValue =
                    bias.get(0, j)
                            - learningRate
                            * biasGradient.get(0, j);

            bias.set(
                    0,
                    j,
                    updatedValue
            );
        }
    }

    public double getLearningRate() {
        return learningRate;
    }
}