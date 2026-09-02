package com.nexa.ai.ai.activation;

import com.nexa.ai.ai.tensor.Tensor;

public class ReLU implements ActivationFunction {

    @Override
    public Tensor forward(Tensor input) {

        Tensor result =
                new Tensor(input.rows(), input.columns());

        for (int i = 0; i < input.rows(); i++) {

            for (int j = 0; j < input.columns(); j++) {

                double value = input.get(i, j);

                result.set(
                        i,
                        j,
                        Math.max(0.0, value)
                );
            }
        }

        return result;
    }

    @Override
    public Tensor backward(
            Tensor input,
            Tensor gradient
    ) {

        if (input.rows() != gradient.rows()
                || input.columns() != gradient.columns()) {

            throw new IllegalArgumentException(
                    "Input and gradient shapes must match"
            );
        }

        Tensor result =
                new Tensor(
                        input.rows(),
                        input.columns()
                );

        for (int i = 0; i < input.rows(); i++) {

            for (int j = 0; j < input.columns(); j++) {

                double derivative =
                        input.get(i, j) > 0
                                ? 1.0
                                : 0.0;

                result.set(
                        i,
                        j,
                        gradient.get(i, j) * derivative
                );
            }
        }

        return result;
    }
}