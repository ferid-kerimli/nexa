package com.nexa.ai.ai.layers;

import com.nexa.ai.ai.activation.ActivationFunction;
import com.nexa.ai.ai.tensor.Tensor;

public class ActivationLayer implements Layer {

    private final ActivationFunction activationFunction;

    private Tensor lastInput;

    public ActivationLayer(
            ActivationFunction activationFunction
    ) {

        if (activationFunction == null) {
            throw new IllegalArgumentException(
                    "Activation function cannot be null"
            );
        }

        this.activationFunction =
                activationFunction;
    }

    @Override
    public Tensor forward(Tensor input) {

        this.lastInput = input;

        return activationFunction.forward(input);
    }

    @Override
    public Tensor backward(Tensor gradient) {

        if (lastInput == null) {
            throw new IllegalStateException(
                    "forward() must be called before backward()"
            );
        }

        return activationFunction.backward(
                lastInput,
                gradient
        );
    }
}