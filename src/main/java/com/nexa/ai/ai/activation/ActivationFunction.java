package com.nexa.ai.ai.activation;

import com.nexa.ai.ai.tensor.Tensor;

public interface ActivationFunction {

    Tensor forward(Tensor input);

    Tensor backward(Tensor input, Tensor gradient);
}