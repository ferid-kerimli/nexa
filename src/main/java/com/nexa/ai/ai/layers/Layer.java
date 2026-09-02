package com.nexa.ai.ai.layers;

import com.nexa.ai.ai.tensor.Tensor;

public interface Layer {

    Tensor forward(Tensor input);

    Tensor backward(Tensor gradient);
}