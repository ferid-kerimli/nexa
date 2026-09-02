package com.nexa.ai.ai.loss;

import com.nexa.ai.ai.tensor.Tensor;

public interface LossFunction {

    double calculate(
            Tensor prediction,
            Tensor target
    );

    Tensor gradient(
            Tensor prediction,
            Tensor target
    );
}