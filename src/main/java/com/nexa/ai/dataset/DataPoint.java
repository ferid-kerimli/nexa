package com.nexa.ai.dataset;

import com.nexa.ai.ai.tensor.Tensor;

public record DataPoint(
        Tensor input,
        Tensor target
) {

    public DataPoint {

        if (input == null) {
            throw new IllegalArgumentException(
                    "Input cannot be null"
            );
        }

        if (target == null) {
            throw new IllegalArgumentException(
                    "Target cannot be null"
            );
        }
    }
}