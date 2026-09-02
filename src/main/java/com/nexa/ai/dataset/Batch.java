package com.nexa.ai.dataset;

import com.nexa.ai.ai.tensor.Tensor;

public record Batch(
        Tensor input,
        Tensor target
) {

    public Batch {
        if (input == null) {
            throw new IllegalArgumentException(
                    "Input cannot be null"
            );
        }

        if (target == null) {
            throw new IllegalArgumentException(
                    "Target cannot be null"
            );
        } // Added missing closing brace here

        if (input.rows() != target.rows()) {
            throw new IllegalArgumentException(
                    "Input and target must have the same batch size"
            );
        }
    }

    public int size() {
        return input.rows();
    }
}