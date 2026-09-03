package com.nexa.ai.dataset;

import java.util.List;

public record LanguageExample(
        List<Integer> inputTokens,
        List<Integer> targetTokens
) {

    public LanguageExample {

        if (inputTokens == null) {
            throw new IllegalArgumentException(
                    "Input tokens cannot be null"
            );
        }

        if (targetTokens == null) {
            throw new IllegalArgumentException(
                    "Target tokens cannot be null"
            );
        }

        if (inputTokens.isEmpty()) {
            throw new IllegalArgumentException(
                    "Input tokens cannot be empty"
            );
        }

        if (inputTokens.size() != targetTokens.size()) {
            throw new IllegalArgumentException(
                    "Input and target must have the same length"
            );
        }
    }
}