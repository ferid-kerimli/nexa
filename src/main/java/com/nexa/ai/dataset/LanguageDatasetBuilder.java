package com.nexa.ai.dataset;

import java.util.ArrayList;
import java.util.List;

public class LanguageDatasetBuilder {

    public LanguageDataset build(
            List<Integer> tokenIds,
            int contextSize
    ) {

        if (tokenIds == null || tokenIds.isEmpty()) {
            throw new IllegalArgumentException(
                    "Token IDs cannot be null or empty"
            );
        }

        if (contextSize <= 0) {
            throw new IllegalArgumentException(
                    "Context size must be greater than zero"
            );
        }

        if (tokenIds.size() <= contextSize) {
            throw new IllegalArgumentException(
                    "Token sequence must contain more tokens than context size"
            );
        }

        List<LanguageExample> examples = new ArrayList<>();

        for (int i = 0;
             i + contextSize < tokenIds.size();
             i++) {

            List<Integer> input =
                    new ArrayList<>(
                            tokenIds.subList(
                                    i,
                                    i + contextSize
                            )
                    );

            List<Integer> target =
                    new ArrayList<>(
                            tokenIds.subList(
                                    i + 1,
                                    i + contextSize + 1
                            )
                    );

            examples.add(
                    new LanguageExample(input, target)
            );
        }

        return new LanguageDataset(examples);
    }
}