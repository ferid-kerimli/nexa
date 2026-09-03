package com.nexa.ai.dataset;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class LanguageDataset {

    private final List<LanguageExample> examples;

    public LanguageDataset(List<LanguageExample> examples) {

        if (examples == null || examples.isEmpty()) {
            throw new IllegalArgumentException(
                    "Examples cannot be null or empty"
            );
        }

        this.examples =
                Collections.unmodifiableList(
                        new ArrayList<>(examples)
                );
    }

    public int size() {
        return examples.size();
    }

    public LanguageExample get(int index) {
        if (index < 0 || index >= examples.size()) {
            throw new IndexOutOfBoundsException(
                    "Invalid example index: " + index
            );
        }

        return examples.get(index);
    }

    public List<LanguageExample> getExamples() {
        return examples;
    }
}