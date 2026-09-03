package com.nexa.ai.dataset;

import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

class LanguageDatasetBuilderTest {

    @Test
    void shouldCreateSlidingWindowExamples() {

        LanguageDatasetBuilder builder =
                new LanguageDatasetBuilder();

        LanguageDataset dataset =
                builder.build(
                        List.of(1, 2, 3, 4, 5),
                        3
                );

        assertEquals(2, dataset.size());

        assertEquals(
                List.of(1, 2, 3),
                dataset.get(0).inputTokens()
        );

        assertEquals(
                List.of(2, 3, 4),
                dataset.get(0).targetTokens()
        );

        assertEquals(
                List.of(2, 3, 4),
                dataset.get(1).inputTokens()
        );

        assertEquals(
                List.of(3, 4, 5),
                dataset.get(1).targetTokens()
        );
    }

    @Test
    void shouldRejectInvalidContextSize() {

        LanguageDatasetBuilder builder =
                new LanguageDatasetBuilder();

        assertThrows(
                IllegalArgumentException.class,
                () -> builder.build(
                        List.of(1, 2, 3),
                        0
                )
        );
    }

    @Test
    void shouldRejectTooShortSequence() {

        LanguageDatasetBuilder builder =
                new LanguageDatasetBuilder();

        assertThrows(
                IllegalArgumentException.class,
                () -> builder.build(
                        List.of(1, 2, 3),
                        3
                )
        );
    }

    @Test
    void shouldCreateCorrectNumberOfExamples() {

        LanguageDatasetBuilder builder =
                new LanguageDatasetBuilder();

        LanguageDataset dataset =
                builder.build(
                        List.of(1, 2, 3, 4, 5, 6, 7),
                        2
                );

        assertEquals(5, dataset.size());
    }
}