package com.nexa.ai.ai.transformer;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class OutputProjectionTest {

    @Test
    void shouldCreateProjection() {

        OutputProjection projection =
                new OutputProjection(4, 10);

        assertEquals(
                4,
                projection.getEmbeddingSize()
        );

        assertEquals(
                10,
                projection.getVocabularySize()
        );
    }

    @Test
    void shouldConvertEmbeddingToVocabularySize() {

        OutputProjection projection =
                new OutputProjection(4, 10);

        double[][] input = {
                {1.0, 0.5, 0.2, 0.1},
                {0.3, 0.8, 0.4, 0.2},
                {0.7, 0.1, 0.9, 0.5}
        };

        double[][] output =
                projection.forward(input);

        assertEquals(3, output.length);

        for (double[] row : output) {

            assertEquals(
                    10,
                    row.length
            );
        }
    }

    @Test
    void shouldProduceFiniteLogits() {

        OutputProjection projection =
                new OutputProjection(4, 10);

        double[][] input = {
                {1.0, 2.0, 3.0, 4.0}
        };

        double[][] output =
                projection.forward(input);

        for (double value : output[0]) {

            assertTrue(
                    Double.isFinite(value)
            );
        }
    }

    @Test
    void shouldRejectNullInput() {

        OutputProjection projection =
                new OutputProjection(4, 10);

        assertThrows(
                IllegalArgumentException.class,
                () -> projection.forward(null)
        );
    }

    @Test
    void shouldRejectWrongEmbeddingSize() {

        OutputProjection projection =
                new OutputProjection(4, 10);

        double[][] input = {
                {1.0, 2.0, 3.0}
        };

        assertThrows(
                IllegalArgumentException.class,
                () -> projection.forward(input)
        );
    }

    @Test
    void shouldRejectInvalidEmbeddingSize() {

        assertThrows(
                IllegalArgumentException.class,
                () -> new OutputProjection(0, 10)
        );
    }

    @Test
    void shouldRejectInvalidVocabularySize() {

        assertThrows(
                IllegalArgumentException.class,
                () -> new OutputProjection(4, 0)
        );
    }
}