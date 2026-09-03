package com.nexa.ai.ai.transformer;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class FeedForwardTest {

    @Test
    void shouldCreateFeedForward() {

        FeedForward feedForward =
                new FeedForward(4, 16);

        assertEquals(
                4,
                feedForward.getEmbeddingSize()
        );

        assertEquals(
                16,
                feedForward.getHiddenSize()
        );
    }

    @Test
    void shouldPreserveSequenceShape() {

        FeedForward feedForward =
                new FeedForward(4, 16);

        double[][] input = {
                {1.0, 0.5, 0.2, 0.1},
                {0.3, 0.8, 0.4, 0.2},
                {0.7, 0.1, 0.9, 0.5}
        };

        double[][] output =
                feedForward.forward(input);

        assertEquals(3, output.length);

        for (double[] row : output) {
            assertEquals(4, row.length);
        }
    }

    @Test
    void shouldProduceFiniteValues() {

        FeedForward feedForward =
                new FeedForward(4, 16);

        double[][] input = {
                {1.0, 0.5, 0.2, 0.1},
                {0.3, 0.8, 0.4, 0.2}
        };

        double[][] output =
                feedForward.forward(input);

        for (double[] row : output) {

            for (double value : row) {

                assertTrue(
                        Double.isFinite(value)
                );
            }
        }
    }

    @Test
    void shouldRejectNullInput() {

        FeedForward feedForward =
                new FeedForward(4, 16);

        assertThrows(
                IllegalArgumentException.class,
                () -> feedForward.forward(null)
        );
    }

    @Test
    void shouldRejectEmptyInput() {

        FeedForward feedForward =
                new FeedForward(4, 16);

        assertThrows(
                IllegalArgumentException.class,
                () -> feedForward.forward(
                        new double[0][0]
                )
        );
    }

    @Test
    void shouldRejectWrongEmbeddingSize() {

        FeedForward feedForward =
                new FeedForward(4, 16);

        double[][] input = {
                {1.0, 2.0, 3.0}
        };

        assertThrows(
                IllegalArgumentException.class,
                () -> feedForward.forward(input)
        );
    }

    @Test
    void shouldRejectInvalidEmbeddingSize() {

        assertThrows(
                IllegalArgumentException.class,
                () -> new FeedForward(0, 16)
        );
    }

    @Test
    void shouldRejectInvalidHiddenSize() {

        assertThrows(
                IllegalArgumentException.class,
                () -> new FeedForward(4, 0)
        );
    }
}