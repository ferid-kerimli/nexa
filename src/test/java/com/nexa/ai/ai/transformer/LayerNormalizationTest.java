package com.nexa.ai.ai.transformer;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class LayerNormalizationTest {

    @Test
    void shouldCreateLayerNormalization() {

        LayerNormalization normalization =
                new LayerNormalization(4);

        assertEquals(
                4,
                normalization.getEmbeddingSize()
        );
    }

    @Test
    void shouldPreserveInputShape() {

        LayerNormalization normalization =
                new LayerNormalization(4);

        double[][] input = {
                {1.0, 2.0, 3.0, 4.0},
                {5.0, 6.0, 7.0, 8.0},
                {2.0, 4.0, 6.0, 8.0}
        };

        double[][] output =
                normalization.forward(input);

        assertEquals(3, output.length);

        for (double[] row : output) {
            assertEquals(4, row.length);
        }
    }

    @Test
    void shouldNormalizeRowsToApproximatelyZeroMean() {

        LayerNormalization normalization =
                new LayerNormalization(4);

        double[][] input = {
                {1.0, 2.0, 3.0, 4.0},
                {10.0, 20.0, 30.0, 40.0}
        };

        double[][] output =
                normalization.forward(input);

        for (double[] row : output) {

            double sum = 0.0;

            for (double value : row) {
                sum += value;
            }

            double mean =
                    sum / row.length;

            assertEquals(
                    0.0,
                    mean,
                    1e-10
            );
        }
    }

    @Test
    void shouldNormalizeRowsToApproximatelyUnitVariance() {

        LayerNormalization normalization =
                new LayerNormalization(4);

        double[][] input = {
                {1.0, 2.0, 3.0, 4.0},
                {10.0, 20.0, 30.0, 40.0}
        };

        double[][] output =
                normalization.forward(input);

        for (double[] row : output) {

            double sum = 0.0;

            for (double value : row) {
                sum += value;
            }

            double mean =
                    sum / row.length;

            double varianceSum = 0.0;

            for (double value : row) {

                double difference =
                        value - mean;

                varianceSum +=
                        difference * difference;
            }

            double variance =
                    varianceSum / row.length;

            assertEquals(
                    1.0,
                    variance,
                    1e-4
            );
        }
    }

    @Test
    void shouldProduceFiniteValues() {

        LayerNormalization normalization =
                new LayerNormalization(4);

        double[][] input = {
                {1.0, 2.0, 3.0, 4.0},
                {100.0, 200.0, 300.0, 400.0}
        };

        double[][] output =
                normalization.forward(input);

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

        LayerNormalization normalization =
                new LayerNormalization(4);

        assertThrows(
                IllegalArgumentException.class,
                () -> normalization.forward(null)
        );
    }

    @Test
    void shouldRejectEmptyInput() {

        LayerNormalization normalization =
                new LayerNormalization(4);

        assertThrows(
                IllegalArgumentException.class,
                () -> normalization.forward(
                        new double[0][0]
                )
        );
    }

    @Test
    void shouldRejectWrongEmbeddingSize() {

        LayerNormalization normalization =
                new LayerNormalization(4);

        double[][] input = {
                {1.0, 2.0, 3.0}
        };

        assertThrows(
                IllegalArgumentException.class,
                () -> normalization.forward(input)
        );
    }

    @Test
    void shouldRejectInvalidEmbeddingSize() {

        assertThrows(
                IllegalArgumentException.class,
                () -> new LayerNormalization(0)
        );
    }
}