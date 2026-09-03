package com.nexa.ai.ai.transformer;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class SelfAttentionTest {

    @Test
    void shouldCreateSelfAttention() {

        SelfAttention attention =
                new SelfAttention(4);

        assertEquals(4, attention.getEmbeddingSize());
    }

    @Test
    void shouldPreserveInputShape() {

        SelfAttention attention =
                new SelfAttention(4);

        double[][] input = {
                {1.0, 0.5, 0.2, 0.1},
                {0.3, 0.8, 0.4, 0.2},
                {0.7, 0.1, 0.9, 0.5}
        };

        double[][] output =
                attention.forward(input);

        assertEquals(3, output.length);
        assertEquals(4, output[0].length);
        assertEquals(4, output[1].length);
        assertEquals(4, output[2].length);
    }

    @Test
    void shouldProduceFiniteValues() {

        SelfAttention attention =
                new SelfAttention(4);

        double[][] input = {
                {1.0, 0.5, 0.2, 0.1},
                {0.3, 0.8, 0.4, 0.2},
                {0.7, 0.1, 0.9, 0.5}
        };

        double[][] output =
                attention.forward(input);

        for (double[] row : output) {
            for (double value : row) {
                assertTrue(
                        Double.isFinite(value),
                        "Output contains non-finite value"
                );
            }
        }
    }

    @Test
    void shouldRejectNullInput() {

        SelfAttention attention =
                new SelfAttention(4);

        assertThrows(
                IllegalArgumentException.class,
                () -> attention.forward(null)
        );
    }

    @Test
    void shouldRejectEmptyInput() {

        SelfAttention attention =
                new SelfAttention(4);

        assertThrows(
                IllegalArgumentException.class,
                () -> attention.forward(new double[0][0])
        );
    }

    @Test
    void shouldRejectWrongEmbeddingSize() {

        SelfAttention attention =
                new SelfAttention(4);

        double[][] input = {
                {1.0, 2.0, 3.0}
        };

        assertThrows(
                IllegalArgumentException.class,
                () -> attention.forward(input)
        );
    }

    @Test
    void shouldApplyCausalMask() {

        SelfAttention attention =
                new SelfAttention(4);

        double[][] input = {
                {1.0, 0.5, 0.2, 0.1},
                {0.3, 0.8, 0.4, 0.2},
                {0.7, 0.1, 0.9, 0.5}
        };

        attention.forward(input, true);

        double[][] weights =
                attention.getLastAttentionWeights();

        // Token 1 cannot attend to token 2 or 3
        assertEquals(0.0, weights[0][1], 1e-12);
        assertEquals(0.0, weights[0][2], 1e-12);

        // Token 2 cannot attend to token 3
        assertEquals(0.0, weights[1][2], 1e-12);

        // Token 3 has access to all previous tokens
        assertTrue(weights[2][0] > 0.0);
        assertTrue(weights[2][1] > 0.0);
        assertTrue(weights[2][2] > 0.0);
    }

    @Test
    void attentionWeightsShouldSumToOne() {

        SelfAttention attention =
                new SelfAttention(4);

        double[][] input = {
                {1.0, 0.5, 0.2, 0.1},
                {0.3, 0.8, 0.4, 0.2},
                {0.7, 0.1, 0.9, 0.5}
        };

        attention.forward(input, true);

        double[][] weights =
                attention.getLastAttentionWeights();

        for (double[] row : weights) {

            double sum = 0.0;

            for (double value : row) {
                sum += value;
            }

            assertEquals(1.0, sum, 1e-10);
        }
    }

    @Test
    void shouldRejectAccessingAttentionWeightsBeforeForward() {

        SelfAttention attention =
                new SelfAttention(4);

        assertThrows(
                IllegalStateException.class,
                attention::getLastAttentionWeights
        );
    }
}