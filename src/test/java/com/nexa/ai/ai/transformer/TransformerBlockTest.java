package com.nexa.ai.ai.transformer;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class TransformerBlockTest {

    @Test
    void shouldCreateTransformerBlock() {

        TransformerBlock block =
                new TransformerBlock(4, 16);

        assertEquals(
                4,
                block.getEmbeddingSize()
        );

        assertEquals(
                16,
                block.getFeedForwardSize()
        );
    }

    @Test
    void shouldPreserveSequenceShape() {

        TransformerBlock block =
                new TransformerBlock(4, 16);

        double[][] input = {
                {1.0, 0.5, 0.2, 0.1},
                {0.3, 0.8, 0.4, 0.2},
                {0.7, 0.1, 0.9, 0.5}
        };

        double[][] output =
                block.forward(input);

        assertEquals(
                3,
                output.length
        );

        for (double[] row : output) {
            assertEquals(
                    4,
                    row.length
            );
        }
    }

    @Test
    void shouldProduceFiniteValues() {

        TransformerBlock block =
                new TransformerBlock(4, 16);

        double[][] input = {
                {1.0, 0.5, 0.2, 0.1},
                {0.3, 0.8, 0.4, 0.2},
                {0.7, 0.1, 0.9, 0.5}
        };

        double[][] output =
                block.forward(input);

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
    void shouldUseCausalAttentionByDefault() {

        TransformerBlock block =
                new TransformerBlock(4, 16);

        double[][] input = {
                {1.0, 0.5, 0.2, 0.1},
                {0.3, 0.8, 0.4, 0.2},
                {0.7, 0.1, 0.9, 0.5}
        };

        block.forward(input);

        double[][] weights =
                block.getSelfAttention()
                        .getLastAttentionWeights();

        assertEquals(
                0.0,
                weights[0][1],
                1e-12
        );

        assertEquals(
                0.0,
                weights[0][2],
                1e-12
        );

        assertEquals(
                0.0,
                weights[1][2],
                1e-12
        );
    }

    @Test
    void shouldSupportNonCausalAttention() {

        TransformerBlock block =
                new TransformerBlock(4, 16);

        double[][] input = {
                {1.0, 0.5, 0.2, 0.1},
                {0.3, 0.8, 0.4, 0.2},
                {0.7, 0.1, 0.9, 0.5}
        };

        block.forward(input, false);

        double[][] weights =
                block.getSelfAttention()
                        .getLastAttentionWeights();

        for (int i = 0; i < weights.length; i++) {

            double sum = 0.0;

            for (int j = 0; j < weights[i].length; j++) {
                sum += weights[i][j];
            }

            assertEquals(
                    1.0,
                    sum,
                    1e-10
            );
        }
    }

    @Test
    void shouldRejectNullInput() {

        TransformerBlock block =
                new TransformerBlock(4, 16);

        assertThrows(
                IllegalArgumentException.class,
                () -> block.forward(null)
        );
    }

    @Test
    void shouldRejectEmptyInput() {

        TransformerBlock block =
                new TransformerBlock(4, 16);

        assertThrows(
                IllegalArgumentException.class,
                () -> block.forward(
                        new double[0][0]
                )
        );
    }

    @Test
    void shouldRejectWrongEmbeddingSize() {

        TransformerBlock block =
                new TransformerBlock(4, 16);

        double[][] input = {
                {1.0, 2.0, 3.0}
        };

        assertThrows(
                IllegalArgumentException.class,
                () -> block.forward(input)
        );
    }

    @Test
    void shouldRejectInvalidEmbeddingSize() {

        assertThrows(
                IllegalArgumentException.class,
                () -> new TransformerBlock(0, 16)
        );
    }

    @Test
    void shouldRejectInvalidFeedForwardSize() {

        assertThrows(
                IllegalArgumentException.class,
                () -> new TransformerBlock(4, 0)
        );
    }
}