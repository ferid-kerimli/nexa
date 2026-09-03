package com.nexa.ai.ai.transformer;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class PositionalEncodingTest {

    @Test
    void shouldCreateEncoding() {

        PositionalEncoding encoding =
                new PositionalEncoding(128, 64);

        assertEquals(
                128,
                encoding.getMaxSequenceLength()
        );

        assertEquals(
                64,
                encoding.getEmbeddingSize()
        );
    }

    @Test
    void shouldCreateCorrectEncodingShape() {

        PositionalEncoding encoding =
                new PositionalEncoding(10, 8);

        double[][] result =
                encoding.getEncoding();

        assertEquals(10, result.length);
        assertEquals(8, result[0].length);
    }

    @Test
    void firstPositionShouldHaveExpectedValues() {

        PositionalEncoding encoding =
                new PositionalEncoding(10, 8);

        double[][] result =
                encoding.getEncoding();

        for (int dimension = 0;
             dimension < 8;
             dimension++) {

            if (dimension % 2 == 0) {
                assertEquals(
                        0.0,
                        result[0][dimension],
                        1e-10
                );
            } else {
                assertEquals(
                        1.0,
                        result[0][dimension],
                        1e-10
                );
            }
        }
    }

    @Test
    void shouldAddPositionInformationToEmbeddings() {

        PositionalEncoding encoding =
                new PositionalEncoding(10, 4);

        double[][] embeddings = {
                {1.0, 2.0, 3.0, 4.0},
                {5.0, 6.0, 7.0, 8.0}
        };

        double[][] result =
                encoding.add(embeddings);

        assertEquals(
                1.0,
                result[0][0],
                1e-10
        );

        assertEquals(
                3.0,
                result[0][1],
                1e-10
        );

        assertEquals(
                3.0,
                result[0][2],
                1e-10
        );

        assertEquals(
                5.0,
                result[0][3],
                1e-10
        );
    }

    @Test
    void shouldRejectSequenceThatIsTooLong() {

        PositionalEncoding encoding =
                new PositionalEncoding(3, 4);

        double[][] embeddings = {
                {1, 2, 3, 4},
                {1, 2, 3, 4},
                {1, 2, 3, 4},
                {1, 2, 3, 4}
        };

        assertThrows(
                IllegalArgumentException.class,
                () -> encoding.add(embeddings)
        );
    }

    @Test
    void shouldRejectWrongEmbeddingDimension() {

        PositionalEncoding encoding =
                new PositionalEncoding(10, 4);

        double[][] embeddings = {
                {1, 2, 3}
        };

        assertThrows(
                IllegalArgumentException.class,
                () -> encoding.add(embeddings)
        );
    }
}