package com.nexa.ai.ai.transformer;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class EmbeddingTest {

    @Test
    void shouldCreateEmbedding() {

        Embedding embedding =
                new Embedding(100, 16);

        assertEquals(
                100,
                embedding.getVocabularySize()
        );

        assertEquals(
                16,
                embedding.getEmbeddingSize()
        );
    }

    @Test
    void shouldReturnVectorForToken() {

        Embedding embedding =
                new Embedding(100, 16);

        double[] vector =
                embedding.forward(5);

        assertEquals(
                16,
                vector.length
        );
    }

    @Test
    void shouldReturnVectorsForMultipleTokens() {

        Embedding embedding =
                new Embedding(100, 16);

        int[] tokens = {
                2, 5, 8
        };

        double[][] result =
                embedding.forward(tokens);

        assertEquals(
                3,
                result.length
        );

        assertEquals(
                16,
                result[0].length
        );

        assertEquals(
                16,
                result[1].length
        );

        assertEquals(
                16,
                result[2].length
        );
    }

    @Test
    void shouldRejectInvalidTokenId() {

        Embedding embedding =
                new Embedding(100, 16);

        assertThrows(
                IllegalArgumentException.class,
                () -> embedding.forward(100)
        );
    }

    @Test
    void shouldRejectNegativeTokenId() {

        Embedding embedding =
                new Embedding(100, 16);

        assertThrows(
                IllegalArgumentException.class,
                () -> embedding.forward(-1)
        );
    }

    @Test
    void shouldReturnIndependentVectorCopy() {

        Embedding embedding =
                new Embedding(100, 16);

        double[] vector =
                embedding.forward(5);

        double originalValue = vector[0];

        vector[0] = 999999;

        double[] secondVector =
                embedding.forward(5);

        assertEquals(
                originalValue,
                secondVector[0]
        );
    }
}