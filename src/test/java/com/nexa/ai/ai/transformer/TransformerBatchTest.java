package com.nexa.ai.ai.transformer;

import com.nexa.ai.ai.tensor.Tensor3D;
import com.nexa.ai.dataset.LanguageExample;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

class TransformerBatchTest {

    @Test
    void shouldCreateTransformerBatch() {

        Embedding embedding =
                new Embedding(10, 4);

        PositionalEncoding positionalEncoding =
                new PositionalEncoding(8, 4);

        List<LanguageExample> examples =
                List.of(
                        new LanguageExample(
                                List.of(2, 3, 4),
                                List.of(3, 4, 5)
                        ),
                        new LanguageExample(
                                List.of(5, 6, 7),
                                List.of(6, 7, 8)
                        )
                );

        TransformerBatch batch =
                TransformerBatch.fromExamples(
                        examples,
                        embedding,
                        positionalEncoding
                );

        assertEquals(
                2,
                batch.batchSize()
        );

        assertEquals(
                3,
                batch.sequenceLength()
        );

        assertEquals(
                4,
                batch.embeddingSize()
        );
    }

    @Test
    void shouldCreateCorrectTensorShape() {

        Embedding embedding =
                new Embedding(10, 4);

        PositionalEncoding positionalEncoding =
                new PositionalEncoding(8, 4);

        List<LanguageExample> examples =
                List.of(
                        new LanguageExample(
                                List.of(2, 3, 4),
                                List.of(3, 4, 5)
                        ),
                        new LanguageExample(
                                List.of(5, 6, 7),
                                List.of(6, 7, 8)
                        )
                );

        TransformerBatch batch =
                TransformerBatch.fromExamples(
                        examples,
                        embedding,
                        positionalEncoding
                );

        Tensor3D tensor =
                batch.inputs();

        assertEquals(
                2,
                tensor.depth()
        );

        assertEquals(
                3,
                tensor.rows()
        );

        assertEquals(
                4,
                tensor.columns()
        );
    }

    @Test
    void shouldPreserveTargets() {

        Embedding embedding =
                new Embedding(10, 4);

        PositionalEncoding positionalEncoding =
                new PositionalEncoding(8, 4);

        List<LanguageExample> examples =
                List.of(
                        new LanguageExample(
                                List.of(2, 3, 4),
                                List.of(3, 4, 5)
                        ),
                        new LanguageExample(
                                List.of(5, 6, 7),
                                List.of(6, 7, 8)
                        )
                );

        TransformerBatch batch =
                TransformerBatch.fromExamples(
                        examples,
                        embedding,
                        positionalEncoding
                );

        int[][] targets =
                batch.targets();

        assertArrayEquals(
                new int[]{3, 4, 5},
                targets[0]
        );

        assertArrayEquals(
                new int[]{6, 7, 8},
                targets[1]
        );
    }

    @Test
    void shouldApplyPositionalEncoding() {

        Embedding embedding =
                new Embedding(10, 4);

        PositionalEncoding positionalEncoding =
                new PositionalEncoding(8, 4);

        List<LanguageExample> examples =
                List.of(
                        new LanguageExample(
                                List.of(2, 3),
                                List.of(3, 4)
                        )
                );

        TransformerBatch batch =
                TransformerBatch.fromExamples(
                        examples,
                        embedding,
                        positionalEncoding
                );

        double[][] expected =
                positionalEncoding.add(
                        embedding.forward(
                                new int[]{2, 3}
                        )
                );

        Tensor3D inputs =
                batch.inputs();

        for (int sequence = 0;
             sequence < 2;
             sequence++) {

            for (int dimension = 0;
                 dimension < 4;
                 dimension++) {

                assertEquals(
                        expected[sequence][dimension],
                        inputs.get(
                                0,
                                sequence,
                                dimension
                        ),
                        1e-12
                );
            }
        }
    }

    @Test
    void shouldRejectDifferentSequenceLengths() {

        Embedding embedding =
                new Embedding(10, 4);

        PositionalEncoding positionalEncoding =
                new PositionalEncoding(8, 4);

        List<LanguageExample> examples =
                List.of(
                        new LanguageExample(
                                List.of(2, 3, 4),
                                List.of(3, 4, 5)
                        ),
                        new LanguageExample(
                                List.of(5, 6),
                                List.of(6, 7)
                        )
                );

        assertThrows(
                IllegalArgumentException.class,
                () -> TransformerBatch.fromExamples(
                        examples,
                        embedding,
                        positionalEncoding
                )
        );
    }

    @Test
    void shouldRejectSequenceLongerThanPositionalEncoding() {

        Embedding embedding =
                new Embedding(10, 4);

        PositionalEncoding positionalEncoding =
                new PositionalEncoding(2, 4);

        List<LanguageExample> examples =
                List.of(
                        new LanguageExample(
                                List.of(2, 3, 4),
                                List.of(3, 4, 5)
                        )
                );

        assertThrows(
                IllegalArgumentException.class,
                () -> TransformerBatch.fromExamples(
                        examples,
                        embedding,
                        positionalEncoding
                )
        );
    }

    @Test
    void shouldRejectNullExamples() {

        Embedding embedding =
                new Embedding(10, 4);

        PositionalEncoding positionalEncoding =
                new PositionalEncoding(8, 4);

        assertThrows(
                IllegalArgumentException.class,
                () -> TransformerBatch.fromExamples(
                        null,
                        embedding,
                        positionalEncoding
                )
        );
    }

    @Test
    void shouldReturnIndependentTargetCopy() {

        Embedding embedding =
                new Embedding(10, 4);

        PositionalEncoding positionalEncoding =
                new PositionalEncoding(8, 4);

        List<LanguageExample> examples =
                List.of(
                        new LanguageExample(
                                List.of(2, 3),
                                List.of(3, 4)
                        )
                );

        TransformerBatch batch =
                TransformerBatch.fromExamples(
                        examples,
                        embedding,
                        positionalEncoding
                );

        int[][] targets =
                batch.targets();

        targets[0][0] = 999;

        assertEquals(
                3,
                batch.targets()[0][0]
        );
    }
}