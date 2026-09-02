package com.nexa.ai.dataset;

import com.nexa.ai.ai.tensor.Tensor;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;

class BatcherTest {

    @Test
    void shouldCreateBatches() {

        Dataset dataset =
                new Dataset(
                        List.of(
                                dataPoint(1, 2),
                                dataPoint(2, 4),
                                dataPoint(3, 6),
                                dataPoint(4, 8),
                                dataPoint(5, 10)
                        )
                );

        Batcher batcher =
                new Batcher();

        List<Batch> batches =
                batcher.createBatches(
                        dataset,
                        2
                );

        assertEquals(3, batches.size());

        assertEquals(2, batches.get(0).size());
        assertEquals(2, batches.get(1).size());
        assertEquals(1, batches.get(2).size());
    }

    @Test
    void shouldPreserveValues() {

        Dataset dataset =
                new Dataset(
                        List.of(
                                dataPoint(1, 2),
                                dataPoint(2, 4),
                                dataPoint(3, 6)
                        )
                );

        Batcher batcher =
                new Batcher();

        List<Batch> batches =
                batcher.createBatches(
                        dataset,
                        2
                );

        Batch first =
                batches.get(0);

        assertEquals(
                1,
                first.input().get(0, 0)
        );

        assertEquals(
                2,
                first.target().get(0, 0)
        );

        assertEquals(
                2,
                first.input().get(1, 0)
        );

        assertEquals(
                4,
                first.target().get(1, 0)
        );
    }

    private DataPoint dataPoint(
            double input,
            double target
    ) {

        return new DataPoint(
                new Tensor(
                        new double[][]{
                                {input}
                        }
                ),
                new Tensor(
                        new double[][]{
                                {target}
                        }
                )
        );
    }
}