package com.nexa.ai.dataset;

import com.nexa.ai.ai.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;

public class Batcher {

    public List<Batch> createBatches(
            Dataset dataset,
            int batchSize
    ) {

        if (dataset == null) {
            throw new IllegalArgumentException(
                    "Dataset cannot be null"
            );
        }

        if (batchSize <= 0) {
            throw new IllegalArgumentException(
                    "Batch size must be greater than zero"
            );
        }

        List<Batch> batches =
                new ArrayList<>();

        List<DataPoint> points =
                dataset.getDataPoints();

        for (
                int start = 0;
                start < points.size();
                start += batchSize
        ) {

            int end =
                    Math.min(
                            start + batchSize,
                            points.size()
                    );

            List<DataPoint> currentBatch =
                    points.subList(start, end);

            batches.add(
                    createBatch(currentBatch)
            );
        }

        return batches;
    }

    private Batch createBatch(
            List<DataPoint> points
    ) {

        if (points.isEmpty()) {
            throw new IllegalArgumentException(
                    "Batch cannot be empty"
            );
        }

        int batchSize =
                points.size();

        int inputColumns =
                points.get(0)
                        .input()
                        .columns();

        int targetColumns =
                points.get(0)
                        .target()
                        .columns();

        Tensor input =
                new Tensor(
                        batchSize,
                        inputColumns
                );

        Tensor target =
                new Tensor(
                        batchSize,
                        targetColumns
                );

        for (int i = 0; i < batchSize; i++) {

            DataPoint point =
                    points.get(i);

            for (int j = 0;
                 j < inputColumns;
                 j++) {

                input.set(
                        i,
                        j,
                        point.input().get(0, j)
                );
            }

            for (int j = 0;
                 j < targetColumns;
                 j++) {

                target.set(
                        i,
                        j,
                        point.target().get(0, j)
                );
            }
        }

        return new Batch(
                input,
                target
        );
    }
}