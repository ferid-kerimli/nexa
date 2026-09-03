package com.nexa.ai.ai.transformer;

import com.nexa.ai.ai.tensor.Tensor3D;
import com.nexa.ai.dataset.LanguageExample;

import java.util.List;

public class TransformerBatch {

    private final Tensor3D inputs;
    private final int[][] targets;

    public TransformerBatch(
            Tensor3D inputs,
            int[][] targets
    ) {

        if (inputs == null) {
            throw new IllegalArgumentException(
                    "Inputs cannot be null"
            );
        }

        if (targets == null) {
            throw new IllegalArgumentException(
                    "Targets cannot be null"
            );
        }

        if (inputs.depth() != targets.length) {
            throw new IllegalArgumentException(
                    "Batch size does not match targets"
            );
        }

        for (int i = 0; i < targets.length; i++) {

            if (targets[i] == null ||
                    targets[i].length != inputs.rows()) {

                throw new IllegalArgumentException(
                        "Target sequence length does not match input"
                );
            }
        }

        this.inputs = inputs;

        this.targets =
                copyTargets(targets);
    }

    public static TransformerBatch fromExamples(
            List<LanguageExample> examples,
            Embedding embedding,
            PositionalEncoding positionalEncoding
    ) {

        if (examples == null ||
                examples.isEmpty()) {

            throw new IllegalArgumentException(
                    "Examples cannot be null or empty"
            );
        }

        if (embedding == null) {
            throw new IllegalArgumentException(
                    "Embedding cannot be null"
            );
        }

        if (positionalEncoding == null) {
            throw new IllegalArgumentException(
                    "Positional encoding cannot be null"
            );
        }

        int batchSize =
                examples.size();

        int sequenceLength =
                examples.get(0)
                        .inputTokens()
                        .size();

        int embeddingSize =
                embedding.getEmbeddingSize();

        if (sequenceLength <= 0) {
            throw new IllegalArgumentException(
                    "Sequence length must be greater than zero"
            );
        }

        if (sequenceLength >
                positionalEncoding.getMaxSequenceLength()) {

            throw new IllegalArgumentException(
                    "Sequence length exceeds positional encoding limit"
            );
        }

        Tensor3D inputs =
                new Tensor3D(
                        batchSize,
                        sequenceLength,
                        embeddingSize
                );

        int[][] targets =
                new int[batchSize][sequenceLength];

        for (int batch = 0;
             batch < batchSize;
             batch++) {

            LanguageExample example =
                    examples.get(batch);

            if (example == null) {
                throw new IllegalArgumentException(
                        "Example cannot be null"
                );
            }

            if (example.inputTokens().size()
                    != sequenceLength) {

                throw new IllegalArgumentException(
                        "All examples must have the same sequence length"
                );
            }

            if (example.targetTokens().size()
                    != sequenceLength) {

                throw new IllegalArgumentException(
                        "Target sequence length does not match input"
                );
            }

            int[] tokenIds =
                    new int[sequenceLength];

            for (int i = 0;
                 i < sequenceLength;
                 i++) {

                tokenIds[i] =
                        example.inputTokens()
                                .get(i);

                targets[batch][i] =
                        example.targetTokens()
                                .get(i);
            }

            double[][] embedded =
                    embedding.forward(tokenIds);

            double[][] positioned =
                    positionalEncoding.add(
                            embedded
                    );

            for (int sequence = 0;
                 sequence < sequenceLength;
                 sequence++) {

                for (int dimension = 0;
                     dimension < embeddingSize;
                     dimension++) {

                    inputs.set(
                            batch,
                            sequence,
                            dimension,
                            positioned[
                                    sequence
                                    ][dimension]
                    );
                }
            }
        }

        return new TransformerBatch(
                inputs,
                targets
        );
    }

    public Tensor3D inputs() {
        return inputs;
    }

    public int[][] targets() {
        return copyTargets(targets);
    }

    public int batchSize() {
        return inputs.depth();
    }

    public int sequenceLength() {
        return inputs.rows();
    }

    public int embeddingSize() {
        return inputs.columns();
    }

    private static int[][] copyTargets(
            int[][] source
    ) {

        int[][] copy =
                new int[source.length][];

        for (int i = 0;
             i < source.length;
             i++) {

            if (source[i] == null) {
                throw new IllegalArgumentException(
                        "Target row cannot be null"
                );
            }

            copy[i] =
                    source[i].clone();
        }

        return copy;
    }
}