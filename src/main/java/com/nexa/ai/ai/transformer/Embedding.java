package com.nexa.ai.ai.transformer;

import java.util.Random;

public class Embedding {

    private final int vocabularySize;
    private final int embeddingSize;

    private final double[][] weights;

    private final Random random;

    public Embedding(
            int vocabularySize,
            int embeddingSize
    ) {

        if (vocabularySize <= 0) {
            throw new IllegalArgumentException(
                    "Vocabulary size must be greater than zero"
            );
        }

        if (embeddingSize <= 0) {
            throw new IllegalArgumentException(
                    "Embedding size must be greater than zero"
            );
        }

        this.vocabularySize = vocabularySize;
        this.embeddingSize = embeddingSize;

        this.weights =
                new double[vocabularySize][embeddingSize];

        this.random = new Random();

        initialize();
    }

    private void initialize() {

        double limit =
                1.0 / Math.sqrt(embeddingSize);

        for (int token = 0;
             token < vocabularySize;
             token++) {

            for (int dimension = 0;
                 dimension < embeddingSize;
                 dimension++) {

                weights[token][dimension] =
                        -limit
                                + random.nextDouble()
                                * (2 * limit);
            }
        }
    }

    public double[] forward(int tokenId) {

        validateTokenId(tokenId);

        return weights[tokenId].clone();
    }

    public double[][] forward(int[] tokenIds) {

        if (tokenIds == null) {
            throw new IllegalArgumentException(
                    "Token IDs cannot be null"
            );
        }

        double[][] result =
                new double[tokenIds.length][embeddingSize];

        for (int i = 0; i < tokenIds.length; i++) {

            validateTokenId(tokenIds[i]);

            result[i] =
                    weights[tokenIds[i]].clone();
        }

        return result;
    }

    public int getVocabularySize() {
        return vocabularySize;
    }

    public int getEmbeddingSize() {
        return embeddingSize;
    }

    public double[][] getWeights() {
        return copyWeights();
    }

    private void validateTokenId(int tokenId) {

        if (tokenId < 0 ||
                tokenId >= vocabularySize) {

            throw new IllegalArgumentException(
                    "Token ID out of range: " + tokenId
            );
        }
    }

    private double[][] copyWeights() {

        double[][] copy =
                new double[vocabularySize][embeddingSize];

        for (int i = 0; i < vocabularySize; i++) {
            System.arraycopy(
                    weights[i],
                    0,
                    copy[i],
                    0,
                    embeddingSize
            );
        }

        return copy;
    }
}