package com.nexa.ai.training;

public class TrainingConfig {

    private final int epochs;
    private final int batchSize;
    private final int logInterval;

    public TrainingConfig(
            int epochs,
            int batchSize,
            int logInterval
    ) {

        if (epochs <= 0) {
            throw new IllegalArgumentException(
                    "Epochs must be greater than zero"
            );
        }

        if (batchSize <= 0) {
            throw new IllegalArgumentException(
                    "Batch size must be greater than zero"
            );
        }

        if (logInterval <= 0) {
            throw new IllegalArgumentException(
                    "Log interval must be greater than zero"
            );
        }

        this.epochs = epochs;
        this.batchSize = batchSize;
        this.logInterval = logInterval;
    }

    public int getEpochs() {
        return epochs;
    }

    public int getBatchSize() {
        return batchSize;
    }

    public int getLogInterval() {
        return logInterval;
    }
}