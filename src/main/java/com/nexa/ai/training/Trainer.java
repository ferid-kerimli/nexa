package com.nexa.ai.training;

import com.nexa.ai.ai.loss.LossFunction;
import com.nexa.ai.ai.model.NeuralNetwork;
import com.nexa.ai.dataset.Batch;
import com.nexa.ai.dataset.Batcher;
import com.nexa.ai.dataset.Dataset;

import java.util.List;

public class Trainer {

    private final NeuralNetwork network;
    private final LossFunction lossFunction;
    private final TrainingConfig config;
    private final Batcher batcher;

    public Trainer(
            NeuralNetwork network,
            LossFunction lossFunction,
            TrainingConfig config
    ) {

        if (network == null) {
            throw new IllegalArgumentException(
                    "Network cannot be null"
            );
        }

        if (lossFunction == null) {
            throw new IllegalArgumentException(
                    "Loss function cannot be null"
            );
        }

        if (config == null) {
            throw new IllegalArgumentException(
                    "Training config cannot be null"
            );
        }

        this.network = network;
        this.lossFunction = lossFunction;
        this.config = config;
        this.batcher = new Batcher();
    }

    public TrainingResult train(
            Dataset dataset
    ) {

        TrainingResult result =
                new TrainingResult();

        for (
                int epoch = 1;
                epoch <= config.getEpochs();
                epoch++
        ) {

            List<Batch> batches =
                    batcher.createBatches(
                            dataset,
                            config.getBatchSize()
                    );

            double totalLoss = 0.0;

            for (Batch batch : batches) {

                /*
                 * Forward
                 */
                var prediction =
                        network.forward(
                                batch.input()
                        );

                /*
                 * Loss
                 */
                double loss =
                        lossFunction.calculate(
                                prediction,
                                batch.target()
                        );

                totalLoss += loss;

                /*
                 * Gradient
                 */
                var gradient =
                        lossFunction.gradient(
                                prediction,
                                batch.target()
                        );

                /*
                 * Backpropagation
                 */
                network.backward(gradient);

                /*
                 * Update weights
                 */
                network.update();
            }

            double averageLoss =
                    totalLoss / batches.size();

            result.addLoss(averageLoss);

            if (
                    epoch == 1
                            || epoch % config.getLogInterval() == 0
                            || epoch == config.getEpochs()
            ) {

                System.out.printf(
                        "Epoch: %d/%d | Loss: %.6f%n",
                        epoch,
                        config.getEpochs(),
                        averageLoss
                );
            }
        }

        return result;
    }
}