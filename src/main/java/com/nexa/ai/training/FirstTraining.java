package com.nexa.ai.training;

import com.nexa.ai.ai.activation.ReLU;
import com.nexa.ai.ai.layers.ActivationLayer;
import com.nexa.ai.ai.layers.DenseLayer;
import com.nexa.ai.ai.loss.MeanSquaredError;
import com.nexa.ai.ai.model.NeuralNetwork;
import com.nexa.ai.ai.optimizer.SGD;
import com.nexa.ai.ai.tensor.Tensor;
import com.nexa.ai.dataset.DataPoint;
import com.nexa.ai.dataset.Dataset;

import java.util.List;

public class FirstTraining {

    private static DataPoint dataPoint(
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

    public static void main(String[] args) {

        /*
         * Create model
         */
        NeuralNetwork network =
                new NeuralNetwork(
                        new SGD(0.01)
                );

        network.addLayer(
                new DenseLayer(1, 8)
        );

        network.addLayer(
                new ActivationLayer(
                        new ReLU()
                )
        );

        network.addLayer(
                new DenseLayer(8, 1)
        );

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

        Tensor input =
                new Tensor(
                        new double[][]{
                                {1},
                                {2},
                                {3},
                                {4},
                                {5}
                        }
                );

        Tensor target =
                new Tensor(
                        new double[][]{
                                {2},
                                {4},
                                {6},
                                {8},
                                {10}
                        }
                );

        /*
         * Loss
         */
        MeanSquaredError loss =
                new MeanSquaredError();

        /*
         * Training configuration
         */
        TrainingConfig config =
                new TrainingConfig(
                        1000,
                        2,
                        100
                );

        /*
         * Trainer
         */
        Trainer trainer =
                new Trainer(
                        network,
                        loss,
                        config
                );

        /*
         * Train
         */
        TrainingResult result =
                trainer.train(
                        dataset
                );

        /*
         * Results
         */
        System.out.println();
        System.out.println("Training finished.");

        System.out.printf(
                "Initial loss: %.6f%n",
                result.getInitialLoss()
        );

        System.out.printf(
                "Final loss: %.6f%n",
                result.getFinalLoss()
        );

        /*
         * Test model
         */
        Tensor predictions =
                network.forward(input);

        System.out.println();
        System.out.println("Predictions:");

        System.out.println(predictions);
    }
}