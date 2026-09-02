package com.nexa.ai.ai.loss;

import com.nexa.ai.ai.tensor.Tensor;

public class MeanSquaredError implements LossFunction {

    @Override
    public double calculate(
            Tensor prediction,
            Tensor target
    ) {

        validateShape(prediction, target);

        double sum = 0.0;

        int totalElements =
                prediction.rows()
                        * prediction.columns();

        for (int i = 0; i < prediction.rows(); i++) {

            for (int j = 0; j < prediction.columns(); j++) {

                double difference =
                        prediction.get(i, j)
                                - target.get(i, j);

                sum += difference * difference;
            }
        }

        return sum / totalElements;
    }

    @Override
    public Tensor gradient(
            Tensor prediction,
            Tensor target
    ) {

        validateShape(prediction, target);

        int totalElements =
                prediction.rows()
                        * prediction.columns();

        Tensor gradient =
                new Tensor(
                        prediction.rows(),
                        prediction.columns()
                );

        for (int i = 0; i < prediction.rows(); i++) {

            for (int j = 0; j < prediction.columns(); j++) {

                double value =
                        2.0
                                * (prediction.get(i, j)
                                - target.get(i, j))
                                / totalElements;

                gradient.set(i, j, value);
            }
        }

        return gradient;
    }

    private void validateShape(
            Tensor prediction,
            Tensor target
    ) {

        if (prediction.rows() != target.rows()
                || prediction.columns() != target.columns()) {

            throw new IllegalArgumentException(
                    "Prediction and target shapes must match"
            );
        }
    }
}