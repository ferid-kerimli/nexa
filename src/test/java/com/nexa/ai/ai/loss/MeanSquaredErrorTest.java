package com.nexa.ai.ai.loss;

import com.nexa.ai.ai.tensor.Tensor;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class MeanSquaredErrorTest {

    @Test
    void shouldCalculateLoss() {

        Tensor prediction =
                new Tensor(
                        new double[][]{
                                {5}
                        }
                );

        Tensor target =
                new Tensor(
                        new double[][]{
                                {7}
                        }
                );

        MeanSquaredError mse =
                new MeanSquaredError();

        double loss =
                mse.calculate(
                        prediction,
                        target
                );

        assertEquals(
                4.0,
                loss,
                0.000001
        );
    }

    @Test
    void shouldCalculateGradient() {

        Tensor prediction =
                new Tensor(
                        new double[][]{
                                {5}
                        }
                );

        Tensor target =
                new Tensor(
                        new double[][]{
                                {7}
                        }
                );

        MeanSquaredError mse =
                new MeanSquaredError();

        Tensor gradient =
                mse.gradient(
                        prediction,
                        target
                );

        assertEquals(
                -4.0,
                gradient.get(0, 0),
                0.000001
        );
    }
}