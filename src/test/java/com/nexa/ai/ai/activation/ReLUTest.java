package com.nexa.ai.ai.activation;

import com.nexa.ai.ai.tensor.Tensor;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class ReLUTest {

    @Test
    void shouldSetNegativeValuesToZero() {

        Tensor input =
                new Tensor(
                        new double[][]{
                                {-2, -1, 0, 1, 2}
                        }
                );

        ReLU relu = new ReLU();

        Tensor output =
                relu.forward(input);

        assertEquals(0, output.get(0, 0));
        assertEquals(0, output.get(0, 1));
        assertEquals(0, output.get(0, 2));
        assertEquals(1, output.get(0, 3));
        assertEquals(2, output.get(0, 4));
    }

    @Test
    void shouldCalculateBackwardGradient() {

        Tensor input =
                new Tensor(
                        new double[][]{
                                {-2, 2}
                        }
                );

        Tensor gradient =
                new Tensor(
                        new double[][]{
                                {10, 10}
                        }
                );

        ReLU relu = new ReLU();

        Tensor result =
                relu.backward(input, gradient);

        assertEquals(0, result.get(0, 0));
        assertEquals(10, result.get(0, 1));
    }
}