package com.nexa.ai.ai.transformer;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class SoftmaxTest {

    @Test
    void probabilitiesShouldSumToOne() {

        Softmax softmax =
                new Softmax();

        double[][] input = {
                {1.0, 2.0, 3.0},
                {5.0, 2.0, 1.0}
        };

        double[][] output =
                softmax.forward(input);

        for (double[] row : output) {

            double sum = 0.0;

            for (double value : row) {
                sum += value;
            }

            assertEquals(
                    1.0,
                    sum,
                    1e-10
            );
        }
    }

    @Test
    void probabilitiesShouldBeBetweenZeroAndOne() {

        Softmax softmax =
                new Softmax();

        double[][] input = {
                {1.0, 2.0, 3.0},
                {-5.0, 0.0, 5.0}
        };

        double[][] output =
                softmax.forward(input);

        for (double[] row : output) {

            for (double value : row) {

                assertTrue(value >= 0.0);
                assertTrue(value <= 1.0);
            }
        }
    }

    @Test
    void largerLogitShouldHaveLargerProbability() {

        Softmax softmax =
                new Softmax();

        double[][] input = {
                {1.0, 2.0, 5.0}
        };

        double[][] output =
                softmax.forward(input);

        assertTrue(
                output[0][2] >
                        output[0][1]
        );

        assertTrue(
                output[0][1] >
                        output[0][0]
        );
    }

    @Test
    void shouldHandleLargeValues() {

        Softmax softmax =
                new Softmax();

        double[][] input = {
                {1000.0, 1001.0, 1002.0}
        };

        double[][] output =
                softmax.forward(input);

        for (double value : output[0]) {

            assertTrue(
                    Double.isFinite(value)
            );
        }

        double sum = 0.0;

        for (double value : output[0]) {
            sum += value;
        }

        assertEquals(
                1.0,
                sum,
                1e-10
        );
    }

    @Test
    void shouldRejectNullInput() {

        Softmax softmax =
                new Softmax();

        assertThrows(
                IllegalArgumentException.class,
                () -> softmax.forward(null)
        );
    }

    @Test
    void shouldRejectEmptyInput() {

        Softmax softmax =
                new Softmax();

        assertThrows(
                IllegalArgumentException.class,
                () -> softmax.forward(
                        new double[0][0]
                )
        );
    }
}