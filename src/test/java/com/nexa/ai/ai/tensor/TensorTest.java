package com.nexa.ai.ai.tensor;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class TensorTest {

    @Test
    void shouldCreateTensor() {

        Tensor tensor = new Tensor(2, 3);

        assertEquals(2, tensor.rows());
        assertEquals(3, tensor.columns());
    }

    @Test
    void shouldGetAndSetValues() {

        Tensor tensor = new Tensor(2, 2);

        tensor.set(0, 0, 5.0);
        tensor.set(1, 1, 10.0);

        assertEquals(5.0, tensor.get(0, 0));
        assertEquals(10.0, tensor.get(1, 1));
    }

    @Test
    void shouldAddTensors() {

        Tensor a = new Tensor(
                new double[][]{
                        {1, 2},
                        {3, 4}
                }
        );

        Tensor b = new Tensor(
                new double[][]{
                        {5, 6},
                        {7, 8}
                }
        );

        Tensor result = a.add(b);

        assertEquals(6, result.get(0, 0));
        assertEquals(8, result.get(0, 1));
        assertEquals(10, result.get(1, 0));
        assertEquals(12, result.get(1, 1));
    }

    @Test
    void shouldSubtractTensors() {

        Tensor a = new Tensor(
                new double[][]{
                        {10, 20},
                        {30, 40}
                }
        );

        Tensor b = new Tensor(
                new double[][]{
                        {1, 2},
                        {3, 4}
                }
        );

        Tensor result = a.subtract(b);

        assertEquals(9, result.get(0, 0));
        assertEquals(18, result.get(0, 1));
        assertEquals(27, result.get(1, 0));
        assertEquals(36, result.get(1, 1));
    }

    @Test
    void shouldMultiplyByScalar() {

        Tensor tensor = new Tensor(
                new double[][]{
                        {1, 2},
                        {3, 4}
                }
        );

        Tensor result = tensor.multiply(2);

        assertEquals(2, result.get(0, 0));
        assertEquals(4, result.get(0, 1));
        assertEquals(6, result.get(1, 0));
        assertEquals(8, result.get(1, 1));
    }

    @Test
    void shouldMultiplyMatrices() {

        Tensor a = new Tensor(
                new double[][]{
                        {1, 2, 3},
                        {4, 5, 6}
                }
        );

        Tensor b = new Tensor(
                new double[][]{
                        {7, 8},
                        {9, 10},
                        {11, 12}
                }
        );

        Tensor result = a.matrixMultiply(b);

        assertEquals(58, result.get(0, 0));
        assertEquals(64, result.get(0, 1));
        assertEquals(139, result.get(1, 0));
        assertEquals(154, result.get(1, 1));
    }

    @Test
    void shouldTranspose() {

        Tensor tensor = new Tensor(
                new double[][]{
                        {1, 2, 3},
                        {4, 5, 6}
                }
        );

        Tensor result = tensor.transpose();

        assertEquals(3, result.rows());
        assertEquals(2, result.columns());

        assertEquals(1, result.get(0, 0));
        assertEquals(4, result.get(0, 1));

        assertEquals(3, result.get(2, 0));
        assertEquals(6, result.get(2, 1));
    }

    @Test
    void shouldDivideByScalar() {

        Tensor tensor =
                new Tensor(
                        new double[][]{
                                {2, 4},
                                {6, 8}
                        }
                );

        Tensor result =
                tensor.divide(2);

        assertEquals(1, result.get(0, 0));
        assertEquals(2, result.get(0, 1));
        assertEquals(3, result.get(1, 0));
        assertEquals(4, result.get(1, 1));
    }
}