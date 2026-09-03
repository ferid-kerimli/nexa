package com.nexa.ai.ai.tensor;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class Tensor3DTest {

    @Test
    void shouldCreateTensor() {

        Tensor3D tensor =
                new Tensor3D(2, 3, 4);

        assertEquals(2, tensor.depth());
        assertEquals(3, tensor.rows());
        assertEquals(4, tensor.columns());
    }

    @Test
    void shouldSetAndGetValues() {

        Tensor3D tensor =
                new Tensor3D(2, 3, 4);

        tensor.set(
                1,
                2,
                3,
                42.0
        );

        assertEquals(
                42.0,
                tensor.get(1, 2, 3)
        );
    }

    @Test
    void shouldAddTensors() {

        Tensor3D first =
                new Tensor3D(1, 2, 2);

        Tensor3D second =
                new Tensor3D(1, 2, 2);

        first.set(0, 0, 0, 1.0);
        first.set(0, 0, 1, 2.0);
        first.set(0, 1, 0, 3.0);
        first.set(0, 1, 1, 4.0);

        second.set(0, 0, 0, 5.0);
        second.set(0, 0, 1, 6.0);
        second.set(0, 1, 0, 7.0);
        second.set(0, 1, 1, 8.0);

        Tensor3D result =
                first.add(second);

        assertEquals(6.0, result.get(0, 0, 0));
        assertEquals(8.0, result.get(0, 0, 1));
        assertEquals(10.0, result.get(0, 1, 0));
        assertEquals(12.0, result.get(0, 1, 1));
    }

    @Test
    void shouldSubtractTensors() {

        Tensor3D first =
                new Tensor3D(1, 1, 2);

        Tensor3D second =
                new Tensor3D(1, 1, 2);

        first.set(0, 0, 0, 10.0);
        first.set(0, 0, 1, 20.0);

        second.set(0, 0, 0, 3.0);
        second.set(0, 0, 1, 5.0);

        Tensor3D result =
                first.subtract(second);

        assertEquals(7.0, result.get(0, 0, 0));
        assertEquals(15.0, result.get(0, 0, 1));
    }

    @Test
    void shouldMultiplyByScalar() {

        Tensor3D tensor =
                new Tensor3D(1, 1, 2);

        tensor.set(0, 0, 0, 2.0);
        tensor.set(0, 0, 1, 4.0);

        Tensor3D result =
                tensor.multiply(3.0);

        assertEquals(
                6.0,
                result.get(0, 0, 0)
        );

        assertEquals(
                12.0,
                result.get(0, 0, 1)
        );
    }

    @Test
    void shouldDivideByScalar() {

        Tensor3D tensor =
                new Tensor3D(1, 1, 2);

        tensor.set(0, 0, 0, 10.0);
        tensor.set(0, 0, 1, 20.0);

        Tensor3D result =
                tensor.divide(2.0);

        assertEquals(
                5.0,
                result.get(0, 0, 0)
        );

        assertEquals(
                10.0,
                result.get(0, 0, 1)
        );
    }

    @Test
    void shouldCopyTensor() {

        Tensor3D original =
                new Tensor3D(1, 1, 1);

        original.set(
                0,
                0,
                0,
                10.0
        );

        Tensor3D copy =
                original.copy();

        copy.set(
                0,
                0,
                0,
                99.0
        );

        assertEquals(
                10.0,
                original.get(0, 0, 0)
        );

        assertEquals(
                99.0,
                copy.get(0, 0, 0)
        );
    }

    @Test
    void shouldRejectInvalidDimensions() {

        assertThrows(
                IllegalArgumentException.class,
                () -> new Tensor3D(0, 2, 2)
        );

        assertThrows(
                IllegalArgumentException.class,
                () -> new Tensor3D(2, 0, 2)
        );

        assertThrows(
                IllegalArgumentException.class,
                () -> new Tensor3D(2, 2, 0)
        );
    }

    @Test
    void shouldRejectNullData() {

        assertThrows(
                IllegalArgumentException.class,
                () -> new Tensor3D(null)
        );
    }

    @Test
    void shouldRejectShapeMismatch() {

        Tensor3D first =
                new Tensor3D(2, 3, 4);

        Tensor3D second =
                new Tensor3D(2, 3, 5);

        assertThrows(
                IllegalArgumentException.class,
                () -> first.add(second)
        );
    }

    @Test
    void shouldRejectDivisionByZero() {

        Tensor3D tensor =
                new Tensor3D(1, 1, 1);

        assertThrows(
                IllegalArgumentException.class,
                () -> tensor.divide(0.0)
        );
    }

    @Test
    void shouldReturnIndependentArrayCopy() {

        Tensor3D tensor =
                new Tensor3D(1, 1, 1);

        tensor.set(
                0,
                0,
                0,
                5.0
        );

        double[][][] array =
                tensor.toArray();

        array[0][0][0] = 99.0;

        assertEquals(
                5.0,
                tensor.get(0, 0, 0)
        );
    }
}