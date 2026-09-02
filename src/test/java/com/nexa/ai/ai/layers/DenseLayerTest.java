package com.nexa.ai.ai.layers;

import com.nexa.ai.ai.tensor.Tensor;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class DenseLayerTest {

    @Test
    void shouldCreateLayer() {

        DenseLayer layer =
                new DenseLayer(3, 2);

        assertEquals(3, layer.getInputSize());
        assertEquals(2, layer.getOutputSize());
    }

    @Test
    void shouldProduceCorrectOutputShape() {

        DenseLayer layer =
                new DenseLayer(3, 2);

        Tensor input =
                new Tensor(
                        new double[][]{
                                {1, 2, 3},
                                {4, 5, 6}
                        }
                );

        Tensor output =
                layer.forward(input);

        assertEquals(2, output.rows());
        assertEquals(2, output.columns());
    }

    @Test
    void shouldRejectIncorrectInputSize() {

        DenseLayer layer =
                new DenseLayer(3, 2);

        Tensor input =
                new Tensor(
                        new double[][]{
                                {1, 2}
                        }
                );

        assertThrows(
                IllegalArgumentException.class,
                () -> layer.forward(input)
        );
    }

    @Test
    void shouldHaveCorrectWeightShape() {

        DenseLayer layer =
                new DenseLayer(3, 2);

        Tensor weights =
                layer.getWeights();

        assertEquals(3, weights.rows());
        assertEquals(2, weights.columns());
    }

    @Test
    void shouldHaveCorrectBiasShape() {

        DenseLayer layer =
                new DenseLayer(3, 2);

        Tensor bias =
                layer.getBias();

        assertEquals(1, bias.rows());
        assertEquals(2, bias.columns());
    }
}