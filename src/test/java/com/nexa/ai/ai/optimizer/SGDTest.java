package com.nexa.ai.ai.optimizer;

import com.nexa.ai.ai.layers.DenseLayer;
import com.nexa.ai.ai.tensor.Tensor;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertNotEquals;

class SGDTest {

    @Test
    void shouldUpdateWeights() {

        DenseLayer layer =
                new DenseLayer(2, 1);

        Tensor input =
                new Tensor(
                        new double[][]{
                                {1, 2}
                        }
                );

        Tensor gradient =
                new Tensor(
                        new double[][]{
                                {1}
                        }
                );

        layer.forward(input);

        double oldWeight =
                layer.getWeights().get(0, 0);

        layer.backward(gradient);

        SGD optimizer =
                new SGD(0.01);

        optimizer.update(layer);

        double newWeight =
                layer.getWeights().get(0, 0);

        assertNotEquals(
                oldWeight,
                newWeight
        );
    }
}