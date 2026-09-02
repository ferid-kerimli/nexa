package com.nexa.ai.ai.model;

import com.nexa.ai.ai.layers.DenseLayer;
import com.nexa.ai.ai.layers.Layer;
import com.nexa.ai.ai.optimizer.Optimizer;
import com.nexa.ai.ai.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;

public class NeuralNetwork {

    private final List<Layer> layers =
            new ArrayList<>();

    private final Optimizer optimizer;

    public NeuralNetwork(Optimizer optimizer) {

        if (optimizer == null) {
            throw new IllegalArgumentException(
                    "Optimizer cannot be null"
            );
        }

        this.optimizer = optimizer;
    }

    public void addLayer(Layer layer) {

        if (layer == null) {
            throw new IllegalArgumentException(
                    "Layer cannot be null"
            );
        }

        layers.add(layer);
    }

    public Tensor forward(Tensor input) {

        Tensor output = input;

        for (Layer layer : layers) {
            output = layer.forward(output);
        }

        return output;
    }

    public Tensor backward(Tensor gradient) {

        Tensor currentGradient = gradient;

        for (int i = layers.size() - 1; i >= 0; i--) {

            currentGradient =
                    layers.get(i)
                            .backward(currentGradient);
        }

        return currentGradient;
    }

    public void update() {

        for (Layer layer : layers) {

            if (layer instanceof DenseLayer denseLayer) {
                optimizer.update(denseLayer);
            }
        }
    }

    public int getLayerCount() {
        return layers.size();
    }
}