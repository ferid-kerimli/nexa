package com.nexa.ai.ai.optimizer;

import com.nexa.ai.ai.layers.DenseLayer;

public interface Optimizer {

    void update(DenseLayer layer);
}