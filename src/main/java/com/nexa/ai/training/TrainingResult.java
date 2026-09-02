package com.nexa.ai.training;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class TrainingResult {

    private final List<Double> losses =
            new ArrayList<>();

    public void addLoss(double loss) {
        losses.add(loss);
    }

    public List<Double> getLosses() {
        return Collections.unmodifiableList(losses);
    }

    public double getInitialLoss() {

        if (losses.isEmpty()) {
            return Double.NaN;
        }

        return losses.get(0);
    }

    public double getFinalLoss() {

        if (losses.isEmpty()) {
            return Double.NaN;
        }

        return losses.get(losses.size() - 1);
    }
}