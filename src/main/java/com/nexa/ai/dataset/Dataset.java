package com.nexa.ai.dataset;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class Dataset {

    private final List<DataPoint> dataPoints;

    public Dataset(List<DataPoint> dataPoints) {

        if (dataPoints == null || dataPoints.isEmpty()) {
            throw new IllegalArgumentException(
                    "Dataset cannot be empty"
            );
        }

        this.dataPoints =
                new ArrayList<>(dataPoints);
    }

    public int size() {
        return dataPoints.size();
    }

    public DataPoint get(int index) {
        return dataPoints.get(index);
    }

    public List<DataPoint> getDataPoints() {
        return Collections.unmodifiableList(
                dataPoints
        );
    }
}