package com.nexa.ai.ai.tensor;

import java.util.Arrays;

public class Tensor {

    private final int rows;
    private final int columns;
    private final double[][] data;

    public Tensor(int rows, int columns) {

        if (rows <= 0 || columns <= 0) {
            throw new IllegalArgumentException(
                    "Rows and columns must be greater than zero"
            );
        }

        this.rows = rows;
        this.columns = columns;
        this.data = new double[rows][columns];
    }

    public Tensor(double[][] data) {

        if (data == null || data.length == 0) {
            throw new IllegalArgumentException("Data cannot be empty");
        }

        if (data[0] == null || data[0].length == 0) {
            throw new IllegalArgumentException("Data cannot be empty");
        }

        this.rows = data.length;
        this.columns = data[0].length;

        this.data = new double[rows][columns];

        for (int i = 0; i < rows; i++) {

            if (data[i] == null || data[i].length != columns) {
                throw new IllegalArgumentException(
                        "All rows must have the same number of columns"
                );
            }

            this.data[i] = Arrays.copyOf(data[i], columns);
        }
    }

    public int rows() {
        return rows;
    }

    public int columns() {
        return columns;
    }

    public double get(int row, int column) {
        validateIndex(row, column);
        return data[row][column];
    }

    public void set(int row, int column, double value) {
        validateIndex(row, column);
        data[row][column] = value;
    }

    public Tensor add(Tensor other) {

        if (other == null) {
            throw new IllegalArgumentException(
                    "Tensor cannot be null"
            );
        }

        // Same-shape addition
        if (this.rows == other.rows
                && this.columns == other.columns) {

            Tensor result =
                    new Tensor(rows, columns);

            for (int i = 0; i < rows; i++) {

                for (int j = 0; j < columns; j++) {

                    result.data[i][j] =
                            this.data[i][j]
                                    + other.data[i][j];
                }
            }

            return result;
        }

        // Row-vector broadcasting
        // Example:
        // 2x3 + 1x3 = 2x3
        if (other.rows == 1
                && other.columns == this.columns) {

            Tensor result =
                    new Tensor(rows, columns);

            for (int i = 0; i < rows; i++) {

                for (int j = 0; j < columns; j++) {

                    result.data[i][j] =
                            this.data[i][j]
                                    + other.data[0][j];
                }
            }

            return result;
        }

        throw new IllegalArgumentException(
                "Incompatible tensor shapes for addition: "
                        + this.rows + "x" + this.columns
                        + " and "
                        + other.rows + "x" + other.columns
        );
    }

    public Tensor subtract(Tensor other) {

        validateSameShape(other);

        Tensor result = new Tensor(rows, columns);

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < columns; j++) {

                result.data[i][j] =
                        this.data[i][j] - other.data[i][j];
            }
        }

        return result;
    }

    public Tensor multiply(double scalar) {

        Tensor result = new Tensor(rows, columns);

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < columns; j++) {

                result.data[i][j] =
                        this.data[i][j] * scalar;
            }
        }

        return result;
    }

    public Tensor divide(double scalar) {

        if (scalar == 0) {
            throw new IllegalArgumentException(
                    "Cannot divide by zero"
            );
        }

        Tensor result =
                new Tensor(rows, columns);

        for (int i = 0; i < rows; i++) {

            for (int j = 0; j < columns; j++) {

                result.set(
                        i,
                        j,
                        data[i][j] / scalar
                );
            }
        }

        return result;
    }

    public Tensor matrixMultiply(Tensor other) {

        if (this.columns != other.rows) {
            throw new IllegalArgumentException(
                    "Invalid matrix dimensions: "
                            + this.rows + "x" + this.columns
                            + " cannot be multiplied by "
                            + other.rows + "x" + other.columns
            );
        }

        Tensor result =
                new Tensor(this.rows, other.columns);

        for (int i = 0; i < this.rows; i++) {

            for (int j = 0; j < other.columns; j++) {

                double sum = 0.0;

                for (int k = 0; k < this.columns; k++) {

                    sum += this.data[i][k]
                            * other.data[k][j];
                }

                result.data[i][j] = sum;
            }
        }

        return result;
    }

    public Tensor transpose() {

        Tensor result =
                new Tensor(columns, rows);

        for (int i = 0; i < rows; i++) {

            for (int j = 0; j < columns; j++) {

                result.data[j][i] =
                        this.data[i][j];
            }
        }

        return result;
    }

    public double[][] toArray() {

        double[][] copy =
                new double[rows][columns];

        for (int i = 0; i < rows; i++) {

            copy[i] =
                    Arrays.copyOf(data[i], columns);
        }

        return copy;
    }

    private void validateSameShape(Tensor other) {

        if (other == null) {
            throw new IllegalArgumentException(
                    "Tensor cannot be null"
            );
        }

        if (this.rows != other.rows
                || this.columns != other.columns) {

            throw new IllegalArgumentException(
                    "Tensor shapes must match"
            );
        }
    }

    private void validateIndex(int row, int column) {

        if (row < 0 || row >= rows) {
            throw new IndexOutOfBoundsException(
                    "Invalid row: " + row
            );
        }

        if (column < 0 || column >= columns) {
            throw new IndexOutOfBoundsException(
                    "Invalid column: " + column
            );
        }
    }

    @Override
    public String toString() {

        StringBuilder builder =
                new StringBuilder();

        for (int i = 0; i < rows; i++) {

            builder.append("[ ");

            for (int j = 0; j < columns; j++) {

                builder.append(data[i][j]);

                if (j < columns - 1) {
                    builder.append(", ");
                }
            }

            builder.append(" ]");

            if (i < rows - 1) {
                builder.append(System.lineSeparator());
            }
        }

        return builder.toString();
    }
}