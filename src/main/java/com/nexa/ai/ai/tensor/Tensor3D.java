package com.nexa.ai.ai.tensor;

import java.util.Arrays;

public class Tensor3D {

    private final int depth;
    private final int rows;
    private final int columns;

    private final double[][][] data;

    public Tensor3D(
            int depth,
            int rows,
            int columns
    ) {

        if (depth <= 0) {
            throw new IllegalArgumentException(
                    "Depth must be greater than zero"
            );
        }

        if (rows <= 0) {
            throw new IllegalArgumentException(
                    "Rows must be greater than zero"
            );
        }

        if (columns <= 0) {
            throw new IllegalArgumentException(
                    "Columns must be greater than zero"
            );
        }

        this.depth = depth;
        this.rows = rows;
        this.columns = columns;

        this.data =
                new double[depth][rows][columns];
    }

    public Tensor3D(double[][][] data) {

        if (data == null ||
                data.length == 0) {

            throw new IllegalArgumentException(
                    "Data cannot be null or empty"
            );
        }

        if (data[0] == null ||
                data[0].length == 0) {

            throw new IllegalArgumentException(
                    "Rows cannot be null or empty"
            );
        }

        if (data[0][0] == null ||
                data[0][0].length == 0) {

            throw new IllegalArgumentException(
                    "Columns cannot be null or empty"
            );
        }

        this.depth = data.length;
        this.rows = data[0].length;
        this.columns = data[0][0].length;

        this.data =
                new double[depth][rows][columns];

        for (int d = 0; d < depth; d++) {

            if (data[d] == null ||
                    data[d].length != rows) {

                throw new IllegalArgumentException(
                        "Data must have consistent row dimensions"
                );
            }

            for (int i = 0; i < rows; i++) {

                if (data[d][i] == null ||
                        data[d][i].length != columns) {

                    throw new IllegalArgumentException(
                            "Data must have consistent column dimensions"
                    );
                }

                System.arraycopy(
                        data[d][i],
                        0,
                        this.data[d][i],
                        0,
                        columns
                );
            }
        }
    }

    public int depth() {
        return depth;
    }

    public int rows() {
        return rows;
    }

    public int columns() {
        return columns;
    }

    public double get(
            int d,
            int row,
            int column
    ) {

        validateIndices(
                d,
                row,
                column
        );

        return data[d][row][column];
    }

    public void set(
            int d,
            int row,
            int column,
            double value
    ) {

        validateIndices(
                d,
                row,
                column
        );

        data[d][row][column] = value;
    }

    public Tensor3D add(
            Tensor3D other
    ) {

        validateSameShape(other);

        Tensor3D result =
                new Tensor3D(
                        depth,
                        rows,
                        columns
                );

        for (int d = 0; d < depth; d++) {

            for (int i = 0; i < rows; i++) {

                for (int j = 0; j < columns; j++) {

                    result.data[d][i][j] =
                            data[d][i][j]
                                    + other.data[d][i][j];
                }
            }
        }

        return result;
    }

    public Tensor3D subtract(
            Tensor3D other
    ) {

        validateSameShape(other);

        Tensor3D result =
                new Tensor3D(
                        depth,
                        rows,
                        columns
                );

        for (int d = 0; d < depth; d++) {

            for (int i = 0; i < rows; i++) {

                for (int j = 0; j < columns; j++) {

                    result.data[d][i][j] =
                            data[d][i][j]
                                    - other.data[d][i][j];
                }
            }
        }

        return result;
    }

    public Tensor3D multiply(
            double scalar
    ) {

        Tensor3D result =
                new Tensor3D(
                        depth,
                        rows,
                        columns
                );

        for (int d = 0; d < depth; d++) {

            for (int i = 0; i < rows; i++) {

                for (int j = 0; j < columns; j++) {

                    result.data[d][i][j] =
                            data[d][i][j]
                                    * scalar;
                }
            }
        }

        return result;
    }

    public Tensor3D divide(
            double scalar
    ) {

        if (scalar == 0.0) {
            throw new IllegalArgumentException(
                    "Cannot divide by zero"
            );
        }

        return multiply(1.0 / scalar);
    }

    public Tensor3D copy() {
        return new Tensor3D(data);
    }

    public double[][][] toArray() {

        double[][][] copy =
                new double[depth][rows][columns];

        for (int d = 0; d < depth; d++) {

            for (int i = 0; i < rows; i++) {

                System.arraycopy(
                        data[d][i],
                        0,
                        copy[d][i],
                        0,
                        columns
                );
            }
        }

        return copy;
    }

    private void validateIndices(
            int d,
            int row,
            int column
    ) {

        if (d < 0 || d >= depth) {
            throw new IndexOutOfBoundsException(
                    "Invalid depth index: " + d
            );
        }

        if (row < 0 || row >= rows) {
            throw new IndexOutOfBoundsException(
                    "Invalid row index: " + row
            );
        }

        if (column < 0 || column >= columns) {
            throw new IndexOutOfBoundsException(
                    "Invalid column index: " + column
            );
        }
    }

    private void validateSameShape(
            Tensor3D other
    ) {

        if (other == null) {
            throw new IllegalArgumentException(
                    "Tensor cannot be null"
            );
        }

        if (depth != other.depth ||
                rows != other.rows ||
                columns != other.columns) {

            throw new IllegalArgumentException(
                    "Tensor shapes do not match"
            );
        }
    }

    @Override
    public String toString() {

        return Arrays.deepToString(data);
    }
}