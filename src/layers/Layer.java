package layers;

import java.util.ArrayList;
import java.util.List;

public abstract class Layer {
    protected Layer _nextLayer;
    protected Layer _previousLayer;
    public Layer get_nextLayer() {
        return _nextLayer;
    }
    public void set_nextLayer(Layer _nextLayer) {
        this._nextLayer = _nextLayer;
    }
    public Layer get_previousLayer() {
        return _previousLayer;
    }
    public void set_previousLayer(Layer _previousLayer) {
        this._previousLayer = _previousLayer;
    }
    public abstract double[] getOutput(List<double[][]> input);
    public abstract double[] getOutput(double[] input);

    public abstract int getOuputLength();
    public abstract int getOuputRows();
    public abstract int getOuputCols();
    public abstract int getOuputElements();

    public abstract void backPropagation(List<double[][]> dLd0);
    public abstract void backPropagation(double[] dLd0);

    public double[] matrixToVector(List<double[][]> input) {
        int length = input.size();
        int rows = input.get(0).length;
        int cols = input.get(0)[0].length;

        double[] result = new double[length*rows*cols];
        int l = 0;
        for (int i = 0; i < length; i++) {
            for (int j = 0; j < rows; j++) {
                for (int k = 0; k < cols; k++) {
                    result[l] = input.get(i)[j][k];
                    l++;
                }
            }
        }
        return result;
    }

    public List<double[][]> vectorToMatrix(double[] input, int length, int rows, int cols) {
        List<double[][]> result = new ArrayList<>();

        int l = 0;
        for (int i = 0; i < length; i++) {
            double[][] matrix = new double[rows][cols];
            for (int j = 0; j < rows; j++) {
                for (int k = 0; k < cols; k++) {
                    matrix[j][k] = input[l];
                    l++;
                }
            }
            result.add(matrix);
        }
        return result;
    }
}