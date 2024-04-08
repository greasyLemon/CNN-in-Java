package layers;

import java.util.ArrayList;
import java.util.List;

public class MaxPoolLayer extends Layer{
    private int _stepSize; 
    private int _windowSize;

    private int _inLength;
    private int _inRows;
    private int _inCols;

    List<int[][]> _lastMaxRow;
    List<int[][]> _lastMaxCol;

    public MaxPoolLayer(int _stepSize, int _windowSize, int _inLength, int _inRows, int _inCols) {
        this._stepSize = _stepSize;
        this._windowSize = _windowSize;
        this._inLength = _inLength;
        this._inRows = _inRows;
        this._inCols = _inCols;
    }

    public List<double[][]> maxPoolForwardPass(List<double[][]> input) {
        List<double[][]> output = new ArrayList<>();
        _lastMaxRow = new ArrayList<>();
        _lastMaxCol = new ArrayList<>();

        for (int i = 0; i < input.size(); i++) {
            output.add(pool(input.get(i)));
        }

        return output;
    }

    public double[][] pool(double[][] input) {
        double[][] output = new double[getOutputRows()][getOutputCols()];
        int[][] maxRows = new int[getOutputRows()][getOutputCols()];
        int[][] maxCols = new int[getOutputRows()][getOutputCols()];

        for (int r = 0; r < getOutputRows(); r += _stepSize) {
            for (int c = 0; c < getOutputCols(); c += _stepSize) {
                double max = 0.0;
                for (int x = 0; x < _windowSize; x++) {
                    for (int y = 0; y < _windowSize; y++) {
                        if (max < input[r+x][c+y]) {
                            max = input[r+x][c+y];
                            maxRows[r][c] = r+x;
                            maxCols[r][c] = c+y;
                        }
                    }
                }
                output[r][c] = max;
            }
        }
        _lastMaxRow.add(maxRows);
        _lastMaxCol.add(maxCols);
        return output;
    }

    @Override
    public double[] getOutput(List<double[][]> input) {
        List<double[][]> outputPool = maxPoolForwardPass(input);
        return _nextLayer.getOutput(outputPool);
    }

    @Override
    public double[] getOutput(double[] input) {
        List<double[][]> matrix = vectorToMatrix(input, _inLength, _inRows, _inCols);
        return getOutput(matrix);
    }

    public void backPropagation(List<double[][]> dLd0) {
        List<double[][]> dLdx = new ArrayList<>();

        int l = 0;
        for (double[][] array : dLd0) {
            double[][] error = new double[_inRows][_inCols];
            for (int r = 0; r < getOutputRows(); r++) {
                for (int c = 0; c < getOutputCols(); c++) {
                    int max_i = _lastMaxRow.get(l)[r][c];
                    int max_j = _lastMaxCol.get(l)[r][c];

                    if (max_i != -1) {
                        error[max_i][max_j] += array[r][c];
                    }
                }
            }
            dLdx.add(error);
            l++;
        }

        if (_previousLayer != null) {
            _previousLayer.backPropagation(dLdx);
        }
    }

    @Override
    public void backPropagation(double[] dLd0) {
        List<double[][]> matrix = vectorToMatrix(dLd0, getOutputLength(), getOutputRows(), getOutputCols());
        backPropagation(matrix);
    }

    @Override
    public int getOutputLength() {
        return _inLength;
    }

    @Override
    public int getOutputRows() {
        return (_inRows-_windowSize)/_stepSize + 1;
    }

    @Override
    public int getOutputCols() {
        return ( _inCols-_windowSize)/_stepSize + 1;
    }

    @Override
    public int getOutputElements() {
        return _inLength*getOutputRows()*getOutputCols();
    }

}
