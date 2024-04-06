package layers;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class ConvolutionLayer extends Layer{

    private long SEED;

    private List<double[][]> _filters;
    private int _filterSize;
    private int _stepSize;

    private int _inLength;
    private int _inRows;
    private int _inCols;
    private double _learningRate;

    public ConvolutionLayer(long SEED, int _filterSize, int _stepSize, int _inLength, int _inRows, int _inCols, double learningRate, int numFilters) {
        this.SEED = SEED;
        this._filterSize = _filterSize;
        this._stepSize = _stepSize;
        this._inLength = _inLength;
        this._inRows = _inRows;
        this._inCols = _inCols;
        _learningRate = learningRate;

        generateRandomFilter(numFilters);
    }

    public void generateRandomFilter(int numFilters) {
        List<double[][]> filters = new ArrayList<>();
        Random random = new Random();

        for (int n = 0; n < numFilters; n++) {
            double[][] newFilter = new double[_filterSize][_filterSize];
            for (int i = 0; i < _filterSize; i++) {
                for (int j = 0; j < _filterSize; j++) {
                    double value = random.nextGaussian();
                    newFilter[i][j] = value;
                }
            }
            filters.add(newFilter);
        }

        _filters = filters;
    }

    public List<double[][]> convolutionalForwardPass(List<double[][]> list) {
        List<double[][]> output = new ArrayList<>();

        for (int i = 0; i < list.size(); i++) {
            for (double[][] filter : _filters) {
                output.add(convolve(list.get(i), filter, _stepSize));
            }
        }

        return output;
    }

    public double[][] convolve(double[][] input, double[][] filter, int stepSize) {
        int outRows = (input.length - filter.length)/stepSize + 1;
        int outCols = (input[0].length - filter[0].length)/stepSize + 1;

        int inRows = input.length;
        int inCols = input[0].length;

        int fRows = filter.length;
        int fCols = filter.length;

        double[][] output = new double[outRows][outCols];

        int outRow = 0;
        int outCol;

        for (int i = 0; i <= (inRows - fRows); i += stepSize) {
            outCol = 0;
            for (int j = 0; j <= (inCols - fCols); i += stepSize) {
                double sum = 0.0;
                for (int x = 0; x < fRows; x++) {
                    for (int y = 0; y < fCols; y++) {
                        int inputRowIndex = x + i;
                        int inputColIndex = y + j;

                        double value = filter[x][y] * input[inputRowIndex][inputColIndex];
                        sum += value;
                    }
                }
                output[outRow][outCol] = sum;
                outCol++;
            }
            outRow++;
        }
        return output;
    }

    @Override
    public double[] getOutput(List<double[][]> input) {
        return new double[0];
    }

    @Override
    public double[] getOutput(double[] input) {
        return new double[0];
    }

    @Override
    public int getOuputLength() {
        return 0;
    }

    @Override
    public int getOuputRows() {
        return 0;
    }

    @Override
    public int getOuputCols() {
        return 0;
    }

    @Override
    public int getOuputElements() {
        return 0;
    }

    @Override
    public void backPropagation(List<double[][]> dLd0) {

    }

    @Override
    public void backPropagation(double[] dLd0) {

    }
}