package layers;

import java.util.List;
import java.util.Random;

public class FullyConnectedLayer extends Layer{
    private long SEED;
    private final double leak = 0.01;

    private double[][] _weights;
    private int _inLength;
    private int _outLength;
    private double _learningRate;
    private double[] lastZ;
    private double[] lastX;

    public FullyConnectedLayer(long SEED, int _inLength, int _outLength, double _learningRate) {
        this.SEED = SEED;
        this._inLength = _inLength;
        this._outLength = _outLength;
        this._learningRate = _learningRate;

        _weights = new double[_inLength][_outLength];
        setRandomWeigths();
    }

    public double[] fullyConnectedForwardPass(double[] input) {
        lastX = input;

        double[] z = new double[_outLength];
        double[] out = new double[_outLength];

        for (int i = 0; i < _inLength; i++) {
            for (int j = 0; j < _outLength; j++) {
                z[j] += input[i]*_weights[i][j];
            }
        }

        lastZ = z;
        for (int i = 0; i < _inLength; i++) {
            for (int j = 0; j < _outLength; j++) {
                out[j] = reLu(z[j]);
            }
        }
        return out;
    }

    public void setRandomWeigths() {
        Random random = new Random(SEED);

        for (int i = 0; i < _inLength; i++) {
            for (int j = 0; j < _outLength; j++) {
                _weights[i][j] = random.nextGaussian();
            }
        }
    }

    public double reLu(double input) {
        if (input <= 0) {
            return 0;
        } else {
            return input;
        }
    }

    public double derivativeReLu(double input) {
        if (input <= 0) {
            return leak;
        } else {
            return 1;
        }
    }

    @Override
    public double[] getOutput(List<double[][]> input) {
        double[] vector = matrixToVector(input);
        return getOutput(vector);
    }

    @Override
    public double[] getOutput(double[] input) {
        double[] forwardPass = fullyConnectedForwardPass(input);

        if (_nextLayer != null) {
            return _nextLayer.getOutput(forwardPass);
        } else {
            return forwardPass;
        }
    }

    @Override
    public void backPropagation(List<double[][]> dLd0) {
        double[] vector = matrixToVector(dLd0);
        backPropagation(vector);
    }

    @Override
    public void backPropagation(double[] dLd0) {
        double[] dLdx = new double[_inLength];

        double dLdw;
        double d0dz;
        double dzdw;
        double dzdx;

        for (int i = 0; i < _inLength; i++) {
            double dLdx_sum = 0;
            for (int j = 0; j < _outLength; j++) {
                d0dz = derivativeReLu(lastZ[j]);
                dzdw = lastX[i];
                dzdx = _weights[i][j];

                dLdw = dLd0[j]*d0dz*dzdw;
                _weights[i][j] -= dLdw*_learningRate;

                dLdx_sum += dLd0[j]*d0dz*dzdx;
            }
            dLdx[i] = dLdx_sum;
        }
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
        return _outLength;
    }

}
