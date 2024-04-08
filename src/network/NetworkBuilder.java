package network;

import layers.Layer;
import layers.MaxPoolLayer;
import layers.ConvolutionLayer;
import layers.FullyConnectedLayer;

import java.util.ArrayList;
import java.util.List;

public class NetworkBuilder {
    private NeuralNetwork net;
    private int _inputRows;
    private int _inputCols;
    private double _scaleFactor;
    List<Layer> _layers;

    public NetworkBuilder(int _inRows, int _inCols, double _scaleFactor) {
        this._inputRows = _inputRows;
        this._inputCols = _inputCols;
        this._scaleFactor = _scaleFactor;
        _layers = new ArrayList<>();
    }

    public void addConvolutionLayer(long SEED, int filterSize, int stepSize, double learningRate, int numFilters) {
        if (_layers.isEmpty()) {
            _layers.add(new ConvolutionLayer(SEED, filterSize, stepSize, 1, _inputRows, _inputCols, learningRate, numFilters));
        } else {
            Layer prev = _layers.getLast();
            _layers.add(new ConvolutionLayer(SEED, filterSize, stepSize, prev.getOutputLength(), prev.getOutputRows(), prev.getOutputCols(), learningRate, numFilters));
        }
    }
    public void addMaxPoolLayer(int stepSize, int windowSize) {
        if (_layers.isEmpty()) {
            _layers.add(new MaxPoolLayer(stepSize, windowSize, 1, _inputRows, _inputCols));
        } else {
            Layer prev = _layers.getLast();
            _layers.add(new MaxPoolLayer(stepSize, windowSize, prev.getOutputLength(), prev.getOutputRows(), prev.getOutputCols()));
        }
    }
    public void addFullyConnectedLayer(int outLength, double learningRate, long SEED) {
        if (_layers.isEmpty()) {
            _layers.add(new FullyConnectedLayer(SEED, _inputCols*_inputRows, outLength, learningRate));
        } else {
            Layer prev = _layers.getLast();
            _layers.add(new FullyConnectedLayer(SEED, prev.getOutputElements(), outLength, learningRate));
        }
    }
    public NeuralNetwork build(){
        net = new NeuralNetwork(_layers, _scaleFactor);
        return net;
    }
}
