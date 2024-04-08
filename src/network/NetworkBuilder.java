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

}
