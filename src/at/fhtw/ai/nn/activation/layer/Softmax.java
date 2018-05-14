package at.fhtw.ai.nn.activation.layer;

import at.fhtw.ai.nn.Layer;

/**
 * Created On: 14.05.2018
 *
 * @author Daniel Kleebinder
 * @since 0.0.1
 */
public class Softmax implements LayerActivationFunction {

    private double sum = 0.0;

    @Override
    public void initialize(Layer layer) {
        sum = 0.0;
        layer.getNeurons().forEach(neuron -> sum += Math.exp(neuron.value));
    }

    @Override
    public double activate(double x) {
        return Math.exp(x) / sum;
    }

    @Override
    public double derivative(double x) {
        return x * (kroneckerDelta(0, 0) - x);
    }

    private double kroneckerDelta(int i, int j) {
        return (i == j) ? 1.0 : 0.0;
    }
}