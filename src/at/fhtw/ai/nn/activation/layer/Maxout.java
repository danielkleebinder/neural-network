package at.fhtw.ai.nn.activation.layer;

import at.fhtw.ai.nn.Layer;

/**
 * Created On: 14.05.2018
 *
 * @author Daniel Kleebinder
 * @since 0.0.1
 */
public class Maxout implements LayerActivationFunction {

    /**
     * Function arg-max for xi.
     */
    private double functionArgMax = Double.MIN_VALUE;

    @Override
    public void initialize(Layer layer) {
        functionArgMax = Double.MIN_VALUE;
        layer.getNeurons().forEach(neuron -> functionArgMax = Math.max(functionArgMax, neuron.value));
    }

    @Override
    public double activate(double x) {
        return functionArgMax;
    }

    @Override
    public double derivative(double x) {
        return (Double.compare(x, functionArgMax) == 0) ? 1.0 : 0.0;
    }
}