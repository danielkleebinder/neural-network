package at.fhtw.ai.nn.activation.layer;

import at.fhtw.ai.nn.Layer;
import at.fhtw.ai.nn.Neuron;

/**
 * Maxout activation function.<br>
 * <code>f(x) = max_i(x_i)</code><br><br>
 * Derivative:<br>
 * <code>f'(x) = 1 for j = argmax_i(x_i)</code><br>
 * <code>f'(x) = 0 for j != argmax_i(x_i)</code><br>
 * <p>
 * Created On: 14.05.2018
 *
 * @author Daniel Kleebinder
 * @since 0.0.1
 */
public class Maxout implements LayerActivationFunction {
    private static final long serialVersionUID = 2303260477953539433L;

    /**
     * Function arg-max for x_i.
     */
    private double functionArgMax = Double.MIN_VALUE;

    @Override
    public void initialize(Layer layer) {
        functionArgMax = Double.MIN_VALUE;
        layer.getNeurons().forEach(neuron -> functionArgMax = Math.max(functionArgMax, neuron.value));
    }

    @Override
    public double activate(Neuron neuron) {
        return functionArgMax;
    }

    @Override
    public double derivative(Neuron neuron) {
        return (Double.compare(neuron.preActivationValue, functionArgMax) == 0) ? 1.0 : 0.0;
    }
}