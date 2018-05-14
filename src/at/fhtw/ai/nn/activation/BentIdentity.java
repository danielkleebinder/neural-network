package at.fhtw.ai.nn.activation;

import at.fhtw.ai.nn.Neuron;

/**
 * Bent identity activation function.<br>
 * <code>f(x) = ((sqrt(x^2 + 1) - 1) / 2) + x</code><br><br>
 * Derivative:<br>
 * <code>f'(x) = (x / (2 * sqrt(x^2 + 1))) + 1</code>
 * <p>
 * Created On: 14.05.2018
 *
 * @author Daniel Kleebinder
 * @since 0.0.1
 */
public class BentIdentity implements ActivationFunction {
    private static final long serialVersionUID = 6158579572102114332L;

    @Override
    public double activate(Neuron neuron) {
        double x = neuron.preActivationValue;
        return ((Math.sqrt(x * x + 1.0) - 1.0) / 2.0) + x;
    }

    @Override
    public double derivative(Neuron neuron) {
        double x = neuron.preActivationValue;
        return (x / (2.0 * Math.sqrt(x * x + 1.0))) + 1.0;
    }
}
