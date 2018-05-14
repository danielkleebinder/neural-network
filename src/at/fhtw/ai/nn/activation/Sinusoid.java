package at.fhtw.ai.nn.activation;

import at.fhtw.ai.nn.Neuron;

/**
 * Sinusoid activation function.<br>
 * <code>f(x) = sin(x)</code><br><br>
 * Derivative:<br>
 * <code>f'(x) = cos(x)</code>
 * <p>
 * Created On: 24.04.2018
 *
 * @author Daniel Kleebinder
 * @since 0.0.1
 */
public class Sinusoid implements ActivationFunction {
    private static final long serialVersionUID = 5656466654313704898L;

    @Override
    public double activate(Neuron neuron) {
        return Math.sin(neuron.preActivationValue);
    }

    @Override
    public double derivative(Neuron neuron) {
        return Math.cos(neuron.preActivationValue);
    }
}
