package at.fhtw.ai.nn.activation;

import at.fhtw.ai.nn.Neuron;

/**
 * Simple, standard identity activation function. The input equals the output.<br>
 * <code>f(x) = x</code><br><br>
 * Derivative:<br>
 * <code>f'(x) = 1</code>
 * <p>
 * Created On: 24.04.2018
 *
 * @author Daniel Kleebinder
 * @since 0.0.1
 */
public class Identity implements ActivationFunction {
    private static final long serialVersionUID = -6905753012349704117L;

    @Override
    public double activate(Neuron neuron) {
        return neuron.preActivationValue;
    }

    @Override
    public double derivative(Neuron neuron) {
        return 1;
    }
}
