package at.fhtw.ai.nn.activation;

import at.fhtw.ai.nn.Neuron;

/**
 * Soft plus activation function.<br>
 * <code>f(x) = ln(1 + e^x)</code><br><br>
 * Derivative:<br>
 * <code>f'(x) = 1 / (1 + e^-x)</code>
 * <p>
 * Created On: 14.05.2018
 *
 * @author Daniel Kleebinder
 * @since 0.0.1
 */
public class SoftPlus implements ActivationFunction {
    private static final long serialVersionUID = -3817107206480448264L;

    @Override
    public double activate(Neuron neuron) {
        return Math.log(1.0 + Math.exp(neuron.preActivationValue));
    }

    @Override
    public double derivative(Neuron neuron) {
        return 1.0 / (1.0 + Math.exp(-neuron.preActivationValue));
    }
}
