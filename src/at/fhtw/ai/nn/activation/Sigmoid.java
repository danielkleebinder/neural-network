package at.fhtw.ai.nn.activation;

import at.fhtw.ai.nn.Neuron;

/**
 * Most commonly used activation function called sigmoid.<br>
 * <code>f(x) = 1 / (1 + e^-x)</code><br><br>
 * Derivative:<br>
 * <code>f'(x) = f(x) * (1 - f(x))</code>
 * <p>
 * Created On: 24.04.2018
 *
 * @author Daniel Kleebinder
 * @since 0.0.1
 */
public class Sigmoid implements ActivationFunction {
    private static final long serialVersionUID = 1613503183509679914L;

    @Override
    public double activate(Neuron neuron) {
        return 1.0 / (1.0 + Math.exp(-neuron.preActivationValue));
    }

    @Override
    public double derivative(Neuron neuron) {
        return neuron.value * (1.0 - neuron.value);
    }

    @Override
    public boolean isStochasticDerivative() {
        return true;
    }
}
