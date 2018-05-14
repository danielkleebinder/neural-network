package at.fhtw.ai.nn.activation;

import at.fhtw.ai.nn.Neuron;

/**
 * Softsign activation function.<br>
 * <code>f(x) = x / (1 + |x|)</code><br><br>
 * Derivative:<br>
 * <code>f'(x) = 1 / (1 + |x|)^2</code>
 * <p>
 * Created On: 14.05.2018
 *
 * @author Daniel Kleebinder
 * @since 0.0.1
 */
public class Softsign implements ActivationFunction {
    private static final long serialVersionUID = -5175503883773980267L;

    @Override
    public double activate(Neuron neuron) {
        double x = neuron.preActivationValue;
        return x / (1.0 + Math.abs(x));
    }

    @Override
    public double derivative(Neuron neuron) {
        double x = neuron.preActivationValue;
        double v = 1.0 + Math.abs(x);
        return 1.0 / (v * v);
    }
}
