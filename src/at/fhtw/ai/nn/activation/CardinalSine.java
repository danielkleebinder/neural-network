package at.fhtw.ai.nn.activation;

import at.fhtw.ai.nn.Neuron;

/**
 * Cardinal sine (also known as "Sinc") activation function.<br>
 * <code>f(x) = 1 for x = 0</code><br>
 * <code>f(x) = sin(x)/x for x != 0</code><br><br>
 * Derivative:<br>
 * <code>f'(x) = 0 for x = 0</code><br>
 * <code>f'(x) = (cos(x)/x) - (sin(x)/x^2) for x != 0</code>
 * <p>
 * Created On: 14.05.2018
 *
 * @author Daniel Kleebinder
 * @since 0.0.1
 */
public class CardinalSine implements ActivationFunction {
    private static final long serialVersionUID = 2185537010378014594L;

    @Override
    public double activate(Neuron neuron) {
        double x = neuron.preActivationValue;
        if (Double.compare(x, 0.0) == 0) {
            return 1.0;
        }
        return Math.sin(x) / x;
    }

    @Override
    public double derivative(Neuron neuron) {
        double x = neuron.preActivationValue;
        if (Double.compare(x, 0.0) == 0) {
            return 0.0;
        }
        return (Math.cos(x) / x) - (Math.sin(x) / (x * x));
    }
}