package at.fhtw.ai.nn.activation;

import at.fhtw.ai.nn.Neuron;

/**
 * Arcus tangents activation function.<br>
 * <code>f(x) = atan(x)</code><br><br>
 * Derivative:<br>
 * <code>f'(x) = 1 / (x^2 + 1)</code>
 * <p>
 * Created On: 14.05.2018
 *
 * @author Daniel Kleebinder
 * @since 0.0.1
 */
public class ArcTan implements ActivationFunction {
    private static final long serialVersionUID = -3889979995966021084L;

    @Override
    public double activate(Neuron neuron) {
        return Math.atan(neuron.preActivationValue);
    }

    @Override
    public double derivative(Neuron neuron) {
        return 1.0 / (neuron.preActivationValue * neuron.preActivationValue + 1.0);
    }
}
