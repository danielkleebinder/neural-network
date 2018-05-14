package at.fhtw.ai.nn.loss;

import at.fhtw.ai.nn.Neuron;

/**
 * Hellinger loss function:<br>
 * <code>C = (sqrt(v) - sqrt(z)) / (sqrt(2) * sqrt(v))</code><br>
 * where z is the expected and v the actual value.
 * <p>
 * Created On: 14.05.2018
 *
 * @author Daniel Kleebinder
 * @since 0.0.1
 */
public class Hellinger implements LossFunction {

    /**
     * Square root of 2.
     */
    private static final double SQRT_2 = Math.sqrt(2.0);

    @Override
    public double compute(Neuron neuron, double expectedValue) {
        double actualValue = neuron.value;
        double t = Math.sqrt(actualValue) - Math.sqrt(expectedValue);
        double b = SQRT_2 * Math.sqrt(actualValue);
        return (t / b) * neuron.getActivationFunction().derivative(neuron);
    }
}
