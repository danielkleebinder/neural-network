package at.fhtw.ai.nn.loss;

import at.fhtw.ai.nn.Neuron;

/**
 * Qudratic loss function using standard loss computation:<br>
 * <code>C = z - v</code><br>
 * where z is the expected and v the actual value.
 * <p>
 * Created On: 14.05.2018
 *
 * @author Daniel Kleebinder
 * @since 0.0.1
 */
public class Quadratic implements LossFunction {

    @Override
    public double compute(Neuron neuron, double expectedValue) {
        return (expectedValue - neuron.value) * neuron.getActivationFunction().derivative(neuron);
    }
}