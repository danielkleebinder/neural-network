package at.fhtw.ai.nn.loss;

import at.fhtw.ai.nn.Neuron;

/**
 * Itakura-Saito distance loss function:<br>
 * <code>C = (z + v^2) / v^2</code><br>
 * where z is the expected and v the actual value.
 * <p>
 * Created On: 14.05.2018
 *
 * @author Daniel Kleebinder
 * @since 0.0.1
 */
public class ItakuraSaito implements LossFunction {

    @Override
    public double compute(Neuron neuron, double expectedValue) {
        double actualValueSquared = neuron.value * neuron.value;
        return ((expectedValue + actualValueSquared) / actualValueSquared) * neuron.getActivationFunction().derivative(neuron);
    }
}
