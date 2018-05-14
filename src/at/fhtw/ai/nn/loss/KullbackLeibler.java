package at.fhtw.ai.nn.loss;

import at.fhtw.ai.nn.Neuron;

/**
 * Generalized Kullback-Leibler loss function:<br>
 * <code>C = (z + v) / v</code><br>
 * where z is the expected and v the actual value.
 * <p>
 * Created On: 14.05.2018
 *
 * @author Daniel Kleebinder
 * @since 0.0.1
 */
public class KullbackLeibler implements LossFunction {

    @Override
    public double compute(Neuron neuron, double expectedValue) {
        double actualValue = neuron.value;
        return ((expectedValue + actualValue) / actualValue) * neuron.getActivationFunction().derivative(neuron);
    }
}
