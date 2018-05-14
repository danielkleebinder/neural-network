package at.fhtw.ai.nn.loss;

import at.fhtw.ai.nn.Neuron;
import at.fhtw.ai.nn.activation.ActivationFunction;

/**
 * Cross entropy loss function.<br>
 * <code>C = -z * log(v) - (1 - z) * log(1 - v)</code><br>
 * where z is the expected value and v the actual value.
 * <p>
 * Created On: 14.05.2018
 *
 * @author Daniel Kleebinder
 * @since 0.0.1
 */
public class CrossEntropy implements LossFunction {

    @Override
    public double compute(Neuron neuron, double expectedValue) {
        double actualValue = neuron.value;
        double t = expectedValue - actualValue;

        ActivationFunction activationFunction = neuron.getActivationFunction();
        if (activationFunction != null && activationFunction.isStochasticDerivative()) {
            return t;
        }

        double b = actualValue * (1.0 - actualValue);
        return (t / b) * activationFunction.derivative(neuron);
    }
}
