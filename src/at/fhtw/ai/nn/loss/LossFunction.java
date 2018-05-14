package at.fhtw.ai.nn.loss;

import at.fhtw.ai.nn.Neuron;

/**
 * Loss functions are used to compute the difference between the expected and the actual neural network output value.
 * <p>
 * Created On: 14.05.2018
 *
 * @author Daniel Kleebinder
 * @since 0.0.1
 */
public interface LossFunction {

    /**
     * Computes the loss value.
     *
     * @param neuron        Output neuron value.
     * @param expectedValue Expected value.
     * @return Loss value.
     */
    double compute(Neuron neuron, double expectedValue);
}