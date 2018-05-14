package at.fhtw.ai.nn.activation;

import at.fhtw.ai.nn.Neuron;

import java.io.Serializable;

/**
 * Basic activation function interface.
 * <p>
 * Created On: 24.04.2018
 *
 * @author Daniel Kleebinder
 * @since 0.0.1
 */
public interface ActivationFunction extends Serializable {

    /**
     * Activates the given value.
     *
     * @param value Value.
     * @return Activated value.
     */
    default double activate(double value) {
        return activate(new Neuron.SimpleNeuron(value));
    }

    /**
     * Activation method for neural networks.
     *
     * @param neuron Neuron.
     * @return Activated value.
     */
    double activate(Neuron neuron);

    /**
     * Derivative of the activation function.
     *
     * @param value Value.
     * @return Result.
     */
    default double derivative(double value) {
        return derivative(new Neuron.SimpleNeuron(value));
    }

    /**
     * Derivative of the activation function.
     *
     * @param neuron Neuron.
     * @return Result.
     */
    double derivative(Neuron neuron);

    /**
     * An activation function with a derivative of:<br>
     * <code>f'(x) = f(x) * (1 - f(x))</code>
     * will return true.
     *
     * @return Result.
     */
    default boolean isStochasticDerivative() {
        return false;
    }
}
