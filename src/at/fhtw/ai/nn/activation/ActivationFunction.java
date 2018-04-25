package at.fhtw.ai.nn.activation;

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
     * Activation method for neural networks.
     *
     * @param x Value.
     * @return Activated value.
     */
    double activate(double x);

    /**
     * Derivative of the activation function.
     *
     * @param x Value.
     * @return Result.
     */
    double derivative(double x);
}
