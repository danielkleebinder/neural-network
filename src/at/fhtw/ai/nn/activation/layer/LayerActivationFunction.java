package at.fhtw.ai.nn.activation.layer;

import at.fhtw.ai.nn.Layer;
import at.fhtw.ai.nn.activation.ActivationFunction;

/**
 * Layer activation functions are activation functions which operate on a per layer perception level.
 * <p>
 * Created On: 14.05.2018
 *
 * @author Daniel Kleebinder
 * @since 0.0.1
 */
public interface LayerActivationFunction extends ActivationFunction {

    /**
     * Initializes the activation function for being used on the neuron based level.
     *
     * @param layer Layer to activate.
     */
    void initialize(Layer layer);
}
