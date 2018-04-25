package at.fhtw.ai.nn;

import at.fhtw.ai.nn.activation.ActivationFunction;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

/**
 * The neural network is an artificial replica of a biological brain. It uses layers, neurons and synapses to compute and predict certain values.
 * <p>
 * Created On: 24.04.2018
 *
 * @author Daniel Kleebinder
 * @since 0.0.1
 */
public class NeuralNetwork implements Serializable {

    /**
     * Contains all layers of the neural network.
     */
    private List<Layer> layers = new ArrayList<>(8);


    /**
     * Returns the input layer. This is the first layer in the list of layers in the network.
     *
     * @return Input layer.
     */
    public Layer getInputLayer() {
        if (layers.isEmpty()) {
            return null;
        }
        return layers.get(0);
    }

    /**
     * Returns the output layer. This is the last layer in the list of layers in the network.
     *
     * @return Output layer.
     */
    public Layer getOutputLayer() {
        if (layers.isEmpty()) {
            return null;
        }
        return layers.get(layers.size() - 1);
    }

    /**
     * Sets all the network layers.
     *
     * @param layers Layers.
     */
    public void setLayers(List<Layer> layers) {
        this.layers = layers;
    }

    /**
     * Returns all the network layers.
     *
     * @return Layers
     */
    public List<Layer> getLayers() {
        return layers;
    }


    /**
     * Sets the activation function for all neurons in all layers.
     *
     * @param activationFunction New global activation function.
     */
    public void setActivationFunctions(ActivationFunction activationFunction) {
        layers.forEach(layer -> layer.setActivationFunctions(activationFunction));
    }

    /**
     * Connects all layers in a standard order.
     */
    public void connectLayersInOrder() {
        if (layers.size() <= 1) {
            return;
        }

        Layer previousLayer = layers.get(0);
        for (int i = 1; i < layers.size(); i++) {
            Layer currentLayer = layers.get(i);
            currentLayer.connectInput(previousLayer);
            previousLayer = currentLayer;
        }
    }

    /**
     * Fires all neurons in all layers at once.
     */
    public void fire() {
        layers.stream().forEach(layer -> layer.fire());
    }

    /**
     * Fires only the output layer neurons.
     */
    public void fireOutput() {
        getOutputLayer().fireParallel();
    }
}
