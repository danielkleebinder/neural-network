package at.fhtw.ai.nn.initialize;

import at.fhtw.ai.nn.NeuralNetwork;

/**
 * Constant initializer are used to reproduce neural network states. Each weight is initialized with a given weight and
 * all biases will have the same constant value.
 * <p>
 * Created On: 30.04.2018
 *
 * @author Daniel Kleebinder
 * @since 0.0.1
 */
public class ConstantInitializer implements Initializer {
    private static final long serialVersionUID = 7281576927798244374L;

    /**
     * Initialization weight.
     */
    private double weight;

    /**
     * Creates a new constant initializer with weight <code>0.2</code>.
     */
    public ConstantInitializer() {
        this(0.2);
    }

    /**
     * Creates a new constant initializer with the given weight.
     *
     * @param weight Weight.
     */
    public ConstantInitializer(double weight) {
        this.weight = weight;
    }

    @Override
    public void initialize(NeuralNetwork neuralNetwork) {
        neuralNetwork.getNeurons().stream().forEach(neuron -> {
            neuron.bias.value = 1.0;
            neuron.bias.weight = weight;
            neuron.value = 0.0;
        });
        neuralNetwork.getSynapses().stream().forEach(synapse -> {
            synapse.weight = weight;
            synapse.change = 0.0;
        });
    }

    /**
     * Sets the constant weight for initialization.
     *
     * @param weight Weight.
     */
    public void setWeight(double weight) {
        this.weight = weight;
    }

    /**
     * Returns the constant weight.
     *
     * @return Weight.
     */
    public double getWeight() {
        return weight;
    }
}
