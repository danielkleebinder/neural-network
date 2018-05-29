package at.fhtw.ai.nn.initialize;

import at.fhtw.ai.nn.NeuralNetwork;

import java.util.Random;

/**
 * A random initializer for weights in a neural network.
 * <p>
 * Created On: 30.04.2018
 *
 * @author Daniel Kleebinder
 * @since 0.0.1
 */
public class RandomInitializer implements Initializer {
    private static final long serialVersionUID = 4534833561802573520L;

    /**
     * Random object for initialization.
     */
    private Random rnd;

    /**
     * Creates a new random initializer with the current system time in milliseconds as seed.
     */
    public RandomInitializer() {
        this(System.currentTimeMillis());
    }

    /**
     * Creates a new random initializer with the given seed.
     *
     * @param seed Seed.
     */
    public RandomInitializer(long seed) {
        rnd = new Random(seed);
    }

    @Override
    public void initialize(NeuralNetwork neuralNetwork) {
        neuralNetwork.getNeurons().stream().forEach(neuron -> {
            neuron.bias.value = randomBias();
            neuron.bias.weight = randomWeight();
            neuron.value = 0.0;
        });
        neuralNetwork.getSynapses().stream().forEach(synapse -> {
            synapse.weight = randomWeight();
            synapse.change = 0.0;
        });
    }

    /**
     * Returns a random weight.
     *
     * @return Random weight.
     */
    private double randomWeight() {
        double result = rnd.nextDouble() * 0.8 + 0.1;
        return rnd.nextBoolean() ? -result : result;
    }

    /**
     * Returns a random bias value (either -1 or 1).
     *
     * @return Random bias.
     */
    private double randomBias() {
        return rnd.nextBoolean() ? -1.0 : 1.0;
    }
}
