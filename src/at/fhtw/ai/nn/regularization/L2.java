package at.fhtw.ai.nn.regularization;

import at.fhtw.ai.nn.NeuralNetwork;
import at.fhtw.ai.nn.utils.AtomicDouble;

/**
 * The L2 regularization technique is an often used squared ridge regression method for adjusting weights in the neural
 * network to prevent over fitting.<br>
 * <code>C = (lambda / 2) * sum(w^2)</code>
 * <p>
 * Created On: 01.05.2018
 *
 * @author Daniel Kleebinder
 * @since 0.0.1
 */
public class L2 extends AbstractRegularization {
    private static final long serialVersionUID = -8635354798349352556L;

    @Override
    public double compute(NeuralNetwork neuralNetwork) {
        AtomicDouble sumWeights = new AtomicDouble(0.0);
        neuralNetwork.getSynapses().stream().forEach(synapse -> sumWeights.value += (synapse.weight * synapse.weight));
        return sumWeights.value * (lambda / 2.0);
    }
}