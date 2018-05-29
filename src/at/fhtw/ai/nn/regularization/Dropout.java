package at.fhtw.ai.nn.regularization;

import at.fhtw.ai.nn.Neuron;
import at.fhtw.ai.nn.Synapse;

/**
 * The dropout regularization technique is a method to prevent over fitting. Neurons are partially ignored during the
 * learning process given a specific probability.
 * <p>
 * Created On: 29.05.2018
 *
 * @author Daniel Kleebinder
 * @since 0.0.1
 */
public class Dropout extends AbstractRegularization {
    private static final long serialVersionUID = -4305114487432226653L;

    /**
     * Creates a new dropout regularization with standard lambda dropout probability of 20%.
     */
    public Dropout() {
        this(0.2);
    }

    /**
     * Creates a new dropout regularization with the given lambda dropout probability.
     *
     * @param lambda Lambda probability.
     */
    public Dropout(double lambda) {
        this.lambda = lambda;
    }

    @Override
    public double compute(Synapse synapse) {
        return 0.0;
    }

    @Override
    public double compute(Neuron neuron) {
        return Math.random() < lambda ? -1.0 : 1.0;
    }
}
