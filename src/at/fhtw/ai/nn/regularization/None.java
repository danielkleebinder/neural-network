package at.fhtw.ai.nn.regularization;

import at.fhtw.ai.nn.Neuron;
import at.fhtw.ai.nn.Synapse;

/**
 * Applies no regularization at all.
 * <p>
 * Created On: 29.05.2018
 *
 * @author Daniel Kleebinder
 * @since 0.0.1
 */
public class None extends AbstractRegularization {

    @Override
    public double compute(Synapse synapse) {
        return 0.0;
    }

    @Override
    public double compute(Neuron neuron) {
        return 1.0;
    }
}
