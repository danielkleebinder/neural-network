package at.fhtw.ai.nn.regularization;

import at.fhtw.ai.nn.Neuron;
import at.fhtw.ai.nn.Synapse;

import java.io.Serializable;

/**
 * The regularization is used to prevent neural networks from over fitting by mostly adjusting the weight delta updates.
 * <p>
 * Created On: 01.05.2018
 *
 * @author Daniel Kleebinder
 * @since 0.0.1
 */
public interface Regularization extends Serializable {

    /**
     * Computes the delta update for the regularization of neural network weights.
     *
     * @param synapse Synapse.
     * @return Regularization term.
     */
    double compute(Synapse synapse);

    /**
     * Computes the delta update for the given neuron.
     *
     * @param neuron Neuron.
     * @return Regularization term.
     */
    double compute(Neuron neuron);

    /**
     * Sets the lambda term for the regularization method.
     *
     * @param lambda Lambda.
     */
    void setLambda(double lambda);

    /**
     * Returns the lambda term.
     *
     * @return Lambda.
     */
    double getLambda();
}