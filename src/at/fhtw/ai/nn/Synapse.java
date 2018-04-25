package at.fhtw.ai.nn;

import at.fhtw.ai.nn.utils.Utils;

import java.io.Serializable;

/**
 * A synapse is a connection between two neurons.
 * <p>
 * Created On: 24.04.2018
 *
 * @author Daniel Kleebinder
 * @since 0.0.1
 */
public class Synapse implements Serializable {

    /**
     * Synapse source neuron.
     */
    public Neuron sourceNeuron;

    /**
     * Synapse destination neuron.
     */
    public Neuron destinationNeuron;

    /**
     * Synapse connection weight.
     */
    public double weight;

    /**
     * Weight change.
     */
    public double change = 0.0;

    /**
     * Creates a new synapse.
     */
    public Synapse() {
        this(null);
    }

    /**
     * Creates a new synapse with the given source neuron.
     *
     * @param sourceNeuron Input neuron.
     */
    public Synapse(Neuron sourceNeuron) {
        this(sourceNeuron, null);
    }

    /**
     * Creates a new synapse with the given source neuron and weight.
     *
     * @param sourceNeuron Input neuron.
     * @param weight       Weight.
     */
    public Synapse(Neuron sourceNeuron, double weight) {
        this(sourceNeuron, null, weight);
    }

    /**
     * Creates a new synapse with the given neurons.
     *
     * @param sourceNeuron      Input neuron.
     * @param destinationNeuron Output neuron.
     */
    public Synapse(Neuron sourceNeuron, Neuron destinationNeuron) {
        this(sourceNeuron, destinationNeuron, Utils.randomWeight());
    }

    /**
     * Creates a new synapse with the given parameters.
     *
     * @param sourceNeuron      Input neuron.
     * @param destinationNeuron Output neuron.
     * @param weight            Weight.
     */
    public Synapse(Neuron sourceNeuron, Neuron destinationNeuron, double weight) {
        this.sourceNeuron = sourceNeuron;
        this.destinationNeuron = destinationNeuron;
        this.weight = weight;
    }

    /**
     * Sets the source neuron. This is the neuron which fires and transmits its information to the destination neuron.
     *
     * @param sourceNeuron Input neuron.
     */
    public void setSourceNeuron(Neuron sourceNeuron) {
        this.sourceNeuron = sourceNeuron;
    }

    /**
     * Returns the source neuron.
     *
     * @return Input neuron.
     */
    public Neuron getSourceNeuron() {
        return sourceNeuron;
    }


    /**
     * Sets the destination neuron. This is the neuron which will receive the information from the source neuron.
     *
     * @param output Output neuron.
     */
    public void setDestinationNeuron(Neuron output) {
        this.destinationNeuron = output;
    }

    /**
     * Returns the destination neuron.
     *
     * @return Output neuron.
     */
    public Neuron getDestinationNeuron() {
        return destinationNeuron;
    }


    /**
     * Sets the connection weight. This factor will be used when calculating a neurons value.
     *
     * @param weight Weight.
     */
    public void setWeight(double weight) {
        this.weight = weight;
    }

    /**
     * Returns the connection weight.
     *
     * @return Weight.
     */
    public double getWeight() {
        return weight;
    }


    /**
     * Sets the weight change. This is mostly used by advanced learning algorithms.
     *
     * @param change Weight change.
     */
    public void setChange(double change) {
        this.change = change;
    }

    /**
     * Returns the weight change.
     *
     * @return Weight change.
     */
    public double getChange() {
        return change;
    }

    /**
     * Computes the synapse destination. This is the source neurons value times the synapses weight.
     *
     * @return Synapse destination.
     */
    public double computeOutput() {
        return getSourceNeuron().fire(false) * weight;
    }
}