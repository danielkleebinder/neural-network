package at.fhtw.ai.nn;

import at.fhtw.ai.nn.activation.ActivationFunction;
import at.fhtw.ai.nn.activation.Identity;
import at.fhtw.ai.nn.utils.AtomicDouble;
import at.fhtw.ai.nn.utils.Utils;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

/**
 * An artificial neuron with activation function, bias and input connections.
 * <p>
 * Created On: 24.04.2018
 *
 * @author Daniel Kleebinder
 * @since 0.0.1
 */
public class Neuron implements Serializable {
    private static final long serialVersionUID = 2743680081104305613L;

    /**
     * Contains all neuron input synapses.
     */
    private List<Synapse> inputSynapses = new ArrayList<>(16);
    private List<Synapse> outputSynapses = new ArrayList<>(16);

    /**
     * Activation function for the neuron.
     */
    private ActivationFunction activationFunction = new Identity();

    /**
     * Neuron activation bias.
     */
    public Bias bias = new Bias();

    /**
     * Latest computed value.
     */
    public double value = Double.NaN;
    /**
     * Latest error value.
     */
    public double errorValue = 0.0;

    /**
     * If the neuron has fired yet.
     */
    private boolean fired = false;


    /**
     * Sets the activation function for this neuron.
     *
     * @param activationFunction Activation function.
     */
    public void setActivationFunction(ActivationFunction activationFunction) {
        this.activationFunction = activationFunction;
    }

    /**
     * Returns the activation function for this neuron.
     *
     * @return Activation function.
     */
    public ActivationFunction getActivationFunction() {
        return activationFunction;
    }

    /**
     * Sets the neurons bias. This value will be added to the result of the synapses calculation.
     *
     * @param bias Bias.
     */
    public void setBias(Bias bias) {
        this.bias = bias;
    }

    /**
     * Returns the neurons bias.
     *
     * @return Bias.
     */
    public Bias getBias() {
        return bias;
    }

    /**
     * Sets all input synapses at once.
     *
     * @param inputSynapses Input synapses.
     */
    public void setInputSynapses(List<Synapse> inputSynapses) {
        this.inputSynapses = inputSynapses;
    }

    /**
     * Returns all input synapses.
     *
     * @return Input synapses.
     */
    public List<Synapse> getInputSynapses() {
        return inputSynapses;
    }

    /**
     * Sets the output synapses.
     *
     * @param outputSynapses Output synapses.
     */
    public void setOutputSynapses(List<Synapse> outputSynapses) {
        this.outputSynapses = outputSynapses;
    }

    /**
     * Returns the output synapses.
     *
     * @return Output synapses.
     */
    public List<Synapse> getOutputSynapses() {
        return outputSynapses;
    }


    /**
     * Connects the given neuron as input to this neuron with synapse weight 1.
     *
     * @param neuron Input neuron.
     */
    public void connectInput(Neuron neuron) {
        connectInput(neuron, Utils.randomWeight());
    }

    /**
     * Connects the given neuron as input to this neuron.
     *
     * @param neuron Input neuron.
     * @param weight Weight.
     */
    public void connectInput(Neuron neuron, double weight) {
        connectInput(new Synapse(neuron, this, weight));
    }

    /**
     * Connects the given synapse as input synapse.
     *
     * @param synapse Input synapse.
     */
    public void connectInput(Synapse synapse) {
        if (synapse.getDestinationNeuron() == null) {
            synapse.setDestinationNeuron(this);
        }
        inputSynapses.add(synapse);
        synapse.getSourceNeuron().getOutputSynapses().add(synapse);
    }

    /**
     * Sets the nodes current value. This method is mainly used for neurons in the input layer.
     *
     * @param value Value.
     */
    public void setValue(double value) {
        this.value = value;
    }

    /**
     * Returns the latest computed value of the neuron. This will be <code>NaN</code> if the neuron has not fired yet.
     *
     * @return Value.
     */
    public double getValue() {
        return value;
    }

    /**
     * Sets the error value.
     *
     * @param errorValue Error value.
     */
    public void setErrorValue(double errorValue) {
        this.errorValue = errorValue;
    }

    /**
     * Returns the latest computed error value of the neuron. This will be <code>NaN</code> if the neuron has not
     * computed the error value yet.
     *
     * @return Error value.
     */
    public double getErrorValue() {
        return errorValue;
    }

    /**
     * Sets if the neuron was fired yet.
     *
     * @param fired True if fired, otherwise false.
     */
    public void setFired(boolean fired) {
        this.fired = fired;
    }

    /**
     * Returns if the neuron was fired yet.
     *
     * @return True if fired, otherwise false.
     */
    public boolean isFired() {
        return fired;
    }


    /**
     * Fires this neuron and calculates the output value.
     *
     * @param parallel True if the firing should be done in multiple threads.
     * @return Neuron output.
     */
    public double fire(boolean parallel) {
        int dataSize = inputSynapses.size();

        if (fired || dataSize <= 0) {
            return value;
        }

        final AtomicDouble currentValue = new AtomicDouble(0.0);
        if (parallel) {
            inputSynapses
                    .parallelStream()
                    .forEach(synapse -> currentValue.value += synapse.computeOutput());
        } else {
            for (int i = 0; i < dataSize; i++) {
                currentValue.value += inputSynapses.get(i).computeOutput();
            }
        }
        currentValue.value += bias.compute();
        currentValue.value = activationFunction.activate(currentValue.value);

        fired = true;

        // Return the final result
        // This is incredibly important for multithreading
        value = currentValue.value;
        return value;
    }
}