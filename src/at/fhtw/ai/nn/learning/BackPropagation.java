package at.fhtw.ai.nn.learning;

import at.fhtw.ai.nn.Bias;
import at.fhtw.ai.nn.Layer;
import at.fhtw.ai.nn.Neuron;
import at.fhtw.ai.nn.Synapse;
import at.fhtw.ai.nn.loss.LossFunction;
import at.fhtw.ai.nn.loss.Quadratic;
import at.fhtw.ai.nn.regularization.L2;
import at.fhtw.ai.nn.regularization.Regularization;

/**
 * Backpropagation is a special case of an older and more general technique called automatic differentiation. In the
 * context of learning, backpropagation is commonly used by the gradient descent optimization algorithm to adjust the
 * weight of neurons by calculating the gradient of the loss function. This technique is also sometimes called backward
 * propagation of errors, because the error is calculated at the output and distributed back through the network layers.
 * <p>
 * Created On: 24.04.2018
 *
 * @author Daniel Kleebinder
 * @since 0.0.1
 */
public class BackPropagation extends LearningAlgorithm {

    // Network back propagation configuration
    private double momentum = 0.9;
    private double learningRate = 0.2;
    private double meanSquareError = 0.05;

    private Regularization regularization = new L2();
    private LossFunction lossFunction = new Quadratic();

    /**
     * Computes the errors for the last aka output layer.
     */
    private void computeOutputLayerErrors() {
        Layer outputLayer = neuralNetwork.getOutputLayer();

        // Illegal value bag size
        if (desiredOutputValues.size() != outputLayer.getNeurons().size()) {
            throw new IllegalStateException("Number of desired values has to be the same as the number of output neurons in the output layer");
        }

        // Compute error for each neuron, error values are now stored in the neurons themselves
        for (int i = 0; i < outputLayer.getNeurons().size(); i++) {
            Neuron neuron = outputLayer.getNeurons().get(i);
            neuron.errorValue = computeError(neuron, desiredOutputValues.get(i));
        }
    }

    /**
     * Computes the errors for all hidden layers.
     */
    private void computeHiddenLayerErrors() {
        // Skip last layer. Output layer errors were already computed!
        for (int i = neuralNetwork.getLayers().size() - 2; i >= 1; i--) {
            Layer currentLayer = neuralNetwork.getLayers().get(i);
            for (Neuron neuron : currentLayer.getNeurons()) {
                double sum = 0.0;
                for (Synapse outputSynapse : neuron.getOutputSynapses()) {
                    sum += outputSynapse.destinationNeuron.errorValue * outputSynapse.weight;
                }
                neuron.errorValue = sum * neuron.getActivationFunction().derivative(neuron);
            }
        }
    }

    /**
     * Tries to adjust all weights in the network as good as possible to better match the desired output values.
     */
    private void adjustLayerWeights() {
        double re, dw, dm;
        for (int i = neuralNetwork.getLayers().size() - 1; i >= 0; i--) {
            Layer currentLayer = neuralNetwork.getLayers().get(i);
            for (Neuron neuron : currentLayer.getNeurons()) {
                if (regularization != null && regularization.compute(neuron) <= 0.0) {
                    continue;
                }
                for (Synapse outputSynapse : neuron.getOutputSynapses()) {
                    re = learningRate * (regularization == null ? 0.0 : regularization.compute(outputSynapse));
                    dw = learningRate * ((outputSynapse.destinationNeuron.errorValue + re) * neuron.value);
                    dm = momentum * outputSynapse.change;

                    outputSynapse.weight += (dw + dm);
                    outputSynapse.change = dw;
                }
                Bias bias = neuron.getBias();
                bias.weight += learningRate * neuron.errorValue * bias.getValue();
            }
        }
    }

    /**
     * Resets backpropagation network parameters.
     */
    private void resetNetwork() {
        for (Layer layer : neuralNetwork.getLayers()) {
            for (Neuron neuron : layer.getNeurons()) {
                neuron.errorValue = 0.0;
                for (Synapse synapse : neuron.getInputSynapses()) {
                    synapse.change = 0.0;
                }
            }
        }
    }

    private void feedForward() {
        neuralNetwork.fireOutput();
    }

    @Override
    public boolean learn() {
        feedForward();
        resetNetwork();
        computeOutputLayerErrors();
        computeHiddenLayerErrors();
        adjustLayerWeights();
        return true;
    }

    /**
     * Computes the error of the given neuron. Uses the standard neuron error function:<br>
     * <code>err = (d - c) * c * (1 - c)</code>
     * <ul>
     * <li><b>err</b> ... Error</li>
     * <li><b>d</b> ... Desired Value</li>
     * <li><b>c</b> ... Computed Value</li>
     * </ul>
     *
     * @param neuron        Neuron.
     * @param expectedValue Expected value (often referred to as "desired value").
     * @return Neuron error.
     */
    private double computeError(Neuron neuron, double expectedValue) {
        if (neuron.getInputSynapses().size() <= 0) {
            return 0.0;
        }
        return lossFunction.compute(neuron, expectedValue);
    }

    /**
     * Computes the total network error.
     *
     * @return Network error.
     */
    public double networkError() {
        Layer outputLayer = neuralNetwork.getOutputLayer();
        double result = 0.0;
        for (int i = 0; i < outputLayer.getNeurons().size(); i++) {
            double neuronError = outputLayer.getNeurons().get(i).value - desiredOutputValues.get(i);
            result += (neuronError * neuronError);
        }
        result /= (double) outputLayer.getNeurons().size();
        return result;
    }

    /**
     * Sets the momentum. The momentum value is needed to ignore a local minimum in the network function. The default
     * value is typically at <code>0.9</code>.
     *
     * @param momentum Momentum.
     */
    public void setMomentum(double momentum) {
        this.momentum = momentum;
    }

    /**
     * Returns the momentum.
     *
     * @return Momentum.
     */
    public double getMomentum() {
        return momentum;
    }

    /**
     * Sets the learning rate. A high learning rate increases the speed of the learning process, but propagates
     * oscillation and reduces the accuracy. A low learning rate dramatically increases the accuracy of the network, but
     * also decreases the speed of learning. The default value is typically at <code>0.2</code>.
     *
     * @param learningRate Learning rate.
     */
    public void setLearningRate(double learningRate) {
        this.learningRate = learningRate;
    }

    /**
     * Returns the learning rate.
     *
     * @return Learning rate.
     */
    public double getLearningRate() {
        return learningRate;
    }

    /**
     * Sets the mean square error. This value is used during the learning phase to indicate the minimum error rate. The
     * default value is typically at <code>0.05</code>.
     *
     * @param meanSquareError Mean square error.
     */
    public void setMeanSquareError(double meanSquareError) {
        this.meanSquareError = meanSquareError;
    }

    /**
     * Returns the mean square error.
     *
     * @return Mean square error.
     */
    public double getMeanSquareError() {
        return meanSquareError;
    }

    /**
     * Sets the regularization algorithm. The standard algorithm is L2 regularization.
     *
     * @param regularization Regularization algorithm.
     */
    public void setRegularization(Regularization regularization) {
        this.regularization = regularization;
    }

    /**
     * Returns the regularization algorithm.
     *
     * @return Regularization algorithm.
     */
    public Regularization getRegularization() {
        return regularization;
    }

    /**
     * Sets the loss function for the back propagation.
     *
     * @param lossFunction Loss function.
     */
    public void setLossFunction(LossFunction lossFunction) {
        this.lossFunction = lossFunction;
    }

    /**
     * Returns the loss function.
     *
     * @return Loss function.
     */
    public LossFunction getLossFunction() {
        return lossFunction;
    }
}