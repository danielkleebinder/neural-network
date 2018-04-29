package test.junit;

import at.fhtw.ai.nn.Layer;
import at.fhtw.ai.nn.NeuralNetwork;
import at.fhtw.ai.nn.activation.Sigmoid;
import at.fhtw.ai.nn.learning.BackPropagation;
import at.fhtw.ai.nn.utils.Utils;
import org.junit.jupiter.api.Test;

import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.*;

class NeuralNetworkTest {
    private static final Double DELTA = 0.00001;
    private static final Double DELTA_2 = 0.01;


    @Test
    void shouldHaveCorrectNumberOfNeurons() {
        NeuralNetwork neuralNetwork = createSimpleNeuralNetwork();

        AtomicInteger numberOfNeurons = new AtomicInteger(0);
        neuralNetwork.getLayers().forEach(layer -> numberOfNeurons.set(numberOfNeurons.get() + layer.getNeurons().size()));

        assertEquals(6, numberOfNeurons.get());
        assertEquals(2, neuralNetwork.getInputLayer().getNeurons().size());
        assertEquals(3, neuralNetwork.getLayers().get(1).getNeurons().size());
        assertEquals(1, neuralNetwork.getOutputLayer().getNeurons().size());
    }

    @Test
    void shouldHaveCorrectNumberOfSynapses() {
        NeuralNetwork neuralNetwork = createSimpleNeuralNetwork();

        assertEquals(3, neuralNetwork.getInputLayer().getNeurons().get(0).getOutputSynapses().size());
        assertEquals(3, neuralNetwork.getInputLayer().getNeurons().get(1).getOutputSynapses().size());

        assertEquals(0, neuralNetwork.getInputLayer().getNeurons().get(0).getInputSynapses().size());
        assertEquals(0, neuralNetwork.getInputLayer().getNeurons().get(1).getInputSynapses().size());

        assertEquals(1, neuralNetwork.getLayers().get(1).getNeurons().get(0).getOutputSynapses().size());
        assertEquals(1, neuralNetwork.getLayers().get(1).getNeurons().get(1).getOutputSynapses().size());
        assertEquals(1, neuralNetwork.getLayers().get(1).getNeurons().get(2).getOutputSynapses().size());

        assertEquals(2, neuralNetwork.getLayers().get(1).getNeurons().get(0).getInputSynapses().size());
        assertEquals(2, neuralNetwork.getLayers().get(1).getNeurons().get(1).getInputSynapses().size());
        assertEquals(2, neuralNetwork.getLayers().get(1).getNeurons().get(2).getInputSynapses().size());

        assertEquals(0, neuralNetwork.getOutputLayer().getNeurons().get(0).getOutputSynapses().size());
        assertEquals(3, neuralNetwork.getOutputLayer().getNeurons().get(0).getInputSynapses().size());
    }

    @Test
    void shouldHaveCorrectInputLayerValues() {
        NeuralNetwork neuralNetwork = createSimpleNeuralNetwork();

        assertEquals(1.0, neuralNetwork.getInputLayer().getNeurons().get(0).value, DELTA_2);
        assertEquals(1.0, neuralNetwork.getInputLayer().getNeurons().get(0).value, DELTA_2);
    }

    @Test
    void shouldHaveCorrectHiddenLayerValues() {
        NeuralNetwork neuralNetwork = createSimpleNeuralNetwork();
        neuralNetwork.fireOutput();

        assertEquals(0.73, neuralNetwork.getLayers().get(1).getNeurons().get(0).value, DELTA_2);
        assertEquals(0.79, neuralNetwork.getLayers().get(1).getNeurons().get(1).value, DELTA_2);
        assertEquals(0.69, neuralNetwork.getLayers().get(1).getNeurons().get(2).value, DELTA_2);
    }


    @Test
    void shouldHaveCorrectOutputLayerValues() {
        NeuralNetwork neuralNetwork = createSimpleNeuralNetwork();
        neuralNetwork.fireOutput();

        assertEquals(0.77, neuralNetwork.getOutputLayer().getNeurons().get(0).value, DELTA_2);
    }


    @Test
    void shouldHaveCorrectNetworkError() {
        NeuralNetwork neuralNetwork = createSimpleNeuralNetwork();
        BackPropagation backPropagation = createSimpleBackPropagation(neuralNetwork, 0.0);
        backPropagation.learn();

        assertEquals(0.5929, backPropagation.networkError(), DELTA_2);
    }

    @Test
    void shouldHaveCorrectErrorOnOutputLayerNeurons() {
        NeuralNetwork neuralNetwork = createSimpleNeuralNetwork();
        BackPropagation backPropagation = createSimpleBackPropagation(neuralNetwork, 0.0);
        backPropagation.learn();

        assertEquals(-0.1344, neuralNetwork.getOutputLayer().getNeurons().get(0).errorValue, DELTA_2);
    }

    @Test
    void shouldHaveCorrectErrorOnHiddenLayerNeurons() {
        NeuralNetwork neuralNetwork = createSimpleNeuralNetwork();
        BackPropagation backPropagation = createSimpleBackPropagation(neuralNetwork, 0.0);
        backPropagation.learn();

        assertEquals(-0.088, neuralNetwork.getLayers().get(1).getNeurons().get(0).errorValue, DELTA_2);
        assertEquals(-0.045, neuralNetwork.getLayers().get(1).getNeurons().get(1).errorValue, DELTA_2);
        assertEquals(-0.032, neuralNetwork.getLayers().get(1).getNeurons().get(2).errorValue, DELTA_2);
    }

    @Test
    void shouldBackPropagateOutputWeightsCorrect() {
        NeuralNetwork neuralNetwork = createSimpleNeuralNetwork();
        BackPropagation backPropagation = createSimpleBackPropagation(neuralNetwork, 0.0);
        backPropagation.learn();

        assertEquals(0.116, neuralNetwork.getOutputLayer().getNeurons().get(0).getInputSynapses().get(0).weight, DELTA_2);
        assertEquals(0.329, neuralNetwork.getOutputLayer().getNeurons().get(0).getInputSynapses().get(1).weight, DELTA_2);
        assertEquals(0.708, neuralNetwork.getOutputLayer().getNeurons().get(0).getInputSynapses().get(2).weight, DELTA_2);
    }

    @Test
    void shouldBackPropagateHiddenWeightsCorrect() {
        NeuralNetwork neuralNetwork = createSimpleNeuralNetwork();
        BackPropagation backPropagation = createSimpleBackPropagation(neuralNetwork, 0.0);
        backPropagation.learn();

        assertEquals(0.712, neuralNetwork.getInputLayer().getNeurons().get(0).getOutputSynapses().get(0).weight, DELTA_2);
        assertEquals(0.355, neuralNetwork.getInputLayer().getNeurons().get(0).getOutputSynapses().get(1).weight, DELTA_2);
        assertEquals(0.268, neuralNetwork.getInputLayer().getNeurons().get(0).getOutputSynapses().get(2).weight, DELTA_2);

        assertEquals(0.112, neuralNetwork.getInputLayer().getNeurons().get(1).getOutputSynapses().get(0).weight, DELTA_2);
        assertEquals(0.855, neuralNetwork.getInputLayer().getNeurons().get(1).getOutputSynapses().get(1).weight, DELTA_2);
        assertEquals(0.468, neuralNetwork.getInputLayer().getNeurons().get(1).getOutputSynapses().get(2).weight, DELTA_2);
    }

    @Test
    void shouldHaveCorrectHiddenLayerValuesAfterBackPropagation() {
        NeuralNetwork neuralNetwork = createSimpleNeuralNetwork();
        BackPropagation backPropagation = createSimpleBackPropagation(neuralNetwork, 0.0);
        backPropagation.learn();
        neuralNetwork.fireOutput();

        assertEquals(0.69, neuralNetwork.getLayers().get(1).getNeurons().get(0).value, DELTA_2);
        assertEquals(0.77, neuralNetwork.getLayers().get(1).getNeurons().get(1).value, DELTA_2);
        assertEquals(0.68, neuralNetwork.getLayers().get(1).getNeurons().get(2).value, DELTA_2);
    }

    @Test
    void shouldHaveCorrectOutputLayerValuesAfterBackPropagation() {
        NeuralNetwork neuralNetwork = createSimpleNeuralNetwork();
        BackPropagation backPropagation = createSimpleBackPropagation(neuralNetwork, 0.0);
        backPropagation.learn();
        neuralNetwork.fireOutput();

        assertEquals(0.69, neuralNetwork.getOutputLayer().getNeurons().get(0).value, DELTA_2);
    }

    private void modifyNeuralNetworkForXORTutorialExample(NeuralNetwork neuralNetwork) {
        neuralNetwork.getInputLayer().getNeurons().get(0).setValue(1.0);
        neuralNetwork.getInputLayer().getNeurons().get(1).setValue(1.0);
        neuralNetwork.getOutputLayer().getNeurons().get(0).setValue(0.0);

        neuralNetwork.getLayers().forEach(
                layer -> layer.getNeurons().forEach(
                        neuron -> neuron.getBias().value = 0.0
                )
        );

        neuralNetwork.getInputLayer().getNeurons().get(0).getOutputSynapses().get(0).weight = 0.8;
        neuralNetwork.getInputLayer().getNeurons().get(0).getOutputSynapses().get(1).weight = 0.4;
        neuralNetwork.getInputLayer().getNeurons().get(0).getOutputSynapses().get(2).weight = 0.3;

        neuralNetwork.getInputLayer().getNeurons().get(1).getOutputSynapses().get(0).weight = 0.2;
        neuralNetwork.getInputLayer().getNeurons().get(1).getOutputSynapses().get(1).weight = 0.9;
        neuralNetwork.getInputLayer().getNeurons().get(1).getOutputSynapses().get(2).weight = 0.5;

        neuralNetwork.getLayers().get(1).getNeurons().get(0).getOutputSynapses().get(0).weight = 0.3;
        neuralNetwork.getLayers().get(1).getNeurons().get(1).getOutputSynapses().get(0).weight = 0.5;
        neuralNetwork.getLayers().get(1).getNeurons().get(2).getOutputSynapses().get(0).weight = 0.9;
    }

    private NeuralNetwork createSimpleNeuralNetwork() {
        Layer inputLayer = Utils.createLayer("Input Layer", 2);
        Layer hiddenLayer = Utils.createLayer("Hidden Layer", 3);
        Layer outputLayer = Utils.createLayer("Output Layer", 1);

        NeuralNetwork neuralNetwork = new NeuralNetwork();
        neuralNetwork.getLayers().add(inputLayer);
        neuralNetwork.getLayers().add(hiddenLayer);
        neuralNetwork.getLayers().add(outputLayer);
        neuralNetwork.setActivationFunctions(new Sigmoid());
        neuralNetwork.connectLayersInOrder();

        modifyNeuralNetworkForXORTutorialExample(neuralNetwork);

        return neuralNetwork;
    }

    private BackPropagation createSimpleBackPropagation(NeuralNetwork neuralNetwork, double desiredValue) {
        BackPropagation backPropagation = new BackPropagation();
        backPropagation.setLearningRate(1.0);
        backPropagation.setMeanSquareError(0.0001);
        backPropagation.setMomentum(0.0);
        backPropagation.getDesiredOutputValues().clear();
        backPropagation.getDesiredOutputValues().add(desiredValue);
        backPropagation.setNeuralNetwork(neuralNetwork);
        return backPropagation;
    }
}