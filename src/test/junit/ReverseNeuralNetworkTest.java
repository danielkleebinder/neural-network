package test.junit;

import at.fhtw.ai.nn.NeuralNetwork;
import at.fhtw.ai.nn.Synapse;
import at.fhtw.ai.nn.activation.Logit;
import at.fhtw.ai.nn.initialize.RandomInitializer;
import at.fhtw.ai.nn.utils.NeuralNetworkBuilder;
import org.junit.jupiter.api.Test;

class ReverseNeuralNetworkTest {
    private static final Double DELTA = 0.0001;

    @Test
    void shouldReverseInputNeuronCorrect() {
        NeuralNetwork reversedNeuralNetwork = createSimpleReversedNeuralNetwork();

        // 0.154596
        // 0.025884375
        // 0.00753

        double value = new Logit().activate(0.2726) - 0.99;
        double sum = 0.0;
        for (Synapse synapse : reversedNeuralNetwork.getInputLayer().getNeurons().get(0).getOutputSynapses()) {
            sum += synapse.weight;
        }
        double x = value / sum;
        System.out.println();
    }

    @Test
    void shouldReverseHiddenNeuronsCorrect() {
        NeuralNetwork reversedNeuralNetwork = createSimpleReversedNeuralNetwork();
        reversedNeuralNetwork.fireOutput();
        System.out.println(reversedNeuralNetwork.getOutputLayer().getNeurons().get(0).value);
        System.out.println(reversedNeuralNetwork.getOutputLayer().getNeurons().get(0).value);
    }

    private void modifyReverseNeuralNetworkForXORPaperExample(NeuralNetwork neuralNetwork) {
        neuralNetwork.getInputLayer().getNeurons().get(0).setValue(0.2726);
        neuralNetwork.getOutputLayer().getNeurons().get(0).setValue(0.0);
        neuralNetwork.getOutputLayer().getNeurons().get(1).setValue(0.0);

        neuralNetwork.getLayers().forEach(
                layer -> layer.getNeurons().forEach(
                        neuron -> neuron.getBias().value = 1.0
                )
        );

        neuralNetwork.getInputLayer().getNeurons().get(0).getOutputSynapses().get(0).weight = 0.71;
        neuralNetwork.getInputLayer().getNeurons().get(0).getOutputSynapses().get(1).weight = -0.39;
        neuralNetwork.getInputLayer().getNeurons().get(0).getBias().value = 1.0;
        neuralNetwork.getInputLayer().getNeurons().get(0).getBias().weight = -0.99;

        neuralNetwork.getLayers().get(1).getNeurons().get(0).getOutputSynapses().get(0).weight = -0.95;
        neuralNetwork.getLayers().get(1).getNeurons().get(0).getOutputSynapses().get(1).weight = 0.83;
        neuralNetwork.getLayers().get(1).getNeurons().get(0).getBias().value = 1.0;
        neuralNetwork.getLayers().get(1).getNeurons().get(0).getBias().weight = 0.09;

        neuralNetwork.getLayers().get(1).getNeurons().get(1).getOutputSynapses().get(0).weight = 0.45;
        neuralNetwork.getLayers().get(1).getNeurons().get(1).getOutputSynapses().get(1).weight = -0.88;
        neuralNetwork.getLayers().get(1).getNeurons().get(1).getBias().value = -1.0;
        neuralNetwork.getLayers().get(1).getNeurons().get(1).getBias().weight = 0.37;
    }

    private NeuralNetwork createSimpleReversedNeuralNetwork() {
        NeuralNetwork neuralNetwork = new NeuralNetworkBuilder()
                .layer("Input Layer", 1)
                .layer("Hidden Layer", 2)
                .layer("Output Layer", 2)
                .activationFunction(new Logit())
                .initializer(new RandomInitializer())
                .build();

        modifyReverseNeuralNetworkForXORPaperExample(neuralNetwork);

        return neuralNetwork;
    }
}