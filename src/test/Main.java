package test;

import at.fhtw.ai.nn.NeuralNetwork;
import at.fhtw.ai.nn.Neuron;
import at.fhtw.ai.nn.Synapse;
import at.fhtw.ai.nn.utils.Utils;

import java.io.IOException;

/**
 * Created On: 25.04.2018
 *
 * @author Daniel Kleebinder
 * @since 0.0.1
 */
public class Main {
    public static void main(String[] args) {
        System.out.println("Loading Neural Network...");
        NeuralNetwork neuralNetwork = null;
        try {
            neuralNetwork = Utils.deserialize("C:/Users/Daniel/Desktop/NeuralNetwork_M4.dat");
        } catch (IOException e) {
            e.printStackTrace();
        } catch (ClassNotFoundException e) {
            e.printStackTrace();
        }

        for (Neuron neuron : neuralNetwork.getInputLayer().getNeurons()) {
            for (Synapse synapse : neuron.getOutputSynapses()) {
                System.out.println(synapse.change + ", " + synapse.weight);
            }
        }
    }
}
