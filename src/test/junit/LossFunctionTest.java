package test.junit;

import at.fhtw.ai.nn.Neuron;
import at.fhtw.ai.nn.loss.CrossEntropy;
import at.fhtw.ai.nn.loss.LossFunction;
import at.fhtw.ai.nn.loss.Quadratic;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class LossFunctionTest {
    private static final Double DELTA = 0.00001;

    @Test
    void shouldComputeCorrectQuadraticLossFunctionValue() {
        LossFunction lossFunction = new Quadratic();
        assertEquals(0.0, lossFunction.compute(new Neuron.SimpleNeuron(0.7), 0.7), DELTA);
        assertEquals(1.0, lossFunction.compute(new Neuron.SimpleNeuron(1.0), 0.0), DELTA);
        assertEquals(-0.4, lossFunction.compute(new Neuron.SimpleNeuron(0.2), 0.6), DELTA);
    }

    @Test
    void shouldComputeCorrectCrossEntropyLossFunctionValue() {
        LossFunction lossFunction = new CrossEntropy();
        assertEquals(0.610864302, lossFunction.compute(new Neuron.SimpleNeuron(0.7), 0.7), DELTA);
        assertEquals(2.30258509, lossFunction.compute(new Neuron.SimpleNeuron(1.0), 0.1), DELTA);
        assertEquals(0.83519771, lossFunction.compute(new Neuron.SimpleNeuron(0.2), 0.6), DELTA);
    }
}