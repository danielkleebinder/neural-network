package test.junit;

import at.fhtw.ai.nn.Neuron;
import at.fhtw.ai.nn.activation.*;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class ActivationFunctionTest {
    private static final Double DELTA = 0.00001;

    @Test
    void shouldReturnSameValuesForArcTan() {
        ActivationFunction activationFunction = new ArcTan();
        assertEquals(0.0, activationFunction.activate(new Neuron.SimpleNeuron(0.0)), DELTA);
        assertEquals(0.785398, activationFunction.activate(new Neuron.SimpleNeuron(1.0)), DELTA);
        assertEquals(-0.785398, activationFunction.activate(new Neuron.SimpleNeuron(-1.0)), DELTA);
        assertEquals(-1.5707862, activationFunction.activate(new Neuron.SimpleNeuron(-98723.0)), DELTA);
        assertEquals(1.57079622, activationFunction.activate(new Neuron.SimpleNeuron(8975987.0)), DELTA);
    }

    @Test
    void shouldReturnSameValuesForArcTanDerivative() {
        ActivationFunction activationFunction = new ArcTan();
        assertEquals(1.0, activationFunction.derivative(new Neuron.SimpleNeuron(0.0)), DELTA);
        assertEquals(0.5, activationFunction.derivative(new Neuron.SimpleNeuron(1.0)), DELTA);
        assertEquals(0.5, activationFunction.derivative(new Neuron.SimpleNeuron(-1.0)), DELTA);
        assertEquals(0.0, activationFunction.derivative(new Neuron.SimpleNeuron(-98723.0)), DELTA);
        assertEquals(0.0, activationFunction.derivative(new Neuron.SimpleNeuron(876493.0)), DELTA);
    }

    @Test
    void shouldReturnCorrectValuesForBinaryStep() {
        ActivationFunction activationFunction = new BinaryStep();
        assertEquals(1.0, activationFunction.activate(new Neuron.SimpleNeuron(100.0)), DELTA);
        assertEquals(0.0, activationFunction.activate(new Neuron.SimpleNeuron(-200.0)), DELTA);
        assertEquals(1.0, activationFunction.activate(new Neuron.SimpleNeuron(0.5)), DELTA);
        assertEquals(1.0, activationFunction.activate(new Neuron.SimpleNeuron(0.0)), DELTA);
    }

    @Test
    void shouldReturnCorrectValuesForBinaryStepDerivative() {
        ActivationFunction activationFunction = new BinaryStep();
        assertEquals(0.0, activationFunction.derivative(new Neuron.SimpleNeuron(100.0)), DELTA);
        assertEquals(0.0, activationFunction.derivative(new Neuron.SimpleNeuron(-200.0)), DELTA);
        assertEquals(0.0, activationFunction.derivative(new Neuron.SimpleNeuron(0.5)), DELTA);
    }

    @Test
    void shouldReturnCorrectValuesForGaussian() {
        ActivationFunction activationFunction = new Gaussian();
        assertEquals(1.0, activationFunction.activate(new Neuron.SimpleNeuron(0.0)), DELTA);
        assertEquals(0.36787944117, activationFunction.activate(new Neuron.SimpleNeuron(1.0)), DELTA);
        assertEquals(0.36787944117, activationFunction.activate(new Neuron.SimpleNeuron(-1.0)), DELTA);
        assertEquals(0.0, activationFunction.activate(new Neuron.SimpleNeuron(100.0)), DELTA);
        assertEquals(0.0, activationFunction.activate(new Neuron.SimpleNeuron(-100.0)), DELTA);
    }

    @Test
    void shouldReturnCorrectValuesForGaussianDerivative() {
        ActivationFunction activationFunction = new Gaussian();
        assertEquals(-2.0 * 0.0 * activationFunction.activate(new Neuron.SimpleNeuron(0.0)), activationFunction.derivative(new Neuron.SimpleNeuron(0.0)), DELTA);
        assertEquals(-2.0 * 1.0 * activationFunction.activate(new Neuron.SimpleNeuron(1.0)), activationFunction.derivative(new Neuron.SimpleNeuron(1.0)), DELTA);
        assertEquals(-2.0 * -1.0 * activationFunction.activate(new Neuron.SimpleNeuron(-1.0)), activationFunction.derivative(new Neuron.SimpleNeuron(-1.0)), DELTA);
        assertEquals(-2.0 * 100.0 * activationFunction.activate(new Neuron.SimpleNeuron(100.0)), activationFunction.derivative(new Neuron.SimpleNeuron(100.0)), DELTA);
        assertEquals(-2.0 * -100.0 * activationFunction.activate(new Neuron.SimpleNeuron(-100.0)), activationFunction.derivative(new Neuron.SimpleNeuron(-100.0)), DELTA);
    }

    @Test
    void shouldReturnCorrectValuesForHyperbolicTangent() {
        ActivationFunction activationFunction = new HyperbolicTangent();
        assertEquals(0.995054754, activationFunction.activate(new Neuron.SimpleNeuron(3.0)), DELTA);
        assertEquals(0.0, activationFunction.activate(new Neuron.SimpleNeuron(0.0)), DELTA);
        assertEquals(0.761594156, activationFunction.activate(new Neuron.SimpleNeuron(1.0)), DELTA);
        assertEquals(-0.761594156, activationFunction.activate(new Neuron.SimpleNeuron(-1.0)), DELTA);
        assertEquals(-1.0, activationFunction.activate(new Neuron.SimpleNeuron(-100.0)), DELTA);
        assertEquals(0.462117157, activationFunction.activate(new Neuron.SimpleNeuron(0.5)), DELTA);
    }

    @Test
    void shouldReturnCorrectValuesForHyperbolicTangentDerivative() {
        ActivationFunction activationFunction = new HyperbolicTangent();
        assertEquals(1.0 - 3 * 3, activationFunction.derivative(new Neuron.SimpleNeuron(3.0)), DELTA);
        assertEquals(1.0 - 0 * 0, activationFunction.derivative(new Neuron.SimpleNeuron(0.0)), DELTA);
        assertEquals(1.0 - 1 * 1, activationFunction.derivative(new Neuron.SimpleNeuron(1.0)), DELTA);
        assertEquals(1.0 - -1 * -1, activationFunction.derivative(new Neuron.SimpleNeuron(-1.0)), DELTA);
        assertEquals(1.0 - -100 * -100, activationFunction.derivative(new Neuron.SimpleNeuron(-100.0)), DELTA);
        assertEquals(1.0 - 0.5 * 0.5, activationFunction.derivative(new Neuron.SimpleNeuron(0.5)), DELTA);
    }

    @Test
    void shouldReturnSameValuesForIdentity() {
        ActivationFunction activationFunction = new Identity();
        assertEquals(0.0, activationFunction.activate(new Neuron.SimpleNeuron(0.0)), DELTA);
        assertEquals(1.0, activationFunction.activate(new Neuron.SimpleNeuron(1.0)), DELTA);
        assertEquals(-1.0, activationFunction.activate(new Neuron.SimpleNeuron(-1.0)), DELTA);
        assertEquals(-98723.0, activationFunction.activate(new Neuron.SimpleNeuron(-98723.0)), DELTA);
        assertEquals(8975987.0, activationFunction.activate(new Neuron.SimpleNeuron(8975987.0)), DELTA);
    }

    @Test
    void shouldReturnSameValuesForIdentityDerivative() {
        ActivationFunction activationFunction = new Identity();
        assertEquals(1.0, activationFunction.derivative(new Neuron.SimpleNeuron(0.0)), DELTA);
        assertEquals(1.0, activationFunction.derivative(new Neuron.SimpleNeuron(1.0)), DELTA);
        assertEquals(1.0, activationFunction.derivative(new Neuron.SimpleNeuron(-1.0)), DELTA);
        assertEquals(1.0, activationFunction.derivative(new Neuron.SimpleNeuron(-98723.0)), DELTA);
        assertEquals(1.0, activationFunction.derivative(new Neuron.SimpleNeuron(876493.0)), DELTA);
    }

    @Test
    void shouldReturnCorrectValuesForSigmoid() {
        ActivationFunction activationFunction = new Sigmoid();
        assertEquals(0.952574127, activationFunction.activate(new Neuron.SimpleNeuron(3.0)), DELTA);
        assertEquals(0.622459331, activationFunction.activate(new Neuron.SimpleNeuron(0.5)), DELTA);
        assertEquals(0.268941421, activationFunction.activate(new Neuron.SimpleNeuron(-1.0)), DELTA);
        assertEquals(0.5, activationFunction.activate(new Neuron.SimpleNeuron(0.0)), DELTA);
        assertEquals(0.731058579, activationFunction.activate(new Neuron.SimpleNeuron(1.0)), DELTA);
    }

    @Test
    void shouldReturnCorrectValuesForSigmoidDerivative() {
        ActivationFunction activationFunction = new Sigmoid();
        assertEquals(0.045176659, activationFunction.derivative(new Neuron.SimpleNeuron(0.952574127)), DELTA);
        assertEquals(0.235003712, activationFunction.derivative(new Neuron.SimpleNeuron(0.622459331)), DELTA);
        assertEquals(0.196611933, activationFunction.derivative(new Neuron.SimpleNeuron(0.268941421)), DELTA);
        assertEquals(0.25, activationFunction.derivative(new Neuron.SimpleNeuron(0.5)), DELTA);
        assertEquals(0.196611933, activationFunction.derivative(new Neuron.SimpleNeuron(0.731058579)), DELTA);
    }

    @Test
    void shouldReturnCorrectValuesForSinusoid() {
        ActivationFunction activationFunction = new Sinusoid();
        assertEquals(0.8414709848, activationFunction.activate(new Neuron.SimpleNeuron(1.0)), DELTA);
        assertEquals(-0.8414709848, activationFunction.activate(new Neuron.SimpleNeuron(-1.0)), DELTA);
        assertEquals(0.0, activationFunction.activate(new Neuron.SimpleNeuron(0.0)), DELTA);
    }

    @Test
    void shouldReturnCorrectValuesForSinusoidDerivative() {
        ActivationFunction activationFunction = new Sinusoid();
        assertEquals(0.540302306, activationFunction.derivative(new Neuron.SimpleNeuron(1.0)), DELTA);
        assertEquals(0.540302306, activationFunction.derivative(new Neuron.SimpleNeuron(-1.0)), DELTA);
        assertEquals(1.0, activationFunction.derivative(new Neuron.SimpleNeuron(0.0)), DELTA);
    }
}