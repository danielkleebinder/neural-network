package at.fhtw.ai.nn.activation.layer;

import at.fhtw.ai.nn.Layer;
import at.fhtw.ai.nn.Neuron;

import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantLock;

/**
 * A softmax activation function is an advanced layer based activation function:<br>
 * <code>f(x) = e^x_i / sum(e^x_j)</code><br><br>
 * Derivative:<br>
 * <code>f'(x) = f(x) * (δ - f(x))</code><br><br>
 * The use of any non-linear activation functions which result in negative output values in hidden layers can cause
 * unknown outputs from the softmax function!<br><br>
 * The softmax activation function should only be used with cross-entropy calculations.
 * <p>
 * Created On: 14.05.2018
 *
 * @author Daniel Kleebinder
 * @since 0.0.1
 */
public class Softmax implements LayerActivationFunction {
    private static final long serialVersionUID = -3624111820384845972L;

    /**
     * Sum of all pre computed neuron outputs.
     */
    private double sum = 0.0;

    /**
     * Softmax sum computation lock.
     */
    private Lock lock = new ReentrantLock();

    @Override
    public void initialize(Layer layer) {
        lock.lock();
        try {
            sum = 0.0;
            layer.getNeurons().forEach(neuron -> sum += Math.exp(neuron.preActivationValue));
        } finally {
            lock.unlock();
        }
    }

    @Override
    public double activate(Neuron neuron) {
        return Math.exp(neuron.preActivationValue) / sum;
    }

    @Override
    public double derivative(Neuron neuron) {
        // Mostly ignoring kronecker delta here, because cross-entropy will cancel this out anyways
        return neuron.preActivationValue * (kroneckerDelta(0, 0) - neuron.preActivationValue);
    }

    private double kroneckerDelta(int i, int j) {
        return (i == j) ? 1.0 : 0.0;
    }

    @Override
    public boolean isStochasticDerivative() {
        return true;
    }
}