package at.fhtw.ai.nn.activation.rectifier;

import at.fhtw.ai.nn.Neuron;
import at.fhtw.ai.nn.activation.Sigmoid;

/**
 * Derivative of the sigmoid-weighted linear unit, also known as dSiLU activation function. Take a look at the
 * publication for more details: https://arxiv.org/pdf/1702.03118.pdf<br>
 * <p>
 * Created On: 28.05.2018
 *
 * @author Daniel Kleebinder
 * @since 0.0.1
 */
public class SigmoidRectifier extends Rectifier {
    private static final long serialVersionUID = 5089408988069252703L;
    private static final Sigmoid SIGMOID = new Sigmoid();

    @Override
    public double activate(Neuron neuron) {
        double x = neuron.preActivationValue;
        double z = SIGMOID.activate(neuron);
        return z * (1.0 + x * (1.0 - z));
    }

    @Override
    public double derivative(Neuron neuron) {
        double x = neuron.value;
        double z = SIGMOID.activate(neuron);
        return z * (1.0 - z) * (2.0 + x * (1.0 - z) - x * z);
    }
}
