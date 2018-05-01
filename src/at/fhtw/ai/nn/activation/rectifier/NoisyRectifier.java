package at.fhtw.ai.nn.activation.rectifier;

import java.util.Random;

/**
 * Noisy rectifier activation function.<br>
 * <code>f(x) = max(0,x+Y) for Y ~ N(0,1)</code><br><br>
 * Derivative:<br>
 * <code>f'(x) = f(a,x)+a for x < 0</code><br>
 * <code>f'(x) = 1 for x >= 0</code>
 * <p>
 * Created On: 01.05.2018
 *
 * @author Daniel Kleebinder
 * @since 0.0.1
 */
public class NoisyRectifier extends Rectifier {

    /**
     * Random noisy gaussian generator.
     */
    private Random rnd = new Random();

    @Override
    public double activate(double x) {
        return super.activate(x + rnd.nextGaussian());
    }
}
