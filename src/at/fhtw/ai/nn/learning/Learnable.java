package at.fhtw.ai.nn.learning;

/**
 * Basic learning interface for everything that learns.
 * <p>
 * Created On: 24.04.2018
 *
 * @author Daniel Kleebinder
 * @since 0.0.1
 */
public interface Learnable {
    /**
     * Invokes the learning algorithm.
     */
    void learn();
}
