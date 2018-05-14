package at.fhtw.ai.nn.connect;

import at.fhtw.ai.nn.Layer;

/**
 * A random connector is a connector which randomly decides which neurons should be connected.
 * <p>
 * Created On: 14.05.2018
 *
 * @author Daniel Kleebinder
 * @since 0.0.1
 */
public class RandomConnector implements Connector {
    private static final long serialVersionUID = -610957631094393334L;

    /**
     * Connection probability.
     */
    protected double connectionProbability;

    /**
     * Creates a new random connector.
     */
    public RandomConnector() {
        this(0.8);
    }

    /**
     * Creates a new random connector.
     *
     * @param connectionProbability Connection probability.
     */
    public RandomConnector(double connectionProbability) {
        this.connectionProbability = connectionProbability;
    }

    @Override
    public void connect(Layer left, Layer right) {
        left.getNeurons().forEach(neuron -> {
            right.getNeurons().forEach(current -> {
                if (Math.random() > connectionProbability) {
                    return;
                }
                current.connectInput(neuron);
            });
        });
    }

    /**
     * Sets the probability for a connection. A value of 0.8 would indicate, that 80% of all possible connections are
     * implemented.
     *
     * @param connectionProbability Connection probability.
     */
    public void setConnectionProbability(double connectionProbability) {
        this.connectionProbability = connectionProbability;
    }

    /**
     * Returns the connection probability.
     *
     * @return Connection probability.
     */
    public double getConnectionProbability() {
        return connectionProbability;
    }
}