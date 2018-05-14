package at.fhtw.ai.nn.connect;

import at.fhtw.ai.nn.Layer;

/**
 * A dense connector, also known as fully connector, is used to connect every neuron in the left layer to every neuron
 * in the right layer.
 * <p>
 * Created On: 14.05.2018
 *
 * @author Daniel Kleebinder
 * @since 0.0.1
 */
public class DenseConnector implements Connector {
    private static final long serialVersionUID = 4531951799387530113L;

    @Override
    public void connect(Layer left, Layer right) {
        left.getNeurons().forEach(neuron -> {
            right.getNeurons().forEach(current -> current.connectInput(neuron));
        });
    }
}