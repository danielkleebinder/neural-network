package at.fhtw.ai.nn.connect;

import at.fhtw.ai.nn.Layer;

import java.io.Serializable;

/**
 * Connectors are used to connect the neurons of two different layers.
 * <p>
 * Created On: 14.05.2018
 *
 * @author Daniel Kleebinder
 * @since 0.0.1
 */
public interface Connector extends Serializable {

    /**
     * Connects the two given layers.
     *
     * @param left  Left layer.
     * @param right Right layer.
     */
    void connect(Layer left, Layer right);
}