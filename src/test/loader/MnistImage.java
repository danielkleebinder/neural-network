package test.loader;

/**
 * The MNIST image base class.
 * <p>
 * Created On: 03.05.2018
 *
 * @author Daniel Kleebinder
 * @since 0.0.1
 */
public abstract class MnistImage {

    /**
     * Image label.
     */
    protected int label;

    /**
     * Image data.
     */
    protected double[] data;

    /**
     * Creates a new MNIST image.
     *
     * @param label Label.
     * @param data  Data.
     */
    public MnistImage(int label, byte[] data) {
        this.label = label;
        this.data = new double[data.length];
        convert(data);
    }

    /**
     * Converts and normalizes the MNIST data.
     *
     * @param rawData Raw image data.
     */
    protected abstract void convert(byte[] rawData);

    /**
     * Returns the label of the image.
     *
     * @return Label.
     */
    public int getLabel() {
        return label;
    }

    /**
     * Returns the image data.
     *
     * @return Image data.
     */
    public double[] getData() {
        return data;
    }
}
