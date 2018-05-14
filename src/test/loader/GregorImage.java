package test.loader;

import java.util.function.BiFunction;

/**
 * A simple conversion image for MNIST image data.
 * <p>
 * Created On: 03.05.2018
 *
 * @author Daniel Kleebinder
 * @since 0.0.1
 */
public class GregorImage extends MnistImage {

    /**
     * Creates a new image.
     *
     * @param label Label.
     * @param data  Data.
     */
    public GregorImage(int label, byte[] data) {
        super(label, data);
    }

    @Override
    protected void convert(byte[] rawData) {
        for (int i = 0; i < rawData.length; i++) {
            int value = rawData[i] & 0xFF;
            data[i] = value <= 0 ? 0.0 : 1.0;
        }
    }

    /**
     * A simple factory for creating instances of the image.
     */
    public static class Factory implements BiFunction<Integer, byte[], MnistImage> {
        @Override
        public MnistImage apply(Integer label, byte[] rawData) {
            return new GregorImage(label, rawData);
        }
    }
}
