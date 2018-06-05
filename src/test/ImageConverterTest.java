package test;

import test.loader.GregorImage;
import test.loader.MNISTImageLoadingService;
import test.loader.MnistImage;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * A simple image converter test.
 * <p>
 * Created On: 29.04.2018
 *
 * @author Daniel Kleebinder
 * @since 0.0.1
 */
public class ImageConverterTest {
    public static void main(String[] args) {
        MNISTImageLoadingService loadingService = new MNISTImageLoadingService(
                ".\\train\\t10k-labels-idx1-ubyte.dat",
                ".\\train\\t10k-images-idx3-ubyte.dat");

        List<MnistImage> images = null;
        try {
            images = loadingService.loadMNISTImages(new GregorImage.Factory());
        } catch (IOException e) {
            e.printStackTrace();
        }

        Map<Integer, MnistImage> outputImages = new HashMap<>(11);
        for (MnistImage mnistImage : images) {
            if (outputImages.size() >= 10) {
                break;
            }
            outputImages.putIfAbsent(mnistImage.getLabel(), mnistImage);
        }

        for (MnistImage mnistImage : outputImages.values()) {
            BufferedImage img = new BufferedImage(28, 28, BufferedImage.TYPE_INT_RGB);
            for (int i = 0; i < mnistImage.getData().length; i++) {
                int color = (int) Math.round(mnistImage.getData()[i] * 255.0);
                img.setRGB(i % 28, i / 28, new Color(color, color, color).getRGB());
            }
            try {
                ImageIO.write(
                        img,
                        "PNG",
                        new File(".\\train\\otsu\\number_" + mnistImage.getLabel() + ".png"));
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }
}
