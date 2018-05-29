package test.revert;

import at.fhtw.ai.nn.Layer;
import at.fhtw.ai.nn.NeuralNetwork;
import at.fhtw.ai.nn.Neuron;
import at.fhtw.ai.nn.activation.ActivationFunction;
import at.fhtw.ai.nn.activation.Identity;
import at.fhtw.ai.nn.activation.Logit;
import at.fhtw.ai.nn.utils.Utils;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.Arrays;

/**
 * Created On: 01.05.2018
 *
 * @author Daniel Kleebinder
 * @since 0.0.1
 */
public class RevertTest {
    public static void main(String[] args) throws IOException, ClassNotFoundException {
        System.out.println("Loading Neural Network...");
        //NeuralNetwork neuralNetwork = Utils.deserialize("../NeuralNetwork_E20_97_74.dat");
        NeuralNetwork neuralNetwork = Utils.deserialize("C:\\Users\\Daniel\\Desktop\\pl3\\NeuralNetwork_E9_C0.dat");

        System.out.println("Reversing Neural Network...");
        neuralNetwork.setActivationFunctions(new Logit());

        double[] inputs = new double[10];
        Arrays.fill(inputs, 0.0);
        for (int i = 0; i < 10; i++) {
            if (i - 1 >= 0) {
                inputs[i - 1] = 0.0;
            }
            inputs[i] = 1.0;
            for (int j = 0; j < neuralNetwork.getOutputLayer().getNeurons().size(); j++) {
                neuralNetwork.getOutputLayer().getNeurons().get(j).value = inputs[j];
            }
            reverse(neuralNetwork);
            double[] imageData = new double[28 * 28];
            for (int j = 0; j < neuralNetwork.getInputLayer().getNeurons().size(); j++) {
                imageData[j] = neuralNetwork.getInputLayer().getNeurons().get(j).value;
            }
            MnistImage image = new MnistImage(imageData);
            image.save("../RGI (Vanilla) 2/number_" + i + ".png");
        }
    }

    private static void reverse(NeuralNetwork neuralNetwork) {
        for (Neuron neuron : neuralNetwork.getInputLayer().getNeurons()) {
            neuron.fireReverse();
        }
        for (Layer layer : neuralNetwork.getLayers()) {
            layer.setNeuronsFired(false);
        }
    }

    private static class MnistImage {
        private int width = 28;
        private int height = 28;
        private double[] pixels;

        public MnistImage(double[] pixels) {
            if (pixels.length != (width * height)) {
                throw new IllegalArgumentException("Number of pixels must be " + (width * height));
            }
            this.pixels = pixels;
        }

        public void save(String file) {
            BufferedImage image = new BufferedImage(28, 28, BufferedImage.TYPE_INT_RGB);
            double min = 1.0;
            double max = 0.0;
            ActivationFunction activationFunction = new Identity();
            for (int x = 0; x < width; x++) {
                for (int y = 0; y < height; y++) {
                    double pixel = pixels[y * width + x];

                    min = Math.min(min, pixel);
                    max = Math.max(max, pixel);

                    pixel = activationFunction.activate(pixel + 2.0) * 50.0;

                    pixel = Math.max(pixel, 0.0);
                    pixel = Math.min(pixel, 255.0);
                    int c = (int) Math.round(pixel);

                    Color color = new Color(c, c, c);
                    image.setRGB(x, y, color.getRGB());
                }
            }
            System.out.println("Min: " + min + ", Max: " + max);
            try {
                ImageIO.write(image, "PNG", new File(file));
            } catch (IOException e) {
                e.printStackTrace();
            }
        }

        @Override
        public String toString() {
            StringBuilder result = new StringBuilder();
            for (int i = 0; i < pixels.length; i++) {
                char color = ' ';
                if (pixels[i] > 0.1) {
                    color = '#';
                }
                result.append(color);
                if (i % width == 0) {
                    result.append('\n');
                }
            }
            return result.toString();
        }
    }
}