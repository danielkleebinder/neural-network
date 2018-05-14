package test.genetic.runner;

import at.fhtw.ai.nn.Layer;
import at.fhtw.ai.nn.NeuralNetwork;
import at.fhtw.ai.nn.Neuron;
import at.fhtw.ai.nn.Synapse;
import at.fhtw.ai.nn.activation.ActivationFunction;
import at.fhtw.ai.nn.activation.HyperbolicTangent;
import at.fhtw.ai.nn.activation.Identity;
import at.fhtw.ai.nn.activation.Sigmoid;
import at.fhtw.ai.nn.activation.rectifier.ExponentialRectifier;
import at.fhtw.ai.nn.activation.rectifier.Rectifier;
import at.fhtw.ai.nn.connect.DenseConnector;
import at.fhtw.ai.nn.initialize.XavierInitializer;
import at.fhtw.ai.nn.utils.NeuralNetworkBuilder;

import javax.swing.*;
import java.util.Arrays;
import java.util.Random;

/**
 * Created On: 14.05.2018
 *
 * @author Daniel Kleebinder
 * @since 0.0.1
 */
public class RunnerNetworkMain {
    private static final Random rnd = new Random();
    private static VisualizerFrame visualizer;
    private static int generation = 0;

    public static void main(String[] args) {
        visualizer = new VisualizerFrame("Runner NeuralNetwork - by Daniel Kleebinder");
        visualizer.start();

        spawn();

        new Thread(() -> {
            while (true) {
                SwingUtilities.invokeLater(() -> {
                    if (visualizer.isEveryRunnerDead()) {
                        Runner[] best = visualizer.getScorer(5, 0);
                        System.out.println("Survivors (Generation: " + (++generation) + "): " + Arrays.toString(best));

                        Runner[] babies = new Runner[best.length];
                        for (int i = 0; i < babies.length; i++) {
                            babies[i] = breed(best[i], best[rnd.nextInt(best.length)]);
                        }

                        respawn(babies);
                    }
                });
                try {
                    Thread.sleep(20);
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }
            }
        }).start();
    }

    private static void spawn() {
        for (int i = 0; i < 30; i++) {
            addRunner(i);
        }
        for (int i = 0; i < 4; i++) {
            addObstacle(1800);
        }
    }

    private static void respawn(Runner... children) {
        visualizer.runners.clear();
        visualizer.obstacles.clear();

        for (int i = 0; i < 30; i++) {
            if (i < children.length) {
                children[i].respawn();
                children[i].position.y = 40 + i * 25;
                visualizer.runners.add(children[i]);
            } else {
                addRunner(i);
            }
        }
        for (int i = 0; i < 4; i++) {
            addObstacle(1800);
        }
    }

    private static void addRunner(int i) {
        Runner runner = new Runner();
        runner.position.x = 100;
        runner.position.y = 40 + i * 25;
        runner.probability = rnd.nextInt(20) + 10;
        runner.brain = generateRandomNeuralNetwork();
        visualizer.runners.add(runner);
    }

    private static void addObstacle(int x) {
        Obstacle obstacle = new Obstacle();
        obstacle.position.x = x;
        obstacle.position.y = 20;
        obstacle.size.width = 40;
        obstacle.size.height = 790;
        obstacle.speed = Math.random() * 20.0 + 1.0;
        visualizer.obstacles.add(obstacle);
    }

    private static NeuralNetwork generateRandomNeuralNetwork() {
        NeuralNetworkBuilder nnb = new NeuralNetworkBuilder();

        nnb.inputLayer("Input Layer", 2, new Identity());
        nnb.outputLayer("Output Layer", 1, new Sigmoid());

        for (int i = 0; i < rnd.nextInt(5) + 3; i++) {
            nnb.hiddenLayer("Hidden Layer - " + i, rnd.nextInt(5) + 3, randomActivationFunction());
        }

        nnb.connector(new DenseConnector());
        nnb.initializer(new XavierInitializer());
        return nnb.build();
    }

    private static Runner breed(Runner mother, Runner father) {
        Runner baby = new Runner();
        baby.position.x = 100;
        baby.brain = mother.brain;
        baby.probability = rnd.nextBoolean() ? mother.probability : father.probability;
        mutate(baby);
        return baby;
    }

    private static void mutate(Runner runner) {
        for (Layer layer : runner.brain.getLayers()) {
            for (Neuron neuron : layer.getNeurons()) {
                for (Synapse synapse : neuron.getOutputSynapses()) {
                    synapse.weight += (Math.random() - 0.5) * 0.1;
                    synapse.weight *= 1.0 + ((Math.random() - 0.5) * 0.1);

                    if (rnd.nextInt(runner.probability) == 0) {
                        synapse.weight = 0.0;
                    }
                }
                neuron.bias.weight += (Math.random() - 0.5) * 0.001;
                neuron.bias.weight *= 1.0 + ((Math.random() - 0.5) * 0.001);
            }
        }
    }

    private static ActivationFunction randomActivationFunction() {
        switch (rnd.nextInt(4)) {
            case 0:
                return new Rectifier();
            case 1:
                return new Sigmoid();
            case 2:
                return new Identity();
            case 3:
                return new HyperbolicTangent();
        }
        return new ExponentialRectifier();
    }
}