package test.genetic.runner;

import at.fhtw.ai.nn.Layer;
import at.fhtw.ai.nn.NeuralNetwork;
import at.fhtw.ai.nn.Neuron;
import at.fhtw.ai.nn.Synapse;

import javax.swing.*;
import java.awt.*;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import java.util.concurrent.CopyOnWriteArrayList;

/**
 * Created On: 14.05.2018
 *
 * @author Daniel Kleebinder
 * @since 0.0.1
 */
public class VisualizerFrame extends JFrame {

    private Random random = new Random();
    private RenderPanel renderPanel = new RenderPanel();

    public List<Obstacle> obstacles = new CopyOnWriteArrayList<>();
    public List<Runner> runners = new CopyOnWriteArrayList<>();

    public VisualizerFrame(String title) throws HeadlessException {
        super(title);

        setDefaultCloseOperation(EXIT_ON_CLOSE);
        setPreferredSize(new Dimension(1500, 870));
        setSize(getPreferredSize());
        setLocationRelativeTo(null);

        getContentPane().add(renderPanel);

        setVisible(true);
    }

    public boolean isEveryRunnerDead() {
        for (Runner runner : runners) {
            if (!runner.dead) {
                return false;
            }
        }
        return true;
    }

    public Runner[] getScorer(int top, int ad) {
        Runner[] result = new Runner[top + ad];
        Collections.sort(runners);
        for (int i = 0; i < result.length - ad; i++) {
            result[i] = runners.get(i);
        }
        for (int i = 0; i < ad; i++) {
            result[top + i] = runners.get(random.nextInt(runners.size() - top) + top);
        }
        return result;
    }

    public void start() {
        // Render Thread
        new Thread(() -> {
            while (true) {
                SwingUtilities.invokeLater(() -> {
                    renderPanel.invalidate();
                    renderPanel.repaint();
                });
                try {
                    Thread.sleep(20);
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }
            }
        }).start();

        // Update Thread
        new Thread(() -> {
            while (true) {
                SwingUtilities.invokeLater(() -> {
                    for (Obstacle obstacle : obstacles) {
                        obstacle.position.x -= obstacle.speed;
                        if (obstacle.position.x < -obstacle.size.width) {
                            obstacle.position.x = getWidth();
                            obstacle.speed = random.nextInt(20) + 1;
                        }
                    }

                    double nearestDistance = Double.MAX_VALUE;
                    for (Runner runner : runners) {
                        for (Obstacle obstacle : obstacles) {
                            double dist = distance(runner, obstacle);
                            if (nearestDistance > dist) {
                                nearestDistance = dist;
                            }
                        }
                        double speed = 0.0;
                        if (obstacles.size() > 0) {
                            speed = (obstacles.get(0).speed - 1.0) / 20.0;
                        }
                        runner.input(nearestDistance, speed);
                    }

                    for (Runner runner : runners) {
                        runner.update();
                    }

                    Rectangle runnerCollider = new Rectangle();
                    Rectangle obstacleCollider = new Rectangle();
                    for (Obstacle obstacle : obstacles) {
                        obstacleCollider.setLocation(obstacle.position);
                        obstacleCollider.setSize(obstacle.size);
                        for (Runner runner : runners) {
                            if (runner.tunnelMode) {
                                continue;
                            }

                            runnerCollider.setLocation(runner.position);
                            runnerCollider.setSize(runner.size);

                            if (obstacleCollider.intersects(runnerCollider)) {
                                runner.dead = true;
                            }
                        }
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

    private double distance(Entity runner, Entity obstacle) {
        double rx = runner.position.x + (runner.size.width / 2);
        double ry = runner.position.y + (runner.size.height / 2);
        double ox = obstacle.position.x + (obstacle.size.width / 2);
        double oy = obstacle.position.y + (obstacle.size.height / 2);

        double dx = Math.max(Math.abs(rx - ox) - obstacle.size.width / 2, 0);
        double dy = Math.max(Math.abs(ry - oy) - obstacle.size.height / 2, 0);

        return Math.sqrt(dx * dx + dy * dy) / getWidth();
    }

    private class RenderPanel extends JPanel {

        private Color backgroundColor = new Color(40, 120, 50);
        private Color runnerColor = new Color(120, 40, 50);
        private Color tunnelColor = new Color(50, 40, 120);
        private Color obstacleColor = new Color(120, 120, 40);

        @Override
        public void paintComponent(Graphics g) {
            super.paintComponent(g);

            int width = getWidth();
            int height = getHeight();

            Graphics2D g2d = (Graphics2D) g;
            g2d.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);

            // Draw background
            g2d.setColor(backgroundColor);
            g2d.fillRect(0, 0, width, height);

            // Draw tunnel mode
            g2d.setColor(tunnelColor);
            for (Runner runner : runners) {
                if (!runner.dead && runner.tunnelMode) {
                    g2d.fillRect(runner.position.x - 2, runner.position.y - 2, 24, 24);
                }
            }

            // Draw runner
            g2d.setColor(runnerColor);
            for (Runner runner : runners) {
                if (runner.dead) {
                    continue;
                }
                g2d.fillRect(runner.position.x, runner.position.y, runner.size.width, runner.size.height);
            }

            // Draw obstacles
            g2d.setColor(obstacleColor);
            for (Obstacle obstacle : obstacles) {
                g2d.fillRect(obstacle.position.x, obstacle.position.y, obstacle.size.width, obstacle.size.height);
            }

            // Draw best brain
            if (runners.size() > 0) {
                NeuralNetwork nn = getScorer(1, 0)[0].brain;

                int maxNeurons = 0;
                for (Layer layer : nn.getLayers()) {
                    maxNeurons = Math.max(maxNeurons, layer.getNeurons().size());
                }

                int xsep = 150;
                int ysep = 50;
                int w = 30;
                int h = 30;

                int frameWidth = nn.getLayers().size() * xsep + 20 - xsep + w;

                g2d.setColor(new Color(190, 190, 220));
                g2d.translate(getWidth() - frameWidth - 20, 20);
                g2d.fillRect(0, 0, frameWidth, maxNeurons * ysep + 20 - ysep + h);
                g2d.translate(10, 10);

                g2d.setColor(tunnelColor);
                for (int i = 0; i < nn.getLayers().size(); i++) {
                    Layer layer = nn.getLayers().get(i);
                    Layer childLayer = null;
                    if (i < nn.getLayers().size() - 1) {
                        childLayer = nn.getLayers().get(i + 1);
                    }
                    int numNeurons = layer.getNeurons().size();
                    for (int j = 0; j < numNeurons; j++) {
                        g2d.fillOval(i * xsep, j * ysep + ((maxNeurons - numNeurons) / 2) * ysep, w, h);

                        Neuron neuron = layer.getNeurons().get(j);
                        if (childLayer != null) {
                            int k = 0;
                            int numChildNeurons = childLayer.getNeurons().size();
                            for (Neuron childNeuron : childLayer.getNeurons()) {
                                for (Synapse synapse : neuron.getOutputSynapses()) {
                                    if (Double.compare(synapse.weight, 0.0) == 0) {
                                        continue;
                                    }
                                    if (childNeuron == synapse.destinationNeuron) {
                                        g2d.setStroke(new BasicStroke((int) Math.min(Math.ceil(Math.abs(synapse.weight * 3.0)), 12.0), BasicStroke.CAP_ROUND, BasicStroke.JOIN_MITER));
                                        g2d.drawLine(
                                                i * xsep + w,
                                                j * ysep + h / 2 + ((maxNeurons - numNeurons) / 2) * ysep,
                                                (i + 1) * xsep,
                                                k * ysep + h / 2 + ((maxNeurons - numChildNeurons) / 2) * ysep
                                        );
                                    }
                                }
                                k++;
                            }
                        }
                    }
                }
            }
        }
    }
}