package UI;

import data.DataReader;
import data.Image;
import network.NeuralNetwork;

import java.awt.*;
import java.awt.event.*;
import java.io.FileNotFoundException;
import javax.swing.*;

public class MouseMotionListenerDemo extends JFrame implements MouseMotionListener, ActionListener {
    double[][] matrix = new double[28][28];
    JPanel drawPanel;
    JButton stopButton;
    JButton clearButton;

    public MouseMotionListenerDemo() {
        setSize(300, 350);
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);

        drawPanel = new JPanel() {
            @Override
            protected void paintComponent(Graphics g) {
                super.paintComponent(g);
                for (int i = 0; i < 28; i++) {
                    for (int j = 0; j < 28; j++) {
                        double colorValue = matrix[i][j];
                        g.setColor(new Color((int) colorValue, (int) colorValue, (int) colorValue));
                        g.fillRect(j * 10, i * 10, 10, 10);
                    }
                }
            }
        };
        drawPanel.setPreferredSize(new Dimension(280, 280));
        drawPanel.addMouseMotionListener(this);

        stopButton = new JButton("Stop");
        stopButton.addActionListener(this);

        clearButton = new JButton("Clear");
        clearButton.addActionListener(this);

        JPanel controlPanel = new JPanel();
        controlPanel.add(stopButton);
        controlPanel.add(clearButton);

        add(drawPanel, BorderLayout.CENTER);
        add(controlPanel, BorderLayout.SOUTH);

        setVisible(true);
    }

    @Override
    public void mouseDragged(MouseEvent e) {
        int brushSize = 2;
        int x = e.getX() / 10;
        int y = e.getY() / 10;
        for (int i = y; i < y + brushSize && i < 28; i++) {
            for (int j = x; j < x + brushSize && j < 28; j++) {
                matrix[i][j] = 255;
            }
        }
        drawPanel.repaint(); // Vẽ lại panel
    }

    @Override
    public void mouseMoved(MouseEvent e) {
    }

    @Override
    public void actionPerformed(ActionEvent e) {
        if (e.getSource() == stopButton) {
            saveMatrixToFile("output.txt");
            System.out.println("Lưu ma trận thành công.");

            NeuralNetwork network = new NeuralNetwork();
            NeuralNetwork net = null;
            try {
                net = network.load(3,"ckpt");
            } catch (FileNotFoundException ex) {
                throw new RuntimeException(ex);
            }
            double[][] img = new double[0][];
            try {
                img = DataReader.loadImage("output.txt");
            } catch (FileNotFoundException ex) {
                throw new RuntimeException(ex);
            }
            data.Image test = new Image(img,2);
            int result = net.guess(test);
            System.out.println("Predict: " + result);
        } else if (e.getSource() == clearButton) {
            for (int i = 0; i < 28; i++) {
                for (int j = 0; j < 28; j++) {
                    matrix[i][j] = 0;
                }
            }

            // Trigger repaint for cleared canvas
            drawPanel.repaint();
        }
    }

    private void saveMatrixToFile(String fileName) {
        try (java.io.PrintWriter output = new java.io.PrintWriter(fileName)) {
            for (int i = 0; i < 28; i++) {
                for (int j = 0; j < 28; j++) {
                    output.print(matrix[i][j] + " ");
                }
                output.println();
            }
        } catch (java.io.FileNotFoundException ex) {
            System.out.println("Lỗi khi lưu ma trận: " + ex.getMessage());
        }
    }

    public static void main(String[] args) {
        new MouseMotionListenerDemo();
    }
}