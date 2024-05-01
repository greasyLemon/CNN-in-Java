package UI;

import data.DataReader;
import data.Image;
import network.NeuralNetwork;

import java.awt.*;
import java.awt.event.*;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.List;
import javax.swing.*;

public class Draw extends JFrame implements MouseMotionListener, ActionListener {
    double[][] matrix = new double[28][28];
    int brushSize = 2;
    JLabel predictLabel = new JLabel("Prediction: -");
    JPanel drawPanel;
    JButton stopButton;
    JButton clearButton;
    JButton showButton;
    List<Object> array =  new ArrayList<>();

    public Draw() {
        setSize(395, 365);
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
        drawPanel.setBorder(BorderFactory.createLineBorder(Color.decode("#3700B3"), 6, false));

        stopButton = new JButton("Predict");
        stopButton.addActionListener(this);
        stopButton.setBackground(Color.decode("#03DAC6"));
        stopButton.setForeground(Color.white);

        clearButton = new JButton("Clear");
        clearButton.addActionListener(this);
        clearButton.setBackground(Color.decode("#03DAC6"));
        clearButton.setForeground(Color.white);

        showButton = new JButton("Show");
        showButton.addActionListener(this);
        showButton.setBackground(Color.decode("#03DAC6"));
        showButton.setForeground(Color.white);

        predictLabel.setBorder(BorderFactory.createCompoundBorder(BorderFactory.createLineBorder(Color.decode("#CF6679"), 6, false),BorderFactory.createEmptyBorder(10,10,10,10)));

        JPanel controlPanel = new JPanel();
        controlPanel.add(stopButton);
        controlPanel.add(clearButton);
        controlPanel.add(showButton);
        controlPanel.setBorder(BorderFactory.createLineBorder(Color.decode("#BB86FC"), 4, false));
        //controlPanel.setBackground(Color.white);

        add(drawPanel, BorderLayout.CENTER);
        add(controlPanel, BorderLayout.SOUTH);
        add(predictLabel, BorderLayout.EAST);

        setVisible(true);
    }

    @Override
    public void mouseDragged(MouseEvent e) {

        int x = e.getX() / 10;
        int y = e.getY() / 10;
        for (int i = y; i < y + brushSize && i < 28; i++) {
            for (int j = x; j < x + brushSize && j < 28; j++) {
                matrix[i][j] = 255;
            }
        }
        drawPanel.repaint();
    }

    @Override
    public void mouseMoved(MouseEvent e) {
    }

    @Override
    public void actionPerformed(ActionEvent e) {

        if (e.getSource() == stopButton) {
            saveMatrixToFile("output.txt");
            //System.out.println("Lưu ma trận thành công.");
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
            array.add(result);
            predictLabel.setText("Prediction: " + result);
            //System.out.println("Predict: " + result);
        } else if (e.getSource() == clearButton) {
            for (int i = 0; i < 28; i++) {
                for (int j = 0; j < 28; j++) {
                    matrix[i][j] = 0;                }

            }
            predictLabel.setText("Prediction: -");
            drawPanel.repaint();
        } else if (e.getSource() == showButton) {
            for (int i = 0; i < array.size(); i++) {
                System.out.print((int) array.get(i));
            }
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
        new Draw();
    }
}