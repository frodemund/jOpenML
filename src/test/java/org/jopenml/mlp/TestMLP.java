package org.jopenml.mlp;

import java.util.Arrays;

import org.junit.Test;

public class TestMLP {
	
	protected static Layer[] layer;
	
	@Test
	public void strangeMasterTest() {
		System.out.print("Erstelle neues MLP...");
		final int[] hL = { 2 };
		final ActivationFunction[] f = new ActivationFunction[3];
		Arrays.fill(f, new TangensHyperbolicus());
		final double[] b = { 1 };
		
		MLP(2, 2, hL, f, b, true);
		System.out.println("fertig");
		
		System.out.print("Erstelle Eingabedaten...");
		
		final int[] target = new int[2];
		final double[][] inputValues = new double[2][2];
		
		target[0] = 1;
		target[1] = 0;
		
		inputValues[0][0] = 1;
		inputValues[0][1] = 0;
		inputValues[1][0] = 0;
		inputValues[1][1] = 1;
		
		System.out.println("fertig");
		
		System.out.print("Erstelle Testeingabedaten...");
		
		final int[] ttarget = new int[2];
		final double[][] tinputValues = new double[2][2];
		
		ttarget[0] = 1;
		ttarget[1] = 0;
		
		tinputValues[0][0] = 1;
		tinputValues[0][1] = 0;
		tinputValues[1][0] = 0;
		tinputValues[1][1] = 1;
		
		System.out.println("fertig");
		
		System.out.println("TrainOnline...");
		double error = 1;
		while (error > 0.00001) {
			error = runOnline(target, inputValues, 0.01);
		}
		
		double output[];
		
		// Eingabedaten setzen
		layer[0].setInput(inputValues[1]);
		
		// Ausgabe berechnen
		output = layer[layer.length - 1].getOutput();
		
		System.out.println("\nDas Ergebnis ist: " + output[0] + " | " + output[1] + " mit " + inputValues[1][0] + " | "
				+ inputValues[1][1]);
		
	}
	
	public void MLP(int inputNeurons, int outputNeurons, int[] hiddenLayers, ActivationFunction[] functions,
			double[] bias, boolean autoencoder) {
		// inputLayer erzeugen
		layer = new Layer[hiddenLayers.length + 2];
		layer[0] = new Layer(null, inputNeurons, functions[0], 0);
		
		// hiddenLayer erzeugen
		for (int i = 0; i < hiddenLayers.length; i++) {
			layer[i + 1] = new Layer(layer[i], hiddenLayers[i], functions[i], bias[i]);
		}
		
		// outputLayer erzeugen
		layer[layer.length - 1] = new Layer(layer[layer.length - 2], outputNeurons, functions[functions.length - 1], 0);
		
		if (!autoencoder) {
			return;
		}
		
		// Als Autoencoder trainieren
		layer[layer.length - 1].makeAutoencoder(0.5, 1000, 0.00001, 0.2);
	}
	
	public double runOnline(int[] intTarget, double[][] input, double eta) {
		if (layer == null) {
			return 0;
		}
		
		double[] out;
		final double[] target = new double[layer[layer.length - 1].getSize()];
		
		for (int i = 0; i < intTarget.length; i++) {
			final int theData = intTarget[i];
			
			// Zielwert erzeugen
			Arrays.fill(target, 0);
			target[theData] = 1;
			
			// Eingabedaten setzen
			layer[0].setInput(input[i]);
			
			// Ausgabe berechnen
			out = layer[layer.length - 1].getOutput();
			
			// Den Fehler an der Ausgabeschicht berechnen
			final double[] errVec = new double[target.length];
			for (int h = 0; h < target.length; h++) {
				errVec[h] = out[h] - target[h];
			}
			
			// Fehler zurückpropagieren
			layer[layer.length - 1].backPropagate(errVec);
			
			// Gewichte anpassen
			layer[layer.length - 1].update(eta, 0.3);
		}
		
		// Mittelwert berechnen und Fehler zurückgeben
		return runTest(intTarget, input);
	}
	
	public double runBatch(int[] intTarget, double[][] input, double eta) {
		if (layer == null) {
			return 0;
		}
		
		double[] out;
		final double[] target = new double[layer[layer.length - 1].getSize()];
		
		for (int i = 0; i < intTarget.length; i++) {
			final int theData = intTarget[i];
			
			// Zielwert erzeugen
			Arrays.fill(target, 0);
			target[theData] = 1;
			
			// Eingabedaten setzen
			layer[0].setInput(input[i]);
			
			// Ausgabe berechnen
			out = layer[layer.length - 1].getOutput();
			
			// Den Fehler an der Ausgabeschicht berechnen
			final double[] errVec = new double[target.length];
			for (int h = 0; h < target.length; h++) {
				errVec[h] = out[h] - target[h];
			}
			
			// Fehler zurückpropagieren
			layer[layer.length - 1].backPropagate(errVec);
		}
		
		// Gewichte anpassen
		layer[layer.length - 1].update(eta, 0.3);
		
		// Mittelwert berechnen und Fehler zurückgeben
		return runTest(intTarget, input);
	}
	
	/**
	 * Diese Methode fuehrt einen Test mit den uebergebenen Daten durch.
	 * 
	 * @param dataCollection die Daten, die fuer den Test verwendet werden sollen
	 * @return den Testfehler
	 */
	public double runTest(int[] intTarget, double[][] input) {
		double err = 0;
		double[] out;
		final double[] target = new double[layer[layer.length - 1].getSize()];
		
		for (int i = 0; i < intTarget.length; i++) {
			final int theData = intTarget[i];
			
			// Zielwert erzeugen
			Arrays.fill(target, 0);
			target[theData] = 1;
			
			// Eingabedaten setzen
			layer[0].setInput(input[i]);
			
			// Ausgabe berechnen
			out = layer[layer.length - 1].getOutput();
			
			for (int h = 0; h < target.length; h++) {
				err += Math.pow(out[h] - target[h], 2);
			}
			
		}
		
		// Mittelwert berechnen und Fehler zurückgeben
		return 0.5 * err / intTarget.length;
	}
	
}
