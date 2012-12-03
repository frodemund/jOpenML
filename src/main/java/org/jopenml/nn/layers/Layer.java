package org.jopenml.nn.layers;

import java.io.Serializable;
import java.util.Arrays;

import org.jopenml.nn.activationFunctions.ActivationFunction;

/**
 * This object represents a layer of the neural network.
 */
public class Layer
		implements Serializable {
	
	private static final long serialVersionUID = -4607204450973284028L;
	
	private final Layer prevLayer;
	
	private final double[][] weightMatrix;
	private final double[][] gradientMatrix;
	private final double[][] lastGradientMatrix;
	
	private final ActivationFunction activationFunction;
	
	private double[] layerOutput;
	private double[] layerInput;;
	
	/**
	 * Constructor; Is initialized with the previous layer, activation function number of neurons and the bias
	 * 
	 * @param prevLayer The previous layer; Null if this layer is the first one.
	 * @param neurons Number of neurons.
	 * @param function Activation function of this layer
	 * @param bias The bias
	 */
	public Layer(Layer prevLayer, int neurons, ActivationFunction function) {
		
		// Tests configuration
		if (neurons <= 0) {
			throw new IllegalArgumentException("Illegal amount of neurons: " + neurons);
		}
		
		// Initializing variables
		layerOutput = new double[neurons];
		layerInput = new double[neurons];
		activationFunction = function;
		
		if (prevLayer == null) {
			this.prevLayer = null;
			weightMatrix = null;
			gradientMatrix = null;
			lastGradientMatrix = null;
			return;
		}
		
		this.prevLayer = prevLayer;
		weightMatrix = new double[neurons][prevLayer.getSize()];
		gradientMatrix = new double[neurons][prevLayer.getSize()];
		lastGradientMatrix = new double[neurons][prevLayer.getSize()];
		
		// Setting values of matrices and arrays
		for (int h = 0; h < neurons; h++) {
			for (int i = 0; i < prevLayer.getSize(); i++) {
				weightMatrix[h][i] = 0.5 - Math.random();
			}
		}
	}
	
	/**
	 * This function is to be used at the input layer and sets the input data.
	 * 
	 * @param input The input vector, that needs the same dimension as the layer.
	 */
	public void setInput(double[] input) {
		if (layerOutput.length != input.length) {
			throw new IllegalArgumentException("Provided array has wron length. (is " + input.length
					+ ", but should be " + layerOutput.length + ").");
		}
		
		layerInput = Arrays.copyOf(input, layerInput.length);
	}
	
	/**
	 * This function uses the reference on the last Layer to calculate the output vector of this Layer . It is recursive
	 * reverted to the first Layer and multiplied with the corresponding weights to create the output vector.
	 * 
	 * @return The output vector of this layer. if this layer is the output layer, this vector is the output of the
	 *         network
	 */
	public double[] getOutput() {
		if (prevLayer == null) {
			// beak condition
			layerOutput = layerInput;
			return layerOutput;
		}
		
		layerInput = prevLayer.getOutput();
		
		// Generate the output
		for (int h = 0; h < layerOutput.length; h++) {
			double neuronOut = 0;
			
			// Multiply every output of the last Layer with the corresponding matrix and add it.
			for (int i = 0; i < layerInput.length; i++) {
				neuronOut += layerInput[i] * weightMatrix[h][i];
			}
			
			// Use the activation function on the sum.
			layerOutput[h] = activationFunction.compute(neuronOut);
		}
		
		return layerOutput;
	}
	
	/**
	 * The number on neurons in the current layer.
	 * 
	 * @return Number of neurons
	 */
	public int getSize() {
		return layerOutput.length;
	}
	
	/**
	 * @param error Applies the error of the previuos layer of the current layer. If this layer the last one is, the
	 *            output error will be used. After calculating and saving of the nescesary weight modifications , the
	 *            error of the previous layer will be calculated and passed to the next layer.
	 * @throws BadConfigException if the error vector wrong is.
	 */
	public void backPropagate(double[] error) {
		if (prevLayer == null) {
			// break condition
			return;
		}
		alterGradient(error);
		prevLayer.backPropagate(compurePreviousLayerError(error));
	}
	
	private void alterGradient(double[] error) {
		// alter gradient
		for (int i = 0; i < gradientMatrix.length; i++) {
			for (int h = 0; h < prevLayer.layerOutput.length; h++) {
				gradientMatrix[i][h] += error[i] * prevLayer.layerOutput[h];
			}
		}
	}
	
	private double[] compurePreviousLayerError(double[] error) {
		final double[] preLayerError = new double[prevLayer.getSize()];
		
		for (int i = 0; i < prevLayer.getSize(); i++) {
			for (int h = 0; h < error.length; h++) {
				preLayerError[i] += error[h] * weightMatrix[h][i];
			}
			
			preLayerError[i] *= prevLayer.activationFunction.derivation(prevLayer.layerOutput[i]);
		}
		return preLayerError;
	}
	
	/**
	 * Adjusts recursively the weights of the net.
	 * 
	 * @param eta The learning rate to be used.
	 */
	public void update(double eta) {
		if (prevLayer == null) {
			return;
		}
		
		for (int i = 0; i < layerOutput.length; i++) {
			for (int h = 0; h < prevLayer.getSize(); h++) {
				weightMatrix[i][h] -= eta * gradientMatrix[i][h];
				gradientMatrix[i][h] = 0;
			}
		}
		
		prevLayer.update(eta);
	}
	
	/**
	 * Adjusts recursively the weights of the net and uses a momentum to avoide oscillations.
	 * 
	 * @param eta The learning rate to be used.
	 * @param momentum By using the momentum as a value which defines the proportion between the last weightMatrix
	 *            adjustment and the current gradient, oscillations will be avoided. The formula is eta * ((1-momentum)
	 *            * gradient + momentum * lastChange
	 */
	public void update(double eta, double momentum) {
		if (prevLayer == null) {
			return;
		}
		final double negMomentum = 1 - momentum;
		
		for (int i = 0; i < layerOutput.length; i++) {
			for (int h = 0; h < prevLayer.getSize(); h++) {
				lastGradientMatrix[i][h] = eta
						* (negMomentum * gradientMatrix[i][h] + momentum * lastGradientMatrix[i][h]);
				weightMatrix[i][h] -= lastGradientMatrix[i][h];
				gradientMatrix[i][h] = 0;
			}
		}
		
		prevLayer.update(eta, momentum);
	}
	
	/**
	 * String-representation of the current weights and neurons.
	 */
	@Override
	public String toString() {
		if (weightMatrix == null) {
			return "";
		}
		
		final StringBuffer buffer = new StringBuffer();
		
		buffer.append("lastLayer:");
		for (int i = 0; i < weightMatrix[0].length; i++) {
			buffer.append("\t[" + i + "]");
		}
		
		for (int i = 0; i < layerOutput.length; i++) {
			buffer.append("\nNeuron [" + i + "]");
			for (int h = 0; h < prevLayer.getSize(); h++) {
				buffer.append("\t" + weightMatrix[i][h]);
			}
		}
		return buffer.toString();
	}
	
	/**
	 * Returns the {@link ActivationFunction activation function}.
	 * 
	 * @return The activation function, that is used in this layer.
	 */
	public ActivationFunction getActivationFunction() {
		return activationFunction;
	}
	
	public double[] getLayerInput() {
		return layerInput;
	}
}
