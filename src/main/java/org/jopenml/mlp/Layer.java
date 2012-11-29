package org.jopenml.mlp;

import java.io.Serializable;
import java.util.Arrays;

import org.jopenml.mlp.activationFunctions.ActivationFunction;

/**
 * This object represents a layer of the neural network.
 */
public class Layer
		implements Serializable {
	
	private static final long serialVersionUID = -4607204450973284028L;
	
	private Layer prevLayer;
	
	private double[][] weightMatrix;
	private double[][] gradientMatrix;
	private double[][] lastGradientMatrix;
	
	private ActivationFunction function;
	
	private double[] output;
	double[] input = null;
	private double bias;
	double[] layerInput;
	
	/**
	 * Constructor; Is initialized with the previous layer, activation function number of neurons and the bias
	 * 
	 * @param prevLayer The previous layer; Null if this layer is the first one.
	 * @param neurons Number of neurons.
	 * @param function Activation function of this layer
	 * @param bias The bias
	 * @throws BadConfigException Is thrown in case of incorrect configuration
	 */
	public Layer(Layer prevLayer, int neurons, ActivationFunction function, double bias) {
		
		neurons++;
		
		// Tests configuration
		if (neurons <= 0) {
			throw new IllegalArgumentException("Illegal amount of neurons: " + neurons);
		}
		
		// Initializing variables
		this.function = function;
		this.bias = bias;
		output = new double[neurons];
		layerInput = new double[neurons];
		
		if (prevLayer == null) {
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
	 * This function is to be used at the inout layer and sets the input data.
	 * 
	 * @param input The input vector, that needs the same dimension as the layer.
	 * @throws BadConfigException Is thrown in case of wrong configuration
	 */
	public boolean setInput(double[] input) {
		if (output.length - 1 != input.length) {
			return false;
		}
		
		this.input = input;
		
		output = Arrays.copyOf(input, output.length);
		output[output.length - 1] = 0;
		return true;
	}
	
	/**
	 * This function uses the reference on the last Layer to calculate the output vector of this Layer . It is recursive
	 * reverted to the first Layer and multiplied with the corresponding weights to create the output vector.
	 * 
	 * @return The output vector of this layer. if this layer is the output layer, this vector is the output of the
	 *         network
	 */
	public double[] getOutput() {
		// Recursion cancel
		if (prevLayer == null) {
			layerInput = output;
			return output;
		}
		
		input = prevLayer.getOutput();
		
		// Generate the output
		for (int h = 0; h < output.length - 1; h++) {
			// Reset
			output[h] = 0;
			
			// Multiply every output of the last Layer with the corresponding matrix and add it.
			for (int i = 0; i < input.length; i++) {
				output[h] += input[i] * weightMatrix[h][i];
			}
			
			layerInput[h] = output[h];
			// Use the activation function on the sum.
			output[h] = function.compute(output[h]);
		}
		
		output[output.length - 1] = bias;
		
		return output;
	}
	
	/**
	 * The number on neurons in the current layer.
	 * 
	 * @return Number of neurons
	 */
	public int getSize() {
		return output.length;
	}
	
	/**
	 * @param error Applies the error of the previuos layer of the current layer. If this layer the last one is, the
	 *            output error will be used. After calculating and saving of the nescesary weight modifications , the
	 *            error of the previous layer will be calculated and passed to the next layer.
	 * @throws BadConfigException if the error vector wrong is.
	 */
	public double backPropagate(double[] error) {
		if (prevLayer == null) {
			return 0;
		}
		
		double retVal = 0;
		// init
		final double[] preLayerError = new double[prevLayer.getSize()];
		
		// alter gradient
		for (int i = 0; i < gradientMatrix.length; i++) {
			for (int h = 0; h < prevLayer.output.length; h++) {
				gradientMatrix[i][h] += error[i] * prevLayer.output[h];
				retVal += error[i] * prevLayer.output[h];
			}
		}
		
		// generate preLayerError
		for (int i = 0; i < prevLayer.getSize(); i++) {
			for (int h = 0; h < error.length; h++) {
				preLayerError[i] += error[h] * weightMatrix[h][i];
			}
			
			preLayerError[i] *= prevLayer.function.derivation(prevLayer.layerInput[i]);
		}
		
		retVal /= gradientMatrix.length;
		return retVal += prevLayer.backPropagate(preLayerError);
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
		
		for (int i = 0; i < output.length; i++) {
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
		
		for (int i = 0; i < output.length; i++) {
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
	 * Returns the current Bias.
	 * 
	 * @return The current bias.
	 */
	public double getBias() {
		return bias;
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
		
		for (int i = 0; i < output.length; i++) {
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
		return function;
	}
	
	/**
	 * Returns the WeightMatrix of the connection of this layer to the previous one.
	 * 
	 * @return the connecting WeightMatrix.
	 */
	public double[][] getWeightMatrix() {
		return weightMatrix;
	}
	
	/**
	 * This Function makes the layer to an autoencoder.
	 * 
	 * @param value This value will be used as input to all neurons of the previous layer and as targetvector.
	 * @param maxIterations This parameter is an break condition for the learning process of this layer.
	 * @param upperBound This parameter is an break condition for the learning process of this layer.
	 * @param eta The used learning rate.
	 * @return Training values for the adjacent layer.
	 * @throws BadConfigException
	 */
	public double[] makeAutoencoder(double value, int maxIterations, double upperBound, double eta) {
		
		final double[] trainingValues = new double[getSize()];
		Arrays.fill(trainingValues, value);
		if (prevLayer == null) {
			return trainingValues;
		}
		
		// Input Data
		final double[] lastLayerOutput = prevLayer.makeAutoencoder(value, maxIterations, upperBound, eta);
		
		// Remove tha layer
		final Layer prevLayer = this.prevLayer.prevLayer;
		this.prevLayer.prevLayer = null;
		
		// Add the new layer
		final Layer additionalLayer = new Layer(this, this.prevLayer.getSize() - 1,
				this.prevLayer.getActivationFunction(), this.prevLayer.getBias());
		
		final double[] firstLayerinput = Arrays.copyOf(lastLayerOutput, lastLayerOutput.length - 1);
		
		// use output ans new input
		if (!this.prevLayer.setInput(firstLayerinput)) {
			System.out.println("Lenght of lastLayerOutput: " + lastLayerOutput.length);
		}
		
		// train Online
		double out[] = null;
		final double[] errVec = new double[additionalLayer.getSize()];
		double overallError;
		
		for (int i = 0; i < maxIterations; i++) {
			out = additionalLayer.getOutput();
			
			// calculate the arror at the output layer
			for (int h = 0; h < errVec.length; h++) {
				errVec[h] = out[h] - lastLayerOutput[h];
			}
			
			overallError = 0;
			for (final double e : errVec) {
				overallError += Math.pow(e, 2);
			}
			if (overallError < (2 * upperBound)) {
				break;
			}
			
			// Backpropagate the error
			additionalLayer.backPropagate(errVec);
			
			// Adjust the weights
			additionalLayer.update(eta, 0);
		}
		
		// Restore the layer
		this.prevLayer.prevLayer = prevLayer;
		
		return getOutput();
	}
}
