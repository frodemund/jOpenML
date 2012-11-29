package org.jopenml.mlp;

import java.io.Serializable;
import java.util.Collection;
import java.util.List;

import org.jopenml.mlp.Data;

public class MLPStub implements Serializable, NeuralNetwork {

	private static final long serialVersionUID = 315293690250102356L;

	private String name;
	private int[] layers;
	private ActivationFunction[] activations;
	private double[] biases;
	private boolean autoEncoder;
	private MLP mlp;
	private List<ActivationFunction> allActivations;
	private int maxIterations;
	private double maxError;
	private double eta;
	private Data.Type type;

	public MLPStub(String name, int numLayers, int numNeurons,
			int activationFunction, double bias,
			List<ActivationFunction> allActivations) {
		this.name = name;
		this.allActivations = allActivations;
		layers = new int[numLayers];
		activations = new ActivationFunction[numLayers];
		biases = new double[numLayers];
		for (int i = 0; i < numLayers; i++) {
			layers[i] = numNeurons;
			activations[i] = allActivations.get(activationFunction);
			biases[i] = bias;
		}
		layers[0] = 1;
	}

	/**
	 * Deletes the current (trained) MLP but keeps the configuration.
	 * 
	 * If there is no MLP, this function does nothing.
	 */
	public void reset() {
		mlp = null;
		maxIterations = 0;
		maxError = 0.0;
		eta = 0.0;
		type = null;
		layers[0] = 1;
	}

	public boolean setNumInputNeurons(int numNeurons) {
		if (isTrained() && layers[0] != numNeurons)
			return false;
		layers[0] = numNeurons;
		return true;
	}

	public boolean setNumOutputNeurons(int numNeurons) {
		if (isTrained() && layers[layers.length - 1] != numNeurons)
			return false;
		layers[layers.length - 1] = numNeurons;
		return true;
	}

	public boolean setDataType(Data.Type type) {
		if (isTrained() && this.type != type)
			return false;
		this.type = type;
		return true;
	}

	public void resetIterations() {
		initMLP();
		mlp.resetIterations();
	}

	/**
	 * This function performs the online calculation with the the Network
	 * 
	 * 
	 * @param dataCollection
	 *            A collection of the type dvv.Data with input- and targetvalues
	 * 
	 * @param eta
	 *            The learning rate to be used.
	 * @throws BadConfigException
	 */
	public double runOnline(Collection<Data> dataCollection, double eta,
			double momentum) throws BadConfigException {
		initMLP();
		return mlp.runOnline(dataCollection, eta, momentum);
	}

	/**
	 * This function performs the batch calculation with the Network
	 * 
	 * @param dataCollection
	 *            A collection of the type dvv.Data with input- and targetvalues
	 * 
	 * @param eta
	 *            The learning rate to be used.
	 * @return den Testfehler. In case of an error returns 0.
	 */
	public double runBatch(Collection<Data> dataCollection, int batchSize,
			double eta, double momentum) {
		initMLP();
		return mlp.runBatch(dataCollection, batchSize, eta, momentum);
	}

	/**
	 * This method performs the test using delivered data.
	 * 
	 * @param dataCollection
	 *            The data, tu be used for the test
	 * @return The test error. If an error occurse, returns 0.
	 */
	public double runTest(Collection<Data> dataCollection) {
		return mlp.runTest(dataCollection);
	}

	/**
	 * This method starts a testrun.
	 * 
	 * @return the output of neurons in percents.
	 */
	public double[] classify(double[] input) {
		initMLP();
		return mlp.classify(input);
	}

	/**
	 * Returns the number of layers in the neuronal network, including input and
	 * output layers.
	 * 
	 * @return Number of layers.
	 */
	public int getNumLayers() {
		return layers.length;
	}

	/**
	 * Returns the size (number of neurons) of a layer. If layer is 0 and
	 * isTrained() returns false, this function will return 1.
	 * 
	 * @param layer
	 *            the index of the layer
	 * @return number of neurons in the layer
	 */
	public int getLayerSize(int layer) {
		return layers[layer];
	}

	/**
	 * Returns the weight matrix between two layers. If layer is 1 and isTrained
	 * returns false, this function will return null. If layer is 0, this
	 * function will return null.
	 * 
	 * @param layer
	 *            the index of the second layer (must be >= 1)
	 * @return the weight matrix of type double[][], where the first dimension
	 *         represents the number of neurons in the second layer and the
	 *         second- in the first one.
	 * 
	 */
	public double[][] getWeights(int layer) {
		if (mlp != null)
			return mlp.getWeights(layer);
		if (layer != 0) {
			double[][] result = new double[layers[layer]][layers[layer - 1]];
			for (int i = 0; i < result.length; i++)
				java.util.Arrays.fill(result[i], 0.0);
			return result;
		} else
			return null;
	}

	/**
	 * Returns the activation function of the layer. If layer is 0, this
	 * function will return null.
	 * 
	 * @param layer
	 *            the index of the layer
	 * @return the activation function of the layer
	 */
	public ActivationFunction getActivationFunction(int layer) {
		if (layer != 0)
			return activations[layer];
		else
			return null;
	}

	/**
	 * Returns the bias. If layer is 0, this function will return 0.0 .
	 * 
	 * @param layer
	 *            the index of the layer
	 * @return the bias
	 */
	public double getBias(int layer) {
		if (layer != 0)
			return biases[layer];
		else
			return 0.0;
	}

	/**
	 * Returns a readable name of NeuralNetwork
	 * 
	 * @return name of the NeuralNetwork
	 */
	public String getName() {
		return name;
	}

	/**
	 * Returns true if the network has been trained.
	 * 
	 * @return true if this network has been trained; false otherwise
	 */
	public boolean isTrained() {
		return mlp != null;
	}

	/**
	 * Sets a readable name for the NeuralNetwork
	 * 
	 * @param name
	 *            new name of the NeuralNetwork
	 */
	public void setName(String name) {
		this.name = name;
	}

	/**
	 * Sets the number of layers, including input and output layers. If
	 * this.isTrained() returns true, this function does nothing. If numLayers <=
	 * 0, behavior is undefined.
	 * 
	 * @param numLayers
	 *            the number of layers
	 * @return the same as !this.isTrained()
	 */
	public boolean setnumLayers(int numLayers) {
		if (!isTrained()) {
			int[] newLayers = new int[numLayers];
			double[] newBias = new double[numLayers];
			ActivationFunction[] newActivations = new ActivationFunction[numLayers];
			final int min = numLayers < layers.length ? numLayers
					: layers.length;
			for (int i = 0; i < min; i++) {
				newLayers[i] = layers[i];
				newBias[i] = biases[i];
				newActivations[i] = activations[i];
			}
			for (int i = min; i < newLayers.length; i++) {
				newLayers[i] = newLayers[min - 1];
				newBias[i] = newBias[min - 1];
				newActivations[i] = newActivations[min - 1];
			}
			layers = newLayers;
			biases = newBias;
			activations = newActivations;
		}
		return !isTrained();
	}

	/**
	 * Sets the size of a layer. If this.isTrained() returns true, this function
	 * does nothing. If layer is 0, this function does nothing. If size <= 0,
	 * behavior is undefined.
	 * 
	 * @param layer
	 *            the index of the layer
	 * @param size
	 *            the number of neurons
	 * @return the same as !this.isTrained()
	 */
	public boolean setLayerSize(int layer, int size) {
		if (!isTrained() && layer != 0 && layer < layers.length)
			layers[layer] = size;
		return !isTrained();
	}

	/**
	 * Sets the activation function of a layer. If this.isTrained() returns
	 * true, this function does nothing. If layer is 0, this function does
	 * nothing. If activationFunction == null, behavior is undefined.
	 * 
	 * @param layer
	 *            the index of the layer
	 * @param activationFunction
	 *            the activation function
	 * @return the same as !this.isTrained()
	 */
	public boolean setActivationFunction(int layer,
			ActivationFunction activationFunction) {
		if (!isTrained() && layer != 0 && layer < activations.length)
			activations[layer] = activationFunction;
		return !isTrained();
	}

	/**
	 * Sets the bias of a layer. If this.isTrained() returns true, this function
	 * does nothing. If layer is 0, this function does nothing.
	 * 
	 * @param layer
	 *            the index of the layer
	 * @param bias
	 *            the bias
	 * @return the same as !this.isTrained()
	 */
	public boolean setBias(int layer, double bias) {
		if (!isTrained() && layer != 0 && layer < biases.length)
			biases[layer] = bias;
		return !isTrained();
	}

	private void initMLP() {
		if (!isTrained()) {
			int[] newLayers = new int[layers.length - 2];
			double[] newBias = new double[layers.length - 2];
			for (int i = 0; i < newLayers.length; i++) {
				newLayers[i] = layers[i + 1];
				newBias[i] = biases[i + 1];
			}
			try {
				mlp = new MLP(layers[0], layers[layers.length - 1], newLayers,
						activations, newBias);
				if (maxIterations > 0 && maxError > 0 && eta > 0)
					mlp.makeAutoencoder(maxIterations, maxError, eta);
			} catch (BadConfigException e) {
				e.printStackTrace();
			}
		}
	}

	public void setAutoencoder(int maxIterations, double maxError, double eta) {
		this.maxIterations = maxIterations;
		this.maxError = maxError;
		this.eta = eta;
	}

}
