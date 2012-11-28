package org.jopenml.mlp;

/** This interface represents a {@link MLP Multilayer Perceptron} and is implemented by {@link MLPStub}.
 *  The general contract for all functions is as follows:
 *
 *  1. {@link Layer}s are numbered 0 through n-1 when there are n layers.
 *     The function {@link #getNumLayers()}returns n.
 *     Layers 0 and n-1 are input and output layers, respectively.
 *
 *  2. If for any function which requires the index of a {@link Layer},
 *     this number is not in the range 0 to n-1 (both inclusive), the behavior is undefined.
 */
public interface NeuralNetwork {

	/**
	 * Returns the number of layers in the neuronal network, including input and output layers.
	 * 
	 * @return Number of layers.
	 */
	public int getNumLayers();

	/**
	 * Returns the size (number of neurons) of a {@link Layer}.
	 * If layer is 0 and isTrained() returns false, this function will return 1.
	 * 
	 * @param layer the index of the {@link Layer}
	 * @return number of neurons in the {@link Layer}
	 */
	public int getLayerSize(int layer);

	/**
	 * Returns the weight matrix between two {@link Layer}s.
	 * If layer is 1 and isTrained returns false, this function will return null.
	 * If layer is 0, this function will return null.
	 * 
	 * @param layer the index of the second layer (must be >= 1)
	 * @return the weight matrix of type double[][], where the first dimension represents
	 *         the number of neurons in the second {@link Layer} and the second- in the first one.
	 *         
	 */
	public double[][] getWeights(int layer);

	/**
	 * Returns the {@link ActivationFunction activation function} of the {@link Layer}.
	 * If layer is 0, this function will return null.
	 * 
	 * @param layer the index of the {@link Layer}
	 * @return the {@link ActivationFunction activation function} of the {@link Layer}
	 */
	public ActivationFunction getActivationFunction(int layer);

	/**
	 * Returns the bias.
	 * If layer is 0, this function will return 0.0 .
	 * 
	 * @param layer the index of the layer
	 * @return the bias
	 */
	public double getBias(int layer);

	/**
	 * Returns a readable name of NeuralNetwork
	 * 
	 * @return name of the NeuralNetwork
	 */
	public String getName();

	/**
	 * Returns true if the network has been trained.
	 * 
	 * @return true if this network has been trained; false otherwise
	 */
	public boolean isTrained();

	/**
	 * Sets a readable name for the NeuralNetwork
	 * 
	 * @param name new name of the NeuralNetwork
	 */
	public void setName(String name);

	/**
	 * Sets the number of layers, including input and output {@link Layer}s.
	 * If this.isTrained() returns true, this function does nothing.
	 * If numLayers <= 0, behavior is undefined.
	 *  
	 * @param numLayers the number of layers
	 * @return the same as !this.isTrained()
	 */
	public boolean setnumLayers(int numLayers);

	/**
	 * Sets the size of a {@link Layer}.
	 * If this.isTrained() returns true, this function does nothing.
	 * If layer is 0, this function does nothing.
	 * If size <= 0, behavior is undefined.
	 * 
	 * @param layer the index of the {@link Layer}
	 * @param size the number of neurons
	 * @return the same as !this.isTrained()
	 */
	public boolean setLayerSize(int layer, int size);

	/**
	 * Sets the activation function of a {@link Layer}.
	 * If this.isTrained() returns true, this function does nothing.
	 * If layer is 0, this function does nothing.
	 * If activationFunction == null, behavior is undefined.
	 *
	 * @param layer the index of the {@link Layer}
	 * @param activationFunction the activation function
	 * @return the same as !this.isTrained()
	 */
	public boolean setActivationFunction(int layer, ActivationFunction activationFunction);

	/**
	 * Sets the bias of a {@link Layer}.
	 * If this.isTrained() returns true, this function does nothing.
	 * If layer is 0, this function does nothing.
	 *
	 * @param layer the index of the {@link Layer}
	 * @param bias the bias
	 * @return the same as !this.isTrained()
	 */
	public boolean setBias(int layer, double bias);

	/**
	 * Deletes the current (trained) {@link MLP}, but keeps the configuration.
	 *
	 * If there is no {@link MLP}, this function does nothing.
	 */
	public void reset();

}
