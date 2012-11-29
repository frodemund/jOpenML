package org.jopenml.mlp;

import java.io.Serializable;
import java.util.Collection;

import org.jopenml.mlp.Data;
/**
 * This class represents the MLP and makes heavy use of the class {@link Layer}.
 */
public class MLP implements Serializable {

	private static final long serialVersionUID = -5212835785366190139L;

	private Layer[] layer; 
	int iterations = 0;

	/**
	 * Builds the Network and sets all necessary variables.
	 * 
	 * @param inputNeurons
	 * @param outputNeurons
     * @param hiddenLayers array with hidden layers.
     * @param functions array with activation functions.
     * @param bias Array with biases.
	 */
	public MLP(int inputNeurons, int outputNeurons, int[] hiddenLayers, ActivationFunction[] functions, double[] bias) throws BadConfigException {

		// Tests the configuration
		if (inputNeurons < 1)
			throw new BadConfigException(
					"Invalid number of neurons in the input layer!",
					BadConfigException.INVALID_INPUT_LAYER_DIMENSION);
		if (outputNeurons < 1)
			throw new BadConfigException(
					"Invalid number of neurons in the output layer!",
					BadConfigException.INVALID_OUTPUT_LAYER_DIMENSION);
		if (functions.length != hiddenLayers.length + 2)
			throw new BadConfigException(
					"The number of activiation functions is not equal to the number of layers!",
					BadConfigException.INVALID_NUMBER_OF_FUNCTIONS);
		if (bias.length != hiddenLayers.length)
			throw new BadConfigException(
					"The number of bias values is not equal to the number of layers!",
					BadConfigException.INVALID_NUMBER_OF_BIAS);

		// Creates the input layer
		layer = new Layer[hiddenLayers.length + 2];
		layer[0] = new Layer(null, inputNeurons, functions[0], 0);

		// creates  hidden layers
		for (int i = 0; i < hiddenLayers.length; i++) {
			layer[i + 1] = new Layer(layer[i], hiddenLayers[i], functions[i],
					bias[i]);
		}

		// creates the outputlayer
		layer[layer.length - 1] = new Layer(layer[layer.length - 2],
				outputNeurons, functions[functions.length - 1], 0);

	}

	/**
	 * This function trains the MLP as autoencoder by using the function {@link Layer#makeAutoencoder(double, int, double, double) makeAutoencoder}
	 * @param maxIterations Break condition for the training. Training will stop if maxIterations is reached.
	 * @param maxError If the net's error falls below this border, training will be aborted.
	 * @param eta the learning rate
	 * @return false if training failes, otherwise true.
	 */
	public boolean makeAutoencoder(int maxIterations, double maxError, double eta) {
		try {
			layer[layer.length - 1].makeAutoencoder(Math.random(), maxIterations, maxError, eta);
			return true;
		} catch (BadConfigException e) {
			return false;
		}
	}
	
	/**
	 * This function performs the online calculation with the the network.
	 *  
	 * @param dataCollection A collection of the type {@link Data} with input and target values.
	 *            
	 * @param eta The learning rate to be used.
	 * @throws BadConfigException
	 */
	public double runOnline(Collection<Data> dataCollection, double eta, double momentum) {
		if (layer == null)
			return 0;
		
		Layer outLayer = layer[layer.length - 1];
		ActivationFunction outAVF = outLayer.getActivationFunction();
		int outLayerSize = outLayer.getSize();

		double[] errVec = new double[outLayerSize];

		for (Data theData : dataCollection) {
			// get the index where the target value is 1
			int target = theData.getTarget(); 

			// Sets the input data
			if(!layer[0].setInput(theData.getData()))
				System.out.println("Can't set input data!");

			// Calculate the output
			double[] out = outLayer.getOutput();

			// Calculates the error of the output layer
			for (int h = 0; h < outLayerSize; h++)
				if(h == target)
					errVec[h] = (out[h] - 1) * outAVF.derivation(outLayer.layerInput[h]);
				else
					errVec[h] = out[h] * outAVF.derivation(outLayer.layerInput[h]);

			// Error backpropagation 
			/*System.out.println("Mittlerer Gadient: " + */
			outLayer.backPropagate(errVec)
			/* / layer.length)*/;
			
			// Adjust the weights
			if(momentum > 0)
				outLayer.update(eta, momentum);
			else
				outLayer.update(eta);

		}

		return runTest(dataCollection);
	}

	/**This function performs the batch calculation  with the Network
	 * 
 	 * @param dataCollection A collection of the type {@link Data}a with input and target values.
	 *            
	 * @param eta The learning rate to be used.
	 * @return The Training error computed by the function {@link #runTest(Collection)}. In case of an  error returns 0.
	 */
	
	public double runBatch(Collection<Data> dataCollection, int batchSize, double eta, double momentum) {
		if (layer == null)
			return 0;

		Layer outLayer = layer[layer.length - 1];
		ActivationFunction outAVF = outLayer.getActivationFunction();
		int outLayerSize = outLayer.getSize();

		double[] errVec = new double[outLayerSize];
		
		for (Data theData : dataCollection) {
			// get the index where the target value is 1
			int target = theData.getTarget(); 

			// Sets the input data
			if(!layer[0].setInput(theData.getData()))
				System.out.println("Can't set input data!");

			// Calculate the output
			double[] out = outLayer.getOutput();

			// Calculates the error of the output layer
			for (int h = 0; h < outLayerSize; h++)
				if(h == target)
					errVec[h] = (out[h] - 1) * outAVF.derivation(outLayer.layerInput[h]);
				else
					errVec[h] = out[h] * outAVF.derivation(outLayer.layerInput[h]);

			// Error backpropagation 
			System.out.println("Mittlerer Gadient: " + outLayer.backPropagate(errVec) / layer.length);
			iterations++;

			if(iterations % batchSize == 0 ) {
				// Adjust the weights
				if(momentum > 0)
					outLayer.update(eta, momentum);
				else
					outLayer.update(eta);
			}

		}
		return runTest(dataCollection);
	}
	

	/**
	 * This method performs a test using delivered data.
	 * 
	 * @param dataCollection The data, tu be used for the test.
	 * @return The test error. If an error occurse, returns 0.
	 */
	public double runTest(Collection<Data> dataCollection) {
		if (layer == null)
			return 0;

		Layer outLayer = layer[layer.length - 1];
		ActivationFunction outAVF = outLayer.getActivationFunction();
		int outLayerSize = outLayer.getSize();
		
		double err = 0;
		
		for (Data theData : dataCollection) {
			// get the index where the target value is 1
			int target = theData.getTarget(); 

			// Sets the input data
			if (!layer[0].setInput(theData.getData()))
				return 0;

			// Calculate the output
			double[] out = outLayer.getOutput();

			// Calculates the error of the output layer
			for (int h = 0; h < outLayerSize; h++) {
				double v;
				if(h == target)
					v = (out[h] - 1) * outAVF.derivation(outLayer.layerInput[h]);
				else
					v = out[h] * outAVF.derivation(outLayer.layerInput[h]);

				err += v * v;
			}
			
		}

		return (0.5 * err) / dataCollection.size();
	}

	/**
	 * This method is used to classify a given input vector.
	 * @param input The input to classify with a dimension equal to the output neurons.
	 * @return The output of neurons in percents.
	 */
	public double[] classify(double[] input) {
		layer[0].setInput(input);
		double[] netOutput = layer[layer.length - 1].getOutput();
		double[] retVal = new double[netOutput.length];

		for(int i = 0; i < retVal.length; i++)
			retVal[i] = Math.exp(10 * netOutput[i]);
		// Sum up
		double sum = 0;
		for(double v : retVal) sum += v;
		
		for(int i = 0; i < netOutput.length; i++)
			retVal[i] = retVal[i] *  100/ sum;

		return retVal;
	}

	/**
	 * String-representation of the neuronal network
	 * @return An ascii presentation of the network's weights.
	 */
	public String toString() {
		StringBuffer buffer = new StringBuffer();
		for (int i = 0; i < layer.length; i++) {
			buffer.append("Layer " + i);
			buffer.append("\n" + layer[i].toString() + "\n");
		}
		return buffer.toString();
	}

	/**
	 * Returns the {@link ActivationFunction} of the given layer.
	 * @param layer Index of the requested layer.
	 * @return The layer's ActivationFunction.
	 */
	public ActivationFunction getActivationFunction(int layer) {
		if (layer > (this.layer.length - 1))
			return null;

		return this.layer[layer].getActivationFunction();
	}

	/**
	 * Returns the bias of the given {@link Layer}.
	 * @param layer Index of the requested layer.
	 * @return The layer's bias.
	 */
	public double getBias(int layer) {
		if (layer > (this.layer.length - 1))
			return 0;

		return this.layer[layer].getBias();
	}

	/**
	 * Returns the size of the given {@link Layer}.
	 * @param layer Index of the requested layer.
	 * @return The layer's size.
	 */
	public int getLayerSize(int layer) {
		if (layer > (this.layer.length - 1))
			return -1;

		return this.layer[layer].getSize();
	}

	/**
	 * Returns the size of the MLP.
	 * @return The size of the MLP.
	 */
	public int getNumLayers() {
		return this.layer.length;
	}
	
	/**
	 *Returns the WeightMatrix of the {@link Layer}.  
	 * @param layer The layer's index.
	 * @return The layer's WeightMatrix.
	 */
	public double[][] getWeights(int layer) {
		if (layer > (this.layer.length - 1))
			return null;

		return this.layer[layer].getWeightMatrix();
	}

	/**
	 * This function resets the integrated iteration-counter.
	 */
	public void resetIterations(){
		iterations = 0;
	}
}
