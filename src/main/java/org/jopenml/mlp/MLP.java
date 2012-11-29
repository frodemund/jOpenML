package org.jopenml.mlp;

import java.io.Serializable;
import java.util.Collection;
import java.util.List;

import org.jopenml.mlp.layers.Layer;

/**
 * This class represents the MLP and makes heavy use of the class {@link Layer}.
 */
public class MLP
		implements Serializable {
	
	private static final long serialVersionUID = -5212835785366190139L;
	private final Layer inputLayer;
	private final Layer outputLayer;
	
	public MLP(List<Layer> layers) {
		inputLayer = layers.get(0);
		outputLayer = layers.get(layers.size() - 1);
	}
	
	// /**
	// * This function trains the MLP as autoencoder by using the function
	// * {@link Layer#makeAutoencoder(double, int, double, double) makeAutoencoder}
	// *
	// * @param maxIterations Break condition for the training. Training will stop if maxIterations is reached.
	// * @param maxError If the net's error falls below this border, training will be aborted.
	// * @param eta the learning rate
	// */
	// public void makeAutoencoder(int maxIterations, double maxError, double eta) {
	// outputLayer.makeAutoencoder(Math.random(), maxIterations, maxError, eta);
	// }
	
	/**
	 * This function performs the online calculation with the the network.
	 * 
	 * @param dataCollection A collection of the type {@link Datum} with input and target values.
	 * @param eta The learning rate to be used.
	 */
	public double runOnline(Collection<Datum> dataCollection, double eta, double momentum) {
		
		final double[] errVec = new double[outputLayer.getSize()];
		
		for (final Datum theData : dataCollection) {
			final double[] targetValues = theData.getTarget();
			
			// provide data
			inputLayer.setInput(theData.getData());
			
			// Calculate the output
			final double[] out = outputLayer.getOutput();
			
			// Calculates the error of the output layer
			for (int h = 0; h < out.length; h++) {
				errVec[h] = (out[h] - targetValues[h])
						* outputLayer.getActivationFunction().derivation(outputLayer.getLayerInput()[h]);
			}
			
			// Error backpropagation
			outputLayer.backPropagate(errVec);
			
			// Adjust the weights
			if (momentum > 0) {
				outputLayer.update(eta, momentum);
			} else {
				outputLayer.update(eta);
			}
			
		}
		
		return runTest(dataCollection);
	}
	
	/**
	 * This function performs the batch calculation with the Network
	 * 
	 * @param dataCollection A collection of the type {@link Datum}a with input and target values.
	 * @param eta The learning rate to be used.
	 * @return The Training error computed by the function {@link #runTest(Collection)}. In case of an error returns 0.
	 */
	
	public double runBatch(Collection<Datum> dataCollection, int batchSize, double eta, double momentum) {
		
		final double[] errVec = new double[outputLayer.getSize()];
		int iterations = 0;
		
		for (final Datum theData : dataCollection) {
			final double[] targetValues = theData.getTarget();
			
			// provide data
			inputLayer.setInput(theData.getData());
			
			// Calculate the output
			final double[] out = outputLayer.getOutput();
			
			// Calculates the error of the output layer
			for (int h = 0; h < out.length; h++) {
				errVec[h] = (out[h] - targetValues[h])
						* outputLayer.getActivationFunction().derivation(outputLayer.getLayerInput()[h]);
			}
			
			if (++iterations % batchSize == 0) {
				// Adjust the weights
				if (momentum > 0) {
					outputLayer.update(eta, momentum);
				} else {
					outputLayer.update(eta);
				}
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
	public double runTest(Collection<Datum> dataCollection) {
		double err = 0;
		for (final Datum datum : dataCollection) {
			// get the index where the target value is 1
			final double[] targetValues = datum.getTarget();
			
			// provide data
			inputLayer.setInput(datum.getData());
			
			// Calculate the output
			final double[] out = outputLayer.getOutput();
			
			// Calculates the error of the output layer
			for (int h = 0; h < out.length; h++) {
				double v;
				v = (out[h] - targetValues[h])
						* outputLayer.getActivationFunction().derivation(outputLayer.getLayerInput()[h]);
				err += v * v;
			}
			
		}
		
		return (0.5 * err) / dataCollection.size();
	}
	
	/**
	 * This method is used to classify a given input vector.
	 * 
	 * @param input The input to classify with a dimension equal to the output neurons.
	 * @return The output of neurons in percents.
	 */
	public double[] classify(double[] input) {
		inputLayer.setInput(input);
		return outputLayer.getOutput();
	}
	
	/**
	 * String-representation of the neuronal network
	 * 
	 * @return An ascii presentation of the network's weights.
	 */
	@Override
	public String toString() {
		// TODO
		return super.toString();
	}
}
