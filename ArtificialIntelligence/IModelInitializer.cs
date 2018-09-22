using ArtificialIntelligence.Enums;
using ArtificialIntelligence.Models;

namespace ArtificialIntelligence
{
	public interface IModelInitializer
	{
		FullyConnectedNeuralNetworkModel CreateModel(int[] activationCountsPerLayer, ActivationFunction activationFunction);
		FullyConnectedNeuralNetworkModel CreateModel(int[] activationCountsPerLayer, ActivationFunction activationFunction, double[][] biasLayers, double[][,] weightLayers);
	}
}
