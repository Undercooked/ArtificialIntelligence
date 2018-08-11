using ArtificialIntelligence.Models;

namespace ArtificialIntelligence
{
	public interface IModelBreeder
	{
		FullyConnectedNeuralNetworkModel Breed(FullyConnectedNeuralNetworkModel mother, FullyConnectedNeuralNetworkModel father);
	}
}
