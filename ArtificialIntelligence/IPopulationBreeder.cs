using ArtificialIntelligence.Models;

namespace ArtificialIntelligence
{
	public interface IPopulationBreeder
	{
		FullyConnectedNeuralNetworkModel[] CreateNextGeneration(FullyConnectedNeuralNetworkModel[] parents, int populationSize);
	}
}
