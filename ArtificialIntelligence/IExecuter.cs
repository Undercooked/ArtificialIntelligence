using ArtificialIntelligence.Models;

namespace ArtificialIntelligence
{
	public interface IExecuter
	{
		double[][] Execute(FullyConnectedNeuralNetworkModel model, double[] activations);
	}
}
