using ArtificialIntelligence.Models;

namespace ArtificialIntelligence
{
	public interface IModelExecuter
	{
		double[][] Execute(FullyConnectedNeuralNetworkModel model, double[] activations);
	}
}
