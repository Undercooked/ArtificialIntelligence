using ArtificialIntelligence.Models;

namespace ArtificialIntelligence
{
	public interface IExecuter
	{
		float[][] Execute(FullyConnectedNeuralNetworkModel model, float[] activations);
	}
}
