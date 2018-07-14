using ArtificialIntelligence.Models;

namespace ArtificialIntelligence
{
	public interface ILearner
    {
		FullyConnectedNeuralNetworkModel Learn(FullyConnectedNeuralNetworkModel model, InputOutputPairModel[] batch);
	}
}
