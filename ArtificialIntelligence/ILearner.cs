using ArtificialIntelligence.Models;

namespace ArtificialIntelligence
{
	public interface ILearner
	{
		FullyConnectedNeuralNetworkModel Model { get; }

		void Initialize(FullyConnectedNeuralNetworkModel model);
		void Learn(InputOutputPairModel[] batch);
	}
}
