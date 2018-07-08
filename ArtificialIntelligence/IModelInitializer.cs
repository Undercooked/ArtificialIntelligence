using System;
using ArtificialIntelligence.Enums;
using ArtificialIntelligence.Models;

namespace ArtificialIntelligence
{
	public interface IModelInitializer
	{
		FullyConnectedNeuralNetworkModel CreateModel(ActivationFunction activationFunction, int[] activationCountsPerLayer, Random random);
	}
}
