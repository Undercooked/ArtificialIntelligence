using System.Collections.Generic;
using System.Linq;
using ArtificialIntelligence.Models;
using ArtificialIntelligence.RandomNumberServices;

namespace ArtificialIntelligence.Genetic
{
	public class MonogamousPopulationBreeder : IPopulationBreeder
	{
		private readonly IModelBreeder modelBreeder;
		private readonly ThreadSafeRandom random;

		public MonogamousPopulationBreeder(IModelBreeder modelBreeder, ThreadSafeRandom random)
		{
			this.modelBreeder = modelBreeder;
			this.random = random;
		}

		public FullyConnectedNeuralNetworkModel[] CreateNextGeneration(FullyConnectedNeuralNetworkModel[] parents, int populationSize)
		{
			var children = new List<FullyConnectedNeuralNetworkModel>();
			var remainingParentIndexes = Enumerable.Range(0, parents.Length).ToList();

			while (children.Count < populationSize - parents.Length)
			{
				var randomRemainingParentIndex = random.Next(1, remainingParentIndexes.Count);
				var motherIndex = remainingParentIndexes[0];
				var fatherIndex = remainingParentIndexes[randomRemainingParentIndex];
				var child = modelBreeder.Breed(parents[motherIndex], parents[fatherIndex]);

				children.Add(child);
				remainingParentIndexes.Remove(motherIndex);
				remainingParentIndexes.Remove(fatherIndex);
			}

			return children.ToArray();
		}
	}
}
