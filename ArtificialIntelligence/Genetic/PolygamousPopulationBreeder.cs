using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using ArtificialIntelligence.Models;
using ArtificialIntelligence.RandomNumberServices;

namespace ArtificialIntelligence.Genetic
{
	public class PolygamousPopulationBreeder : IPopulationBreeder
	{
		private readonly IModelBreeder modelBreeder;
		private readonly ThreadSafeRandom random;

		public PolygamousPopulationBreeder(IModelBreeder modelBreeder, ThreadSafeRandom random)
		{
			this.modelBreeder = modelBreeder;
			this.random = random;
		}

		public FullyConnectedNeuralNetworkModel[] CreateNextGeneration(FullyConnectedNeuralNetworkModel[] parents, int populationSize)
		{
			var children = new FullyConnectedNeuralNetworkModel[populationSize - parents.Length];
			var indexesBred = new ConcurrentDictionary<int, List<int>>();

			Parallel.For(0, children.Length, childIndex =>
			{
				var motherIndex = 0;
				var fatherIndex = 0;
				var motherHasChildren = false;

				while (motherIndex == fatherIndex || motherHasChildren && indexesBred[motherIndex].Contains(fatherIndex))
				{
					var parentIndexes = new[] { random.Next(parents.Length), random.Next(parents.Length) }.OrderBy(i => i);

					motherIndex = parentIndexes.ElementAt(0);
					fatherIndex = parentIndexes.ElementAt(1);
					motherHasChildren = indexesBred.ContainsKey(motherIndex);
				}

				children[childIndex] = modelBreeder.Breed(parents[motherIndex], parents[fatherIndex]);

				indexesBred.AddOrUpdate(motherIndex, new List<int>() { fatherIndex }, (key, value) =>
				{
					value.Add(fatherIndex);

					return value;
				});
			});

			return children;
		}
	}
}
