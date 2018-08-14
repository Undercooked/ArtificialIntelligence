using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using ArtificialIntelligence.Models;

namespace ArtificialIntelligence.Genetic
{
	public class GeneticLearner : ILearner
	{
		private readonly int populationSize;
		private readonly int selectionSize;
		private readonly Random random;
		private readonly IModelInitializer modelInitializer;
		private readonly IModelExecuter executer;
		private readonly IModelBreeder modelBreeder;
		private readonly object populationCostLock = new object();

		private CostModel<FullyConnectedNeuralNetworkModel>[] population;
		private bool isFirstLearningIteration = true;

		public FullyConnectedNeuralNetworkModel Model => population[0].Model;

		public GeneticLearner(int populationSize, int selectionSize, Random random, IModelInitializer modelInitializer, IModelExecuter executer, IModelBreeder modelBreeder)
		{
			this.populationSize = populationSize;
			this.selectionSize = selectionSize;
			this.random = random;
			this.modelInitializer = modelInitializer;
			this.executer = executer;
			this.modelBreeder = modelBreeder;
			population = new CostModel<FullyConnectedNeuralNetworkModel>[populationSize];
		}

		public void Initialize(FullyConnectedNeuralNetworkModel model)
		{
			population[0] = new CostModel<FullyConnectedNeuralNetworkModel>(model);

			for (var populationIndex = 1; populationIndex < populationSize; populationIndex++)
			{
				var member = modelInitializer.CreateModel(model.ActivationCountsPerLayer, model.ActivationFunction, random);
				population[populationIndex] = new CostModel<FullyConnectedNeuralNetworkModel>(member);
			}
		}

		public void Learn(InputOutputPairModel[] batch)
		{
			Parallel.ForEach(batch, inputOutputPair =>
			{
				CalculateCostForPopulation(inputOutputPair);
				SortPopulation();
				CreateNextGeneration();
			});

			isFirstLearningIteration = false;
		}

		private void CalculateCostForPopulation(InputOutputPairModel inputOutputPair)
		{
			var individualIndex = isFirstLearningIteration ? 0 : selectionSize;

			for (; individualIndex < population.Length; individualIndex++)
			{
				var model = population[individualIndex].Model;
				var allActivations = executer.Execute(model, inputOutputPair.Inputs);
				var cost = CalculateCost(inputOutputPair.Outputs, allActivations.Last());

				lock (populationCostLock)
				{
					population[individualIndex].Cost += cost;
				}
			}
		}

		private double CalculateCost(double[] expectedOutputs, double[] actualOutputs)
		{
			var totalCost = 0.0;

			for (var j = 0; j < actualOutputs.Length; j++)
			{
				var outputDelta = expectedOutputs[j] - actualOutputs[j];
				var cost = Math.Pow(outputDelta, 2);

				totalCost += cost;
			}

			return totalCost;
		}

		private void SortPopulation()
		{
			Array.Sort(population, (x, y) => x.Cost.CompareTo(y.Cost));
		}

		private void CreateNextGeneration()
		{
			var parents = SelectSurvivors();
			var indexesBred = new ConcurrentDictionary<int, List<int>>();

			Parallel.For(parents.Length, population.Length, childIndex =>
			{
				var parentIndexes = new[] { random.Next(selectionSize), random.Next(selectionSize) }.OrderBy(i => i);
				var motherIndex = parentIndexes.ElementAt(0);
				var fatherIndex = parentIndexes.ElementAt(1);
				var motherHasChildren = indexesBred.ContainsKey(motherIndex);

				if (motherIndex != fatherIndex && (!motherHasChildren || indexesBred[motherIndex].Contains(fatherIndex)))
				{
					var mother = parents[motherIndex].Model;
					var father = parents[fatherIndex].Model;
					var child = modelBreeder.Breed(mother, father);

					population[childIndex] = new CostModel<FullyConnectedNeuralNetworkModel>(child);

					indexesBred.AddOrUpdate(motherIndex, new List<int>() { fatherIndex }, (key, value) =>
					{
						value.Add(fatherIndex);

						return value;
					});
				}
			});

			Array.Copy(parents, population, parents.Length);
		}

		private CostModel<FullyConnectedNeuralNetworkModel>[] SelectSurvivors()
		{
			var topSelectionSize = (int)Math.Round(selectionSize * 0.7);
			var randomSelectionSize = selectionSize - topSelectionSize;
			var selection = new CostModel<FullyConnectedNeuralNetworkModel>[selectionSize];
			var uniqueRandomIndexes = CreateUniqueRandomIntegers(topSelectionSize, population.Length, randomSelectionSize);

			for (var populationIndex = 0; populationIndex < topSelectionSize; populationIndex++)
			{
				var model = population[populationIndex].Model;
				selection[populationIndex] = new CostModel<FullyConnectedNeuralNetworkModel>(model);
			}

			for (var populationIndex = topSelectionSize; populationIndex < selection.Length; populationIndex++)
			{
				var selectedMemberIndex = uniqueRandomIndexes[populationIndex - topSelectionSize];
				var model = population[selectedMemberIndex].Model;
				selection[populationIndex] = new CostModel<FullyConnectedNeuralNetworkModel>(model);
			}

			return selection.ToArray();
		}

		private List<int> CreateUniqueRandomIntegers(int minValue, int maxValue, int count)
		{
			var uniqueRandomIntegers = new List<int>();

			while (uniqueRandomIntegers.Count < count)
			{
				var randomInt = random.Next(minValue, maxValue);

				if (!uniqueRandomIntegers.Contains(randomInt))
				{
					uniqueRandomIntegers.Add(randomInt);
				}
			}

			return uniqueRandomIntegers;
		}
	}
}
