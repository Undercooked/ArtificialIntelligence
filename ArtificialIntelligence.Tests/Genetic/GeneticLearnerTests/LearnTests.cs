using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using ArtificialIntelligence.Enums;
using ArtificialIntelligence.Genetic;
using ArtificialIntelligence.Models;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Moq;

namespace ArtificialIntelligence.Tests.Genetic.GeneticLearnerTests
{
	[TestClass]
	public class LearnTests
	{
		private const int populationSize = 3;
		private const int selectionSize = 2;
		private static readonly int[] activationCountsPerLayer = new[] { 4, 3 };
		private static readonly ActivationFunction activationFunction = ActivationFunction.Sigmoid;

		private Random random;
		private Mock<IModelInitializer> mockModelInitializer;
		private Mock<IModelExecuter> mockModelExecuter;
		private Mock<IModelBreeder> mockModelBreeder;
		private List<FullyConnectedNeuralNetworkModel> population;
		private GeneticLearner sut;

		[TestInitialize]
		public void TestInitialize()
		{
			random = new Random();
			mockModelInitializer = new Mock<IModelInitializer>(MockBehavior.Strict);
			mockModelExecuter = new Mock<IModelExecuter>(MockBehavior.Strict);
			mockModelBreeder = new Mock<IModelBreeder>(MockBehavior.Strict);
			population = new List<FullyConnectedNeuralNetworkModel>();

			sut = new GeneticLearner(populationSize, selectionSize, random, mockModelInitializer.Object, mockModelExecuter.Object, mockModelBreeder.Object);

			mockModelInitializer.Setup(m => m.CreateModel(It.Is<int[]>(it => it != null && activationCountsPerLayer.SequenceEqual(it)), activationFunction, random))
				.Returns(() =>
				{
					var model = CreateFullyConnectedNeuralNetworkModel();
					population.Add(model);

					return model;
				});
		}

		[TestMethod]
		public void FirstIterationOfLearnPerformsCorrectCalls()
		{
			// Arrange
			var model = CreateFullyConnectedNeuralNetworkModel();
			var batch = CreateBatches();

			for (var i = 0; i < populationSize; i++)
			{
				mockModelExecuter.Setup(m => m.Execute(It.Is<FullyConnectedNeuralNetworkModel>(it => it.Equals(population[i])), batch[i].Inputs))
					.Returns(activationCountsPerLayer.Select(l => CreateAndInitializeDoubleArray(l, random.NextDouble)).ToArray());
			}

			// TODO: setup mockModelBreeder.Breed

			sut.Initialize(model);

			// Act
			sut.Learn(batch);

			// Assert
		}

		[TestMethod]
		public void SecondIterationOfLearnPerformsCorrectCalls()
		{
			// Arrange
			var model = CreateFullyConnectedNeuralNetworkModel();
			var batch = CreateBatches();

			for (var i = selectionSize; i < populationSize; i++)
			{
				mockModelExecuter.Setup(m => m.Execute(It.Is<FullyConnectedNeuralNetworkModel>(it => it.Equals(population[i])), batch[i].Inputs))
					.Returns(activationCountsPerLayer.Select(l => CreateAndInitializeDoubleArray(l, random.NextDouble)).ToArray());
			}

			sut.Initialize(model);
			sut.Learn(batch);

			// Act
			sut.Learn(batch);

			// Assert
		}

		private FullyConnectedNeuralNetworkModel CreateFullyConnectedNeuralNetworkModel()
		{


			return new FullyConnectedNeuralNetworkModel
			{
				ActivationCountsPerLayer = activationCountsPerLayer,
				ActivationFunction = activationFunction,
				BiasLayers = activationCountsPerLayer.Skip(1)
					.Select(l => CreateAndInitializeDoubleArray(l, random.NextDouble)).ToArray(),
				WeightLayers = activationCountsPerLayer.Skip(1)
					.Select((l, i) => CreateAndInitializeTwoDimentionalDoubleArray(activationCountsPerLayer[i - 1], l, random.NextDouble)).ToArray()
			};
		}

		private double[] CreateAndInitializeDoubleArray(int count, Func<double> elementInitializer)
		{
			var array = new double[count];

			return array.Select(d => random.NextDouble()).ToArray();
		}

		private double[,] CreateAndInitializeTwoDimentionalDoubleArray(int width, int height, Func<double> elementInitializer)
		{
			var twoDimensionalArray = new double[width, height];

			for (var x = 0; x < width; x++)
			{
				for (var y = 0; y < height; y++)
				{
					twoDimensionalArray[x, y] = elementInitializer();
				}
			}

			return twoDimensionalArray;
		}

		private InputOutputPairModel[] CreateBatches()
		{
			return new InputOutputPairModel[]
			{
				new InputOutputPairModel
				{
					Inputs = new[] { 0.56, 0.43, 0.99, 0.14 },
					Outputs = new[] { 1.0, 0.0, 0.0 }
				},
				new InputOutputPairModel
				{
					Inputs = new[] { 0.33, 0.78, 0.02, 0.99 },
					Outputs = new[] { 0.0, 1.0, 0.0 }
				}
			};
		}
	}
}
