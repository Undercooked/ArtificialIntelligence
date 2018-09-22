using System;
using System.Collections.Generic;
using System.Linq;
using ArtificialIntelligence.Enums;
using ArtificialIntelligence.Genetic;
using ArtificialIntelligence.Models;
using ArtificialIntelligence.RandomNumberServices;
using FluentAssertions;
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

		private ThreadSafeRandom random;
		private Mock<IModelInitializer> mockModelInitializer;
		private Mock<IModelExecuter> mockModelExecuter;
		private Mock<IModelBreeder> mockModelBreeder;
		private List<FullyConnectedNeuralNetworkModel> population;
		private GeneticLearner sut;

		[TestInitialize]
		public void TestInitialize()
		{
			random = new ThreadSafeRandom();
			mockModelInitializer = new Mock<IModelInitializer>(MockBehavior.Strict);
			mockModelExecuter = new Mock<IModelExecuter>(MockBehavior.Strict);
			mockModelBreeder = new Mock<IModelBreeder>(MockBehavior.Strict);
			population = new List<FullyConnectedNeuralNetworkModel>();

			sut = new GeneticLearner(populationSize, selectionSize, mockModelInitializer.Object, mockModelExecuter.Object, mockModelBreeder.Object, random);

			mockModelInitializer.Setup(m => m.CreateModel(It.Is<int[]>(it => it != null && activationCountsPerLayer.SequenceEqual(it)), activationFunction))
				.Returns(() =>
				{
					var model = CreateFullyConnectedNeuralNetworkModel();
					population.Add(model);

					return model;
				});
		}

		[TestMethod]
		public void LearnPerformsTheCorrectCalls()
		{
			// Arrange
			var model = CreateFullyConnectedNeuralNetworkModel();
			var batch = CreateBatch();

			mockModelExecuter.Setup(m => m.Execute(It.IsAny<FullyConnectedNeuralNetworkModel>(), It.IsAny<double[]>()))
				.Returns(activationCountsPerLayer.Select(l => CreateAndInitializeDoubleArray(l, () => 0.5)).ToArray());
			// Setup 1 member of the population to return the desired output for the batches
			mockModelExecuter.Setup(m => m.Execute(It.Is<FullyConnectedNeuralNetworkModel>(it => it.Equals(population[1])), batch[0].Inputs))
				.Returns(new[] { batch[0].Outputs });
			mockModelExecuter.Setup(m => m.Execute(It.Is<FullyConnectedNeuralNetworkModel>(it => it.Equals(population[1])), batch[1].Inputs))
				.Returns(new[] { batch[1].Outputs });

			mockModelBreeder.Setup(m => m.Breed(It.Is<FullyConnectedNeuralNetworkModel>(it => it.Equals(population[1])), It.Is<FullyConnectedNeuralNetworkModel>(it => !it.Equals(population[1]) && population.Contains(it))))
				.Returns(CreateFullyConnectedNeuralNetworkModel);

			sut.Initialize(model);
			population.Add(model);

			// Act
			sut.Learn(batch);

			// Assert
			mockModelExecuter.Verify(m => m.Execute(It.IsAny<FullyConnectedNeuralNetworkModel>(), It.IsAny<double[]>()), Times.Exactly(6));
			mockModelExecuter.Verify(m => m.Execute(It.Is<FullyConnectedNeuralNetworkModel>(it => it.Equals(population[1])), batch[0].Inputs), Times.Once);
			mockModelExecuter.Verify(m => m.Execute(It.Is<FullyConnectedNeuralNetworkModel>(it => it.Equals(population[1])), batch[1].Inputs), Times.Once);
			mockModelBreeder.Verify(m => m.Breed(It.Is<FullyConnectedNeuralNetworkModel>(it => it.Equals(population[1])), It.Is<FullyConnectedNeuralNetworkModel>(it => !it.Equals(population[1]) && population.Contains(it))), Times.Once);
			sut.Model.Should().BeSameAs(population[1]);
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
					.Select((l, i) => CreateAndInitializeTwoDimentionalDoubleArray(activationCountsPerLayer[i], l, random.NextDouble)).ToArray()
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

		private InputOutputPairModel[] CreateBatch()
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
