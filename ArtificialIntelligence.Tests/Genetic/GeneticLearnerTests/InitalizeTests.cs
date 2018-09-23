using System;
using System.Linq;
using System.Linq.Expressions;
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
	public class InitalizeTests
	{
		private const int populationSize = 10;
		private const int selectionSize = 4;
		private static readonly int[] activationCountsPerLayer = new[] { 4, 3 };
		private static readonly ActivationFunction activationFunction = ActivationFunction.Sigmoid;

		private Mock<IModelInitializer> mockModelInitializer;
		private Mock<IModelExecuter> mockModelExecuter;
		private Mock<IPopulationBreeder> mockPopulationBreeder;
		private ThreadSafeRandom random;
		private GeneticLearner sut;

		[TestInitialize]
		public void TestInitialize()
		{
			mockModelInitializer = new Mock<IModelInitializer>(MockBehavior.Strict);
			mockModelExecuter = new Mock<IModelExecuter>(MockBehavior.Strict);
			mockPopulationBreeder = new Mock<IPopulationBreeder>(MockBehavior.Strict);
			random = new ThreadSafeRandom();

			sut = new GeneticLearner(populationSize, selectionSize, mockModelInitializer.Object, mockModelExecuter.Object, mockPopulationBreeder.Object, random);
		}

		[TestMethod]
		public void InitializeCorrectlySetsTheModel()
		{
			// Arrange
			var model = CreateFullyConnectedNeuralNetworkModel();
			var activationCountsPerLayer = new[] { 4, 3 };
			Expression<Func<IModelInitializer, FullyConnectedNeuralNetworkModel>> createModelFunction = m => m.CreateModel(activationCountsPerLayer, model.ActivationFunction);

			mockModelInitializer.Setup(createModelFunction)
				.Returns(CreateFullyConnectedNeuralNetworkModel);

			// Act
			sut.Initialize(model);

			// Assert
			mockModelInitializer.Verify(createModelFunction, Times.Exactly(populationSize - 1));
			sut.Model.Should().BeSameAs(model);
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
	}
}
