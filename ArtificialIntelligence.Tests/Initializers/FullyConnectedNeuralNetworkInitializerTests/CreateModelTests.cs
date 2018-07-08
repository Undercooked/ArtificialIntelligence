using System;
using System.Linq;
using ArtificialIntelligence.Enums;
using ArtificialIntelligence.Initializers;
using FluentAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace ArtificialIntelligence.Tests.Initializers.FullyConnectedNeuralNetworkInitializerTests
{
	[TestClass]
	public class CreateModelTests
	{
		private FullyConnectedNeuralNetworkInitializer sut;

		[TestInitialize]
		public void TestInitialize()
		{
			sut = new FullyConnectedNeuralNetworkInitializer();
		}

		[TestMethod]
		public void ModelParametersAreCorrectlyInitialized()
		{
			// Arrange
			var activationFunction = ActivationFunction.Sigmoid;
			var activationCountsPerLayer = new[] { 5, 4, 3, 2 };
			var random = new Random();

			// Act
			var result = sut.CreateModel(activationFunction, activationCountsPerLayer, random);
			var biases = result.BiasLayers.SelectMany(l => l);
			var weights = result.WeightLayers.SelectMany(l => l.Cast<double>());

			// Assert
			result.ActivationFunction.Should().Be(activationFunction);
			biases.Should().Contain(b => b != 0);
			biases.Should().OnlyContain(b => b >= -1 && b < 1);
			weights.Should().Contain(w => w != 0);
			weights.Should().OnlyContain(w => w >= -1 && w < 1);
		}
	}
}
