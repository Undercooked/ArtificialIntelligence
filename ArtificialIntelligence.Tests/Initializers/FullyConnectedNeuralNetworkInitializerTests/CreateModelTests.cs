using System.Linq;
using ArtificialIntelligence.Enums;
using ArtificialIntelligence.Initializers;
using ArtificialIntelligence.RandomNumberServices;
using FluentAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace ArtificialIntelligence.Tests.Initializers.FullyConnectedNeuralNetworkInitializerTests
{
	[TestClass]
	public class CreateModelTests
	{
		private ThreadSafeRandom random;
		private FullyConnectedNeuralNetworkInitializer sut;

		[TestInitialize]
		public void TestInitialize()
		{
			random = new ThreadSafeRandom();
			sut = new FullyConnectedNeuralNetworkInitializer(random);
		}

		[TestMethod]
		public void ModelParametersAreCorrectlyInitialized()
		{
			// Arrange
			var activationFunction = ActivationFunction.Sigmoid;
			var activationCountsPerLayer = new[] { 5, 4, 3, 2 };

			// Act
			var result = sut.CreateModel(activationCountsPerLayer, activationFunction);
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
