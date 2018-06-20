using ArtificialIntelligence.Enums;
using ArtificialIntelligence.Executers;
using ArtificialIntelligence.Models;
using FluentAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Moq;

namespace ArtificialIntelligence.Tests.Executers.FullyConnectedNeuralNetworkExecuterTests
{
	[TestClass]
	public class ExecuteTests
	{
		private Mock<IActivationFunction> mockSigmoidActivationFunction;
		private FullyConnectedNeuralNetworkExecuter sut;

		[TestInitialize]
		public void TestInitialize()
		{
			mockSigmoidActivationFunction = new Mock<IActivationFunction>(MockBehavior.Strict);

			sut = new FullyConnectedNeuralNetworkExecuter(mockSigmoidActivationFunction.Object);
		}

		[TestMethod]
		public void CorrectResultsAreReturnedWhenActivationFunctionIsNone()
		{
			// Arrange
			var network = new FullyConnectedNeuralNetworkModel
			{
				BiasLayers = new[] { new[] { -0.84167747099030643, 0.66957177392652811, 0.78872177740033789 } },
				WeightLayers = new[]
				{
					new[,]
					{
						{ -0.554521067791861, 0.370098950979346, -0.79384722830441179 },
						{ 0.60650056395982421, -0.95214650032676129, 0.67987006608390721 },
						{ -0.35679294977187781, -0.81505192248851621, 0.14074044914019312 },
						{ 0.60933753829884219, 0.11109885252597684, -0.59281794335358673 }
					}
				},
				ActivationFunction = ActivationFunction.None
			};
			var activations = new[] { 0.34, 0.56, 0.78, 0.90 };

			// Act
			var results = sut.Execute(network, activations);

			// Assert
			results.Should().HaveCount(2);
			results[0].Should().BeEquivalentTo(activations);
			results[1].Should().BeEquivalentTo(new[] { -0.42046903457514428, -0.27354815519114406, 0.47578235809494851 });
		}

		[TestMethod]
		public void CorrectResultsAreReturnedWhenActivationFunctionIsUsed()
		{
			// Arrange
			var network = CreateFullyConnectedNeuralNetworkModel(ActivationFunction.Sigmoid);
			var activations = new[] { 0.34, 0.56, 0.78, 0.90 };

			mockSigmoidActivationFunction.Setup(m => m.Calculate(It.IsAny<double>()))
				.Returns<double>(arg1 => 2 * arg1);

			// Act
			var results = sut.Execute(network, activations);

			// Assert
			mockSigmoidActivationFunction.VerifyAll();
			results.Should().HaveCount(2);
			results[0].Should().BeEquivalentTo(activations);
			results[1].Should().BeEquivalentTo(new[] { -0.84093806915028857, -0.54709631038228812, 0.951564716189897 });
		}

		private FullyConnectedNeuralNetworkModel CreateFullyConnectedNeuralNetworkModel(ActivationFunction activationFunction)
		{
			return new FullyConnectedNeuralNetworkModel
			{
				BiasLayers = new[]
				{
					new[] { -0.84167747099030643, 0.66957177392652811, 0.78872177740033789 }
				},
				WeightLayers = new[]
				{
					new[,]
					{
						{ -0.554521067791861, 0.370098950979346, -0.79384722830441179 },
						{ 0.60650056395982421, -0.95214650032676129, 0.67987006608390721 },
						{ -0.35679294977187781, -0.81505192248851621, 0.14074044914019312 },
						{ 0.60933753829884219, 0.11109885252597684, -0.59281794335358673 }
					}
				},
				ActivationFunction = activationFunction
			};
		}
	}
}
