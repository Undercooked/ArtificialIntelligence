using ArtificialIntelligence.Learners;
using ArtificialIntelligence.Models;
using FluentAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Moq;

namespace ArtificialIntelligence.Tests.Learners.BackPropagationLearnerTests
{
	[TestClass]
	public class InitializeTests
	{
		private Mock<IExecuter> mockExecuter;
		private Mock<IActivationFunction> mockSigmoidActivationFunction;
		private BackPropagationLearner sut;

		[TestInitialize]
		public void TestInitialize()
		{
			sut = new BackPropagationLearner(mockExecuter.Object, mockSigmoidActivationFunction.Object);
		}

		[TestMethod]
		public void InitializeSetsTheModelCorrectly()
		{
			// Arrange
			var model = new FullyConnectedNeuralNetworkModel();

			// Act
			sut.Initialize(model);

			// Assert
			sut.Model.Should().BeSameAs(model);
		}
	}
}
