using ArtificialIntelligence.Learners;
using ArtificialIntelligence.Models;
using FluentAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace ArtificialIntelligence.Tests.Learners.BackPropagationLearnerTests
{
	[TestClass]
	public class InitializeTests
	{
		private BackPropagationLearner sut;

		[TestInitialize]
		public void TestInitialize()
		{
			sut = new BackPropagationLearner(null, null);
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
