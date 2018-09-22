using System;
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
		private Mock<IModelInitializer> mockModelInitializer;
		private Mock<IModelExecuter> mockModelExecuter;
		private Mock<IModelBreeder> mockModelBreeder;
		private ThreadSafeRandom random;
		private GeneticLearner sut;

		[TestInitialize]
		public void TestInitialize()
		{
			mockModelInitializer = new Mock<IModelInitializer>(MockBehavior.Strict);
			mockModelExecuter = new Mock<IModelExecuter>(MockBehavior.Strict);
			mockModelBreeder = new Mock<IModelBreeder>(MockBehavior.Strict);
			random = new ThreadSafeRandom();

			sut = new GeneticLearner(populationSize, selectionSize, mockModelInitializer.Object, mockModelExecuter.Object, mockModelBreeder.Object, random);
		}

		[TestMethod]
		public void InitializeCorrectlySetsTheModel()
		{
			// Arrange
			var model = CreateFullyConnectedNeuralNetworkModel();
			var activationCountsPerLayer = new[] { 4, 3 };
			Expression<Func<IModelInitializer, FullyConnectedNeuralNetworkModel>> createModelFunction = m => m.CreateModel(activationCountsPerLayer, model.ActivationFunction);

			mockModelInitializer.Setup(createModelFunction)
				.Returns(new FullyConnectedNeuralNetworkModel());

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
				ActivationCountsPerLayer = new[] { 4, 3 },
				ActivationFunction = ActivationFunction.Sigmoid,
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
				}
			};
		}
	}
}
