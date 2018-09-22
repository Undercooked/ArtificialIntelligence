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

namespace ArtificialIntelligence.Tests.Genetic.GeneticModelBreederTests
{
	[TestClass]
	public class BreedTests
	{
		private Mock<IModelInitializer> mockModelInitializer;
		private ThreadSafeRandom random;
		private GeneticModelBreeder sut;

		[TestInitialize]
		public void TestInitialize()
		{
			mockModelInitializer = new Mock<IModelInitializer>(MockBehavior.Strict);
			random = new ThreadSafeRandom();

			sut = new GeneticModelBreeder(mockModelInitializer.Object, random);
		}

		[TestMethod]
		public void BreedCallsTheCorrectDependencies()
		{
			// Arrange
			var activationCountsPerLayer = new[] { 5, 4, 3 };
			var activationFunction = ActivationFunction.Sigmoid;
			var mother = CreateFullyConnectedNeuralNetworkModel(activationCountsPerLayer, activationFunction);
			var father = CreateFullyConnectedNeuralNetworkModel(activationCountsPerLayer, activationFunction);

			mockModelInitializer.Setup(m => m.CreateModel(mother.ActivationCountsPerLayer, mother.ActivationFunction, It.IsAny<double[][]>(), It.IsAny<double[][,]>()))
				.Returns(new FullyConnectedNeuralNetworkModel());

			// Act
			var child = sut.Breed(mother, father);

			// Assert
			mockModelInitializer.Verify(m => m.CreateModel(mother.ActivationCountsPerLayer, mother.ActivationFunction, It.IsAny<double[][]>(), It.IsAny<double[][,]>()), Times.Once);
			child.Should().NotBeSameAs(mother);
			child.Should().NotBeSameAs(father);
		}

		[TestMethod]
		public void BreedProducesAChildModelThatHasTheCorrectPercentageOfChromosomesFromEachParentAndMutations()
		{
			// Arrange
			var activationCountsPerLayer = new[] { 5, 4, 3 };
			var activationFunction = ActivationFunction.Sigmoid;
			var sampleSize = 10000;
			var mothers = new FullyConnectedNeuralNetworkModel[sampleSize].Select(m => CreateFullyConnectedNeuralNetworkModel(activationCountsPerLayer, activationFunction)).ToArray();
			var fathers = new FullyConnectedNeuralNetworkModel[sampleSize].Select(m => CreateFullyConnectedNeuralNetworkModel(activationCountsPerLayer, activationFunction)).ToArray();
			var totalMotherChromosomeCount = 0;
			var totalFatherChromosomeCount = 0;
			var totalMutationChromosomeCount = 0;

			mockModelInitializer.Setup(m => m.CreateModel(It.IsAny<int[]>(), It.IsAny<ActivationFunction>(), It.IsAny<double[][]>(), It.IsAny<double[][,]>()))
				.Returns<int[], ActivationFunction, double[][], double[][,]>((arg1, arg2, arg3, arg4) => new FullyConnectedNeuralNetworkModel
				{
					ActivationCountsPerLayer = arg1,
					ActivationFunction = arg2,
					BiasLayers = arg3,
					WeightLayers = arg4
				});

			// Act
			for (var i = 0; i < sampleSize; i++)
			{
				var child = sut.Breed(mothers[i], fathers[i]);
				var childDna = FlattenWeightsAndBiases(child);
				var motherDna = FlattenWeightsAndBiases(mothers[i]);
				var fatherDna = FlattenWeightsAndBiases(fathers[i]);

				for (var j = 0; j < motherDna.Length; j++)
				{
					if (childDna[j] == motherDna[j])
					{
						totalMotherChromosomeCount++;
					}
					else if (childDna[j] == fatherDna[j])
					{
						totalFatherChromosomeCount++;
					}
					else
					{
						totalMutationChromosomeCount++;
					}
				}
			}
			var totalChromosomeCount = (double)totalMotherChromosomeCount + totalFatherChromosomeCount + totalMutationChromosomeCount;
			var motherChromosomePercentage = totalMotherChromosomeCount / totalChromosomeCount;
			var fatherChromosomePercentage = totalFatherChromosomeCount / totalChromosomeCount;
			var mutationChromosomePercentage = totalMutationChromosomeCount / totalChromosomeCount;

			// Assert
			mockModelInitializer.Verify(m => m.CreateModel(It.IsAny<int[]>(), It.IsAny<ActivationFunction>(), It.IsAny<double[][]>(), It.IsAny<double[][,]>()), Times.Exactly(sampleSize));
			motherChromosomePercentage.Should().BeInRange(0.4491, 0.5489); // should be around 0.499
			fatherChromosomePercentage.Should().BeInRange(0.4491, 0.5489); // should be around 0.499
			mutationChromosomePercentage.Should().BeInRange(0.0009, 0.0011); // should be around 0.001
		}

		private FullyConnectedNeuralNetworkModel CreateFullyConnectedNeuralNetworkModel(int[] activationCountsPerLayer, ActivationFunction activationFunction)
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

		private double[] FlattenWeightsAndBiases(FullyConnectedNeuralNetworkModel model)
		{
			var flatWeightsAndBiases = new List<double>();

			flatWeightsAndBiases.AddRange(model.WeightLayers.SelectMany(l => l.Cast<double>()));
			flatWeightsAndBiases.AddRange(model.BiasLayers.SelectMany(l => l));

			return flatWeightsAndBiases.ToArray();
		}
	}
}
