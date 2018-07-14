﻿using System.Linq;
using System.Threading.Tasks;
using ArtificialIntelligence.ActivationFunctions;
using ArtificialIntelligence.Enums;
using ArtificialIntelligence.Models;
using Ninject;

namespace ArtificialIntelligence.Learners
{
	public class BackPropagationLearner : ILearner
	{
		private readonly IExecuter executer;
		private readonly IActivationFunction sigmoidActivationFunction;

		private double[][,] layerWeightDeltaTotals;
		private double[][] layerBiasDeltaTotals;

		public BackPropagationLearner(IExecuter executer, [Named(nameof(SigmoidActivationFunction))] IActivationFunction sigmoidActivationFunction)
		{
			this.executer = executer;
			this.sigmoidActivationFunction = sigmoidActivationFunction;
		}

		public FullyConnectedNeuralNetworkModel Learn(FullyConnectedNeuralNetworkModel model, InputOutputPairModel[] batch)
		{
			layerWeightDeltaTotals = new double[model.WeightLayers.Length][,];
			layerBiasDeltaTotals = new double[model.BiasLayers.Length][];

			Parallel.ForEach(batch, inputOutputPair =>
			{
				var allActivations = executer.Execute(model, inputOutputPair.Inputs);
				var deltaOutputActivations = allActivations.Last().Select((output, index) => DerivativeCost(output, inputOutputPair.Outputs[index])).ToArray();
				var inputActivationLayers = allActivations.Take(allActivations.Length - 1).ToArray();

				PropagateBackwards(inputActivationLayers, model, deltaOutputActivations);
			});

			return ApplyLayerDeltas(model, batch);
		}

		private double DerivativeCost(double activation, double desiredValue)
		{
			return 2 * (activation - desiredValue);
		}

		private void PropagateBackwards(double[][] inputActivationLayers, FullyConnectedNeuralNetworkModel model, double[] deltaOutputActivations)
		{
			for (var layerIndex = inputActivationLayers.Length - 1; layerIndex >= 0; layerIndex--)
			{
				deltaOutputActivations = PropagateLayerBackwards(layerIndex, inputActivationLayers[layerIndex], model, deltaOutputActivations);
			}
		}

		private double[] PropagateLayerBackwards(int layerIndex, double[] inputActivations, FullyConnectedNeuralNetworkModel model, double[] deltaOutputActivations)
		{
			var outputActivationCount = deltaOutputActivations.Length;
			var sigmoidDerivatives = new double[outputActivationCount];

			for (var j = 0; j < outputActivationCount; j++)
			{
				sigmoidDerivatives[j] = Activation(inputActivations, model.WeightLayers[layerIndex], model.BiasLayers[layerIndex][j], model.ActivationFunction, j);
			}

			DeltaWeights(layerIndex, inputActivations, sigmoidDerivatives, deltaOutputActivations);
			DeltaBiases(layerIndex, sigmoidDerivatives, deltaOutputActivations);

			return DeltaActivations(model.WeightLayers[layerIndex], sigmoidDerivatives, deltaOutputActivations);
		}

		private FullyConnectedNeuralNetworkModel ApplyLayerDeltas(FullyConnectedNeuralNetworkModel model, InputOutputPairModel[] batch)
		{
			var newModel = new FullyConnectedNeuralNetworkModel
			{
				ActivationFunction = model.ActivationFunction,
				BiasLayers = new double[model.BiasLayers.Length][],
				WeightLayers = new double[model.WeightLayers.Length][,]
			};

			for (var layerIndex = 0; layerIndex < model.WeightLayers.Length; layerIndex++)
			{
				var inputActivationCount = model.WeightLayers[layerIndex].GetLength(0);
				var outputActivationCount = model.WeightLayers[layerIndex].GetLength(1);

				newModel.WeightLayers[layerIndex] = new double[inputActivationCount, outputActivationCount];
				newModel.BiasLayers[layerIndex] = new double[outputActivationCount];

				for (var o = 0; o < outputActivationCount; o++)
				{
					for (var i = 0; i < inputActivationCount; i++)
					{
						var weightLayerDeltaAverage = layerWeightDeltaTotals[layerIndex][i, o] / batch.Length;
						newModel.WeightLayers[layerIndex][i, o] = model.WeightLayers[layerIndex][i, o] - weightLayerDeltaAverage;
					}

					var biasLayerDeltaAverage = layerBiasDeltaTotals[layerIndex][o] / batch.Length;
					newModel.BiasLayers[layerIndex][o] = model.BiasLayers[layerIndex][o] - biasLayerDeltaAverage;
				}
			}

			return newModel;
		}

		private double Activation(double[] activations, double[,] weights, double bias, ActivationFunction activationFunction, int outputIndex)
		{
			var zj = 0D;

			for (var inputIndex = 0; inputIndex < activations.Length; inputIndex++)
			{
				zj += activations[inputIndex] * weights[inputIndex, outputIndex];
			}

			zj += bias;

			switch (activationFunction)
			{
				case ActivationFunction.Sigmoid:
					return sigmoidActivationFunction.CalculateDerivative(zj);
				default:
					return zj;
			}
		}

		private void DeltaWeights(int layerIndex, double[] inputActivations, double[] sigmoidDerivatives, double[] deltaOutputActivations)
		{
			var inputActivationCount = inputActivations.Length;
			var outputActivationCount = deltaOutputActivations.Length;

			for (var inputIndex = 0; inputIndex < inputActivationCount; inputIndex++)
			{
				for (var outputIndex = 0; outputIndex < outputActivationCount; outputIndex++)
				{
					var weightDelta = inputActivations[inputIndex] * sigmoidDerivatives[outputIndex] * deltaOutputActivations[outputIndex];

					if (layerWeightDeltaTotals[layerIndex] == null)
					{
						layerWeightDeltaTotals[layerIndex] = new double[inputActivationCount, outputActivationCount];
					}

					layerWeightDeltaTotals[layerIndex][inputIndex, outputIndex] += weightDelta;
				}
			}
		}

		private void DeltaBiases(int layerIndex, double[] sigmoidDerivatives, double[] deltaOutputActivations)
		{
			var biasCount = deltaOutputActivations.Length;

			for (var j = 0; j < biasCount; j++)
			{
				var biasDelta = sigmoidDerivatives[j] * deltaOutputActivations[j];

				if (layerBiasDeltaTotals[layerIndex] == null)
				{
					layerBiasDeltaTotals[layerIndex] = new double[biasCount];
				}

				layerBiasDeltaTotals[layerIndex][j] += biasDelta;
			}
		}

		private double[] DeltaActivations(double[,] weights, double[] sigmoidDerivatives, double[] deltaOutputActivations)
		{
			var inputActivationCount = weights.GetLength(0);
			var outputActivationCount = weights.GetLength(1);
			var deltaActivations = new double[inputActivationCount];

			for (var inputIndex = 0; inputIndex < inputActivationCount; inputIndex++)
			{
				for (var outputIndex = 0; outputIndex < outputActivationCount; outputIndex++)
				{
					deltaActivations[inputIndex] += weights[inputIndex, outputIndex] * sigmoidDerivatives[outputIndex] * deltaOutputActivations[outputIndex];
				}
			}

			return deltaActivations;
		}
	}
}
