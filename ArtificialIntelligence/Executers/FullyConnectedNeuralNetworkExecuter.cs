using ArtificialIntelligence.ActivationFunctions;
using ArtificialIntelligence.Enums;
using ArtificialIntelligence.Models;
using Ninject;

namespace ArtificialIntelligence.Executers
{
	public class FullyConnectedNeuralNetworkExecuter : IModelExecuter
	{
		private readonly IActivationFunction sigmoidActivationFunction;

		public FullyConnectedNeuralNetworkExecuter([Named(nameof(SigmoidActivationFunction))] IActivationFunction sigmoidActivationFunction)
		{
			this.sigmoidActivationFunction = sigmoidActivationFunction;
		}

		public double[][] Execute(FullyConnectedNeuralNetworkModel model, double[] activations)
		{
			var layerCount = model.WeightLayers.Length;
			var allActivations = new double[layerCount + 1][];
			allActivations[0] = activations;

			for (var layerIndex = 0; layerIndex < layerCount; layerIndex++)
			{
				allActivations[layerIndex + 1] = ExecuteLayer(model.BiasLayers[layerIndex], model.WeightLayers[layerIndex], allActivations[layerIndex], model.ActivationFunction);
			}

			return allActivations;
		}

		private double[] ExecuteLayer(double[] biases, double[,] weights, double[] activations, ActivationFunction activationFunction)
		{
			var outputs = new double[biases.Length];

			for (var outputNeuronIndex = 0; outputNeuronIndex < biases.Length; outputNeuronIndex++)
			{
				for (var inputNeuronIndex = 0; inputNeuronIndex < activations.Length; inputNeuronIndex++)
				{
					var weight = weights[inputNeuronIndex, outputNeuronIndex];
					outputs[outputNeuronIndex] += activations[inputNeuronIndex] * weight;
				}

				outputs[outputNeuronIndex] += biases[outputNeuronIndex];
				outputs[outputNeuronIndex] = ApplyActivationFunction(outputs[outputNeuronIndex], activationFunction);
			}

			return outputs;
		}

		private double ApplyActivationFunction(double input, ActivationFunction activationFunction)
		{
			switch (activationFunction)
			{
				case ActivationFunction.Sigmoid:
					return sigmoidActivationFunction.Calculate(input);
				default:
					return input;
			}
		}
	}
}
