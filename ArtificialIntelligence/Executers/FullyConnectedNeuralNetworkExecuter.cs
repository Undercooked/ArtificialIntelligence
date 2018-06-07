using ArtificialIntelligence.Enums;
using ArtificialIntelligence.Models;

namespace ArtificialIntelligence.Executers
{
	public class FullyConnectedNeuralNetworkExecuter : IExecuter
	{
		private readonly IActivationFunction sigmoidActivationFunction;

		public FullyConnectedNeuralNetworkExecuter(IActivationFunction sigmoidActivationFunction)
		{
			this.sigmoidActivationFunction = sigmoidActivationFunction;
		}

		public float[][] Execute(FullyConnectedNeuralNetworkModel model, float[] activations)
		{
			var layerCount = model.WeightLayers.Length;
			var allActivations = new float[layerCount + 1][];
			allActivations[0] = activations;

			for (var layerIndex = 0; layerIndex < layerCount; layerIndex++)
			{
				allActivations[layerIndex + 1] = ExecuteLayer(model.BiasLayers[layerIndex], model.WeightLayers[layerIndex], allActivations[layerIndex], model.ActivationFunction);
			}

			return allActivations;
		}

		private float[] ExecuteLayer(float[] biases, float[,] weights, float[] activations, ActivationFunction activationFunction)
		{
			var outputs = new float[biases.Length];

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

		private float ApplyActivationFunction(float input, ActivationFunction activationFunction)
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
