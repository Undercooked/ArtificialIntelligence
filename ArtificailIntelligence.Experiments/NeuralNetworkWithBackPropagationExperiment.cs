using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using ArtificialIntelligence;
using ArtificialIntelligence.ActivationFunctions;
using ArtificialIntelligence.Enums;
using ArtificialIntelligence.Executers;
using ArtificialIntelligence.Initializers;
using ArtificialIntelligence.Learners;
using ArtificialIntelligence.Models;

namespace ArtificailIntelligence.Experiments
{
	public class NeuralNetworkWithBackPropagationExperiment : IExperiment
	{
		private const int iterations = 10;
		private const int batchSize = 100;
		private const ActivationFunction activationFunction = ActivationFunction.Sigmoid;
		private static readonly int[] activationCountsPerLayer = new[] { 784, 200, 10 };

		private FullyConnectedNeuralNetworkExecuter executer;
		private BackPropagationLearner learner;
		private IModelInitializer modelInitializer;
		private Random random;
		private MnistDataSource mnistDataSource;
		private FullyConnectedNeuralNetworkModel model;
		private IEnumerable<InputOutputPairModel> trainingInputOutputPairs;
		private IEnumerable<InputOutputPairModel> testInputOutputPairs;

		public string Title => nameof(NeuralNetworkWithBackPropagationExperiment);
		public int Iterations => iterations;

		public NeuralNetworkWithBackPropagationExperiment()
		{
			var sigmoidActivationFunction = new SigmoidActivationFunction();

			executer = new FullyConnectedNeuralNetworkExecuter(sigmoidActivationFunction);
			learner = new BackPropagationLearner(executer, sigmoidActivationFunction);
			modelInitializer = new FullyConnectedNeuralNetworkInitializer();
			random = new Random();
			mnistDataSource = new MnistDataSource();
		}

		public void Initialize()
		{
			model = modelInitializer.CreateModel(activationFunction, activationCountsPerLayer, random);
			trainingInputOutputPairs = mnistDataSource.GetData(DataPurpose.Training);
			testInputOutputPairs = mnistDataSource.GetData(DataPurpose.Test);
		}

		public void TrainModel()
		{
			var miniBatches = CreateMiniBatches(trainingInputOutputPairs, batchSize);

			foreach (var batch in miniBatches)
			{
				model = learner.Learn(model, batch);
			}
		}

		public double GetModelScore()
		{
			var correct = 0;
			var incorrect = 0;

			Parallel.ForEach(testInputOutputPairs, pair =>
			{
				var outputs = executer.Execute(model, pair.Inputs).Last();
				var outputLabel = Array.IndexOf(outputs, outputs.Max());
				var expectedLabel = Array.IndexOf(pair.Outputs, pair.Outputs.Max());

				if (outputLabel == expectedLabel)
				{
					Interlocked.Increment(ref correct);
				}
				else
				{
					Interlocked.Increment(ref incorrect);
				}
			});

			return (double)correct / (correct + incorrect);
		}

		private IEnumerable<InputOutputPairModel[]> CreateMiniBatches(IEnumerable<InputOutputPairModel> orderedInputOutputPairs, int batchSize)
		{
			var shuffled = orderedInputOutputPairs.OrderBy(ioPair => random.Next()).ToArray();

			for (var i = 0; i < shuffled.Length; i += batchSize)
			{
				yield return shuffled.Skip(i).Take(batchSize).ToArray();
			}
		}
	}
}
