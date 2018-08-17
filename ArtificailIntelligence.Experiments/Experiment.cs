using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using ArtificialIntelligence;
using ArtificialIntelligence.Enums;
using ArtificialIntelligence.Models;

namespace ArtificailIntelligence.Experiments
{
	public class Experiment : IExperiment
	{
		private readonly int batchSize;
		private readonly ActivationFunction activationFunction;
		private readonly int[] activationCountsPerLayer;
		private readonly IModelExecuter executer;
		private readonly ILearner learner;
		private readonly IModelInitializer modelInitializer;
		private readonly Random random;
		private readonly IDataSource dataSource;

		private IEnumerable<InputOutputPairModel> trainingInputOutputPairs;
		private IEnumerable<InputOutputPairModel> testInputOutputPairs;

		public string Title { get; }
		public int Iterations { get; }

		public Experiment(
			string title,
			int iterations,
			int batchSize,
			ActivationFunction activationFunction,
			int[] activationCountsPerLayer,
			IModelExecuter executer,
			ILearner learner,
			IModelInitializer modelInitializer,
			IDataSource dataSource,
			Random random)
		{
			Title = title;
			Iterations = iterations;

			this.batchSize = batchSize;
			this.activationFunction = activationFunction;
			this.activationCountsPerLayer = activationCountsPerLayer;
			this.executer = executer;
			this.learner = learner;
			this.modelInitializer = modelInitializer;
			this.dataSource = dataSource;
			this.random = random;
		}

		public void Initialize()
		{
			var model = modelInitializer.CreateModel(activationCountsPerLayer, activationFunction, random);

			trainingInputOutputPairs = dataSource.GetData(DataPurpose.Training);
			testInputOutputPairs = dataSource.GetData(DataPurpose.Test);

			learner.Initialize(model);
		}

		public void TrainModel()
		{
			var miniBatches = CreateMiniBatches(trainingInputOutputPairs, batchSize);

			foreach (var batch in miniBatches)
			{
				learner.Learn(batch);
			}
		}

		public double GetModelScore()
		{
			var correct = 0;
			var incorrect = 0;

			Parallel.ForEach(testInputOutputPairs, pair =>
			{
				var outputs = executer.Execute(learner.Model, pair.Inputs).Last();
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
