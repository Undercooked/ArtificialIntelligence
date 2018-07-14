using System;
using Ninject;
using NLog;

namespace ArtificailIntelligence.Experiments
{
	public class Program
	{
		private static readonly ILogger logger = LogManager.GetCurrentClassLogger();

		static void Main(string[] args)
		{
			Console.SetWindowSize(256, 64);
			Console.SetWindowPosition(Console.WindowWidth / 2 - 128, Console.WindowHeight / 2 - 32);

			var kernel = new StandardKernel(new Module());
			var experiments = kernel.GetAll<IExperiment>();

			foreach(var experiment in experiments)
			{
				experiment.Initialize();

				for (var i = 0; i < experiment.Iterations; i++)
				{
					var score = experiment.GetModelScore();

					logger.Info($"{experiment.Title} - iteration {i}: {score}");

					experiment.TrainModel();
				}

				var finalScore = experiment.GetModelScore();

				logger.Info($"{experiment.Title} - iteration {experiment.Iterations}: {finalScore}");
			}
		}
	}
}
