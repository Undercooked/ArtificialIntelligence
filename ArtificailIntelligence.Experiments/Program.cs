using Ninject;

namespace ArtificailIntelligence.Experiments
{
	public class Program
	{
		static void Main(string[] args)
		{
			var kernel = new StandardKernel(new Module());
			var experiments = kernel.GetAll<IExperiment>();

			foreach(var experiment in experiments)
			{
				experiment.Run();
			}
		}
	}
}
