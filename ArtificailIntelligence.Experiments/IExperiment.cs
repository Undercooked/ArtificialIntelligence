namespace ArtificailIntelligence.Experiments
{
	internal interface IExperiment
	{
		string Title { get; }
		int Iterations { get; }

		void Initialize();
		void TrainModel();
		double GetModelScore();
	}
}
