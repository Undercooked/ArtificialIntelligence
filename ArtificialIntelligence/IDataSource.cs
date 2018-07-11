using System.Collections.Generic;
using ArtificialIntelligence.Enums;
using ArtificialIntelligence.Models;

namespace ArtificialIntelligence
{
	public interface IDataSource
	{
		IEnumerable<InputOutputPairModel> GetData(DataPurpose dataSourceType);
	}
}
