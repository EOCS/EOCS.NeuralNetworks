using System.Data;

namespace EOCS.NeuralNetworks.Data
{
    public class DataSet
    {
        public List<string> Features {  get; set; }

        public List<DataRecord> Records { get; set; }

        public static DataSet Load(string path)
        {
            var contents = File.ReadAllLines(path);

            var features = new List<string>() { "first0" };
            features.AddRange(contents.First().Split(';').SkipLast(1));

            var nbFeatures = features.Count - 1;

            var records = new List<DataRecord>();
            foreach (var content in contents.Skip(1))
            {
                var d = content.Split(';');
                var data = new Dictionary<string, double>
                {
                    { "first0", 1 }
                };

                for (var i = 1; i <= nbFeatures; i++)
                {
                    var feature = features[i];
                    data.Add(feature, Convert.ToDouble(d[i-1]));
                }

                var point = new DataRecord()
                {
                    Data = data,
                    Target = Convert.ToDouble(d[nbFeatures])
                };
                records.Add(point);
            }

            return new DataSet() { Features = features, Records = records }; ;
        }
    }

    public class DataRecord()
    {
        public Dictionary<string, double> Data { get; set; }

        public double Target { get; set; }
    }

    public class DataToPredict()
    {
        public Dictionary<string, double> Data { get; set; }
    }
}
