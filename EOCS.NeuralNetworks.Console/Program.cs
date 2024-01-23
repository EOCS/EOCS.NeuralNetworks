using EOCS.NeuralNetworks.Data;
using EOCS.NeuralNetworks.Data.Trainers;

namespace EOCS.NeuralNetworks.Console
{
    internal class Program
    {
        static void Main(string[] args)
        {
            var path = AppContext.BaseDirectory + "/dataset.csv";
            var dataset = DataSet.Load(path);

            var trainer = new GradientDescentANNTrainer();

            trainer.Train(dataset);

            var p = new DataToPredict()
            {
                Data = new Dictionary<string, double>
                {
                    {"first0", 1 },
                    {"X", 0.45 },
                    {"Y", -0.72 }
                }
            };

            var res = trainer.Predict(p);
        }
    }
}
