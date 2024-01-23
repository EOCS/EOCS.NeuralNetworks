namespace EOCS.NeuralNetworks.Data.Interfaces
{
    public interface IANNTrainer
    {
        void Train(DataSet set);

        double Predict(DataToPredict record);
    }
}
