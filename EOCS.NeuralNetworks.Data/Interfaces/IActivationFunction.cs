namespace EOCS.NeuralNetworks.Data.Interfaces
{
    public interface IActivationFunction
    {
        double Evaluate(double input);

        double EvaluateDerivative(double input);
    }
}
