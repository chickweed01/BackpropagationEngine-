namespace BackpropagationEngine.Activation
{
    public interface IActivationFormulaDelegate
    {
        double applyActivation(double v);
        double[] applyActivation(double[] v);
    }
}
