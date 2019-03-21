using System;

namespace BackpropagationEngine
{
    public class Activation
    {
        // delegate declaration
        public delegate double[] ApplyActivationForVector(double[] valueVector);
        public delegate double ApplyActivationForScalar(double value);

        // Define methods that have the same signature as the delegates

        public double Sigmoid(double x)
        {
            if (x < -10.0) return 0.0f;
            else if (x > 10.0) return 1.0f;
            return (1.0 / (1.0 + Math.Exp(-x)));
        }

        public double[] Softmax(double[] oSums)
        {
            double max = oSums[0];
            for (int i = 0; i < oSums.Length; ++i)
                if (oSums[i] > max)
                    max = oSums[i];

            double scale = 0.0;
            for (int i = 0; i < oSums.Length; ++i)
                scale += Math.Exp(oSums[i] - max);

            double[] result = new double[oSums.Length];
            for (int i = 0; i < oSums.Length; ++i)
                result[i] = Math.Exp(oSums[i] - max) / scale;

            return result; // xi sum to 1.0. 
        }

        public static double HyperTan(double v)
        {
            if (v < -20.0)
                return -1.0;
            else if (v > 20.0)
                return 1.0;
            else
                return Math.Tanh(v);
        }
    }
}
