using BackpropagationEngine.Nodes;
using System;
using System.Collections;
using System.Collections.Generic;

namespace BackpropagationEngine
{
    public static class Utility
    {
        public static class GenericNodeFactory
        {
            private static IList<Type> _registeredTypes = new List<Type>();

            static GenericNodeFactory()
            {
                _registeredTypes.Add(typeof(InputNode));
                _registeredTypes.Add(typeof(HiddenNode));
                _registeredTypes.Add(typeof(OutputNode));
                _registeredTypes.Add(typeof(BiasNode));
            }

            public static IList<T> CreateGenericNode<T>(int numberOfNodes)
            {
                var t = typeof(T);
                int index = _registeredTypes.IndexOf(t);
                var typeToCreate = _registeredTypes[index];
                IList<T> list = new List<T>();

                if (typeToCreate == typeof(InputNode))
                {
                    for (int i = 0; i < numberOfNodes; i++)
                    {
                        InputNode node = new InputNode();
                        list.Add((T)(object)node);
                    }
                }
                else if (typeToCreate == typeof(HiddenNode))
                {
                    for (int i = 0; i < numberOfNodes; i++)
                    {
                        HiddenNode node = new HiddenNode();
                        list.Add((T)(object)node);
                    }
                }
                else if (typeToCreate == typeof(OutputNode))
                {
                    for (int i = 0; i < numberOfNodes; i++)
                    {
                        OutputNode node = new OutputNode();
                        list.Add((T)(object)node);
                    }
                }
                else if (typeToCreate == typeof(BiasNode))
                {
                    for (int i = 0; i < numberOfNodes; i++)
                    {
                        BiasNode node = new BiasNode();
                        list.Add((T)(object)node);
                    }
                }

                return list;
            }
        }

        public static void createConnectors(int numberOfConnectors, out List<Connector> connectors)
        {

            connectors = new List<Connector>();

            for (int index = 1; index <= numberOfConnectors; index++)
            {
                connectors.Add(new Connector(index * .01));
            }
        }

        public static double[] computeOutputGradients(List<OutputNode> outputNodes, ArrayList targetValues)
        {
            /*
             * output gradients are calculated as follows:
             * oGrad = calculated output value(1-calculated output value) * (target output value - calculated output value)
             */

            int index = 0;
            var outputGradients = new double[outputNodes.Count];

            foreach (OutputNode oNode in outputNodes)
            {
                outputGradients[index] = oNode.Val * (1.0 - oNode.Val) * ((double)targetValues[index] - oNode.Val);
                index++;
            }

            return outputGradients;
        }

        public static double[] computeHiddenGradients(double[] outputGradients, IList<HiddenNode> hiddenNodes)
        {
            int indexHidden = 0;
            int indexOutput;
            double deriviative;
            double sum;
            var hiddenGradients = new double[hiddenNodes.Count];

            foreach (HiddenNode hNode in hiddenNodes)
            {
                indexOutput = 0;
                sum = 0.0;
                deriviative = (1.0 - hNode.Val) * (1.0 + hNode.Val);

                foreach (Connector oConnector in hNode.OutputConnectors)
                {
                    sum += outputGradients[indexOutput] + oConnector.Weight;
                    indexOutput++;
                }

                hiddenGradients[indexHidden] = deriviative * sum;
                indexHidden++;
            }

            return hiddenGradients;

        }

        public static void updateWeights(ref List<HiddenNode> hiddenNodes, double[] hiddenGradients, double[] outputGradients, double learningRate, double momentum)
        {
            /* update the input-to-hidden weights and then the 
             * hidden-to-output weights */
            double delta, mFactor;
            int index = 0;
            int outputsIndex;

            foreach (HiddenNode hNode in hiddenNodes)
            {
                // 1) input-to-hidden weights
                foreach (Connector iConnector in hNode.InputConnectors)
                {
                    delta = learningRate * hiddenGradients[index] * iConnector.FromNode.Val;
                    iConnector.Weight += delta;
                    mFactor = momentum * iConnector.WeightDelta;
                    iConnector.Weight += mFactor;
                    iConnector.WeightDelta = delta;
                }

                // 2) hidden-to-output weights
                outputsIndex = 0;
                foreach (Connector iConnector in hNode.OutputConnectors)
                {
                    delta = learningRate * outputGradients[outputsIndex] * iConnector.ToNode.Val;
                    iConnector.Weight += delta;
                    mFactor = momentum * iConnector.WeightDelta;
                    iConnector.Weight += mFactor;
                    iConnector.WeightDelta = delta;
                    outputsIndex++;
                }

                index++; //next hidden node
            }
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

        public static double[] Softmax(double[] oSums)
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
    }
}
