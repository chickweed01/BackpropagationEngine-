using BackpropagationEngine.Nodes;
using System.Collections;
using System.Collections.Generic;
using static BackpropagationEngine.Activation;

namespace BackpropagationEngine
{
    public static class BackpropagationEngine
    {
        public enum ActivationAlgorithm { Sigmoid, HyperTan, SoftMax };
        public static Activation activation = new Activation();
        public static ApplyActivationForScalar logSigActivation = activation.Sigmoid;
        public static ApplyActivationForVector softMaxActivation = activation.Softmax;

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

                foreach (Connector oConnector in hNode.OutboundConnectors)
                {
                    sum += outputGradients[indexOutput] + oConnector.Weight;
                    indexOutput++;
                }

                hiddenGradients[indexHidden] = deriviative * sum;
                indexHidden++;
            }

            return hiddenGradients;

        }

        public static void calculateGradientsForHiddenLayer<T>(ref IList<HiddenNode> hiddenLayer, IList<T> downstreamLayer,
                                                        ActivationAlgorithm activationAlgorithm)
        {
            double derivative = 0.0;
            double sum = 0.0;


            foreach (HiddenNode hNode in hiddenLayer)
            {
                sum = 0.0;

                if (activationAlgorithm == ActivationAlgorithm.HyperTan)
                    derivative = (1 - hNode.Val) * (1 + hNode.Val); // f' of tanh is (1-y)(1+y)

                for (int j = 0; j < downstreamLayer.Count; j++)
                {
                    if (typeof(T) == typeof(OutputNode))
                    {
                        sum += ((OutputNode)(object)downstreamLayer[j]).Gradient *
                                hNode.OutboundConnectors[j].Weight;
                    }
                    else if (typeof(T) == typeof(HiddenNode))
                    {
                        sum += ((HiddenNode)(object)downstreamLayer[j]).Gradient *
                                hNode.OutboundConnectors[j].Weight;
                    }
                }

                hNode.Gradient = derivative * sum;
            }
        }

        public static void calculateGradientsForOutputLayer(ref IList<OutputNode> outputLayer,
                                                        double[] targetValues,
                                                        ActivationAlgorithm activationAlgorithm)
        {
            int index = 0;

            foreach (OutputNode node in outputLayer)
            {
                if (activationAlgorithm == ActivationAlgorithm.Sigmoid ||
                    activationAlgorithm == ActivationAlgorithm.SoftMax)
                {
                    double derivative = node.Val * (1.0 - node.Val);
                    node.Gradient = derivative * (targetValues[index] - node.Val);

                    index++;
                }
            }
        }
        public static void calculateActivatedSumsForLayer<T>(ref IList<T> layer,
                                                    ActivationAlgorithm activationAlgorithm)
        {
            //OutputNode
            if (typeof(T) == typeof(OutputNode))
            {
                foreach (OutputNode oNode in (List<OutputNode>)layer)
                {
                    foreach (Connector connector in oNode.InboundConnectors)
                    {
                        oNode.Val += connector.Weight * connector.FromNode.Val;
                    }

                    if (activationAlgorithm == ActivationAlgorithm.Sigmoid)
                        oNode.Val = logSigActivation(oNode.Val);
                }

                if (activationAlgorithm == ActivationAlgorithm.SoftMax)
                {
                    int index = 0;
                    double[] temp = new double[layer.Count];
                    foreach (OutputNode oNode in (List<OutputNode>)layer)
                    {
                        temp[index] = oNode.Val;
                        index++;
                    }

                    index = 0;
                    temp = softMaxActivation(temp);
                    foreach (OutputNode oNode in (List<OutputNode>)layer)
                    {
                        oNode.Val = temp[index];
                        index++;
                    }
                }
            }
            else if (typeof(T) == typeof(HiddenNode))
            {
                //HiddenNode
                foreach (HiddenNode hNode in (List<HiddenNode>)layer)
                {
                    foreach (Connector connector in hNode.InboundConnectors)
                    {
                        hNode.Val += connector.Weight * connector.FromNode.Val;
                    }

                    if (activationAlgorithm == ActivationAlgorithm.Sigmoid)
                        hNode.Val = logSigActivation(hNode.Val);
                }

                if (activationAlgorithm == ActivationAlgorithm.SoftMax)
                {
                    int index = 0;
                    double[] temp = new double[layer.Count];
                    foreach (HiddenNode oNode in (List<HiddenNode>)layer)
                    {
                        temp[index] = oNode.Val;
                        index++;
                    }

                    index = 0;
                    temp = softMaxActivation(temp);
                    foreach (HiddenNode oNode in (List<HiddenNode>)layer)
                    {
                        oNode.Val = temp[index];
                        index++;
                    }
                }
            }
        }

        public static void adjustWeightsAtLayer(ref IList<HiddenNode> hiddenLayer, double learningRate, double momentum)
        {
            foreach (HiddenNode hNode in hiddenLayer)
            {
                foreach (Connector connector in hNode.OutboundConnectors)
                {
                    double delta = learningRate * connector.ToNode.Gradient * hNode.Val;
                    connector.Weight += (delta + momentum);
                }
            }
        }


        //public static void updateWeights(ref List<HiddenNode> hiddenNodes, double[] hiddenGradients, double[] outputGradients, double learningRate, double momentum)
        //{
        //    /* update the input-to-hidden weights and then the 
        //     * hidden-to-output weights */
        //    double delta, mFactor;
        //    int index = 0;
        //    int outputsIndex;

        //    foreach (HiddenNode hNode in hiddenNodes)
        //    {
        //        // 1) input-to-hidden weights
        //        foreach (Connector iConnector in hNode.InboundConnectors)
        //        {
        //            delta = learningRate * hiddenGradients[index] * iConnector.FromNode.Val;
        //            iConnector.Weight += delta;
        //            mFactor = momentum * iConnector.WeightDelta;
        //            iConnector.Weight += mFactor;
        //            iConnector.WeightDelta = delta;
        //        }

        //        // 2) hidden-to-output weights
        //        outputsIndex = 0;
        //        foreach (Connector iConnector in hNode.OutboundConnectors)
        //        {
        //            delta = learningRate * outputGradients[outputsIndex] * iConnector.ToNode.Val;
        //            iConnector.Weight += delta;
        //            mFactor = momentum * iConnector.WeightDelta;
        //            iConnector.Weight += mFactor;
        //            iConnector.WeightDelta = delta;
        //            outputsIndex++;
        //        }

        //        index++; //next hidden node
        //    }
        //}
    }
}
