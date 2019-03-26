using BackpropagationEngine.Activation;
using NN.Utility;
using NN.Utility.Nodes;
using System;
using System.Collections;
using System.Collections.Generic;

namespace BackpropagationEngine
{
    public class BackpropagationEngine
    {
        public enum ActivationAlgorithm { Sigmoid, HyperTan, SoftMax };
        protected IActivationFormulaDelegate activationFormulaDelegate;

        //private static Activation activation = new Activation();
        //private Activation.ApplyActivationDelegate logSigActivation = activation.Sigmoid;
        //private Activation.ApplyActivationDelegate hyperTanActivation = activation.HyperTan;
        //private Activation.ApplyActivationForVector softMaxActivation = activation.Softmax;

        public void doBackProp(double learningRate, double momentum, double[] targetValues, 
                                      ref IList<InputNode> inputLayer, ref IList<HiddenNode> hiddenLayer, ref IList<OutputNode> outputLayer)
        {
            try
            {
                // Compute and display outputs. 
                calculateActivatedSumsForLayer(ref hiddenLayer, ActivationAlgorithm.HyperTan);
                calculateActivatedSumsForLayer(ref outputLayer, ActivationAlgorithm.SoftMax);

                //compute gradients of output nodes
                calculateGradientsForOutputLayer(ref outputLayer, targetValues, ActivationAlgorithm.Sigmoid);

                //compute gradients of hidden nodes
                calculateGradientsForHiddenLayer(ref hiddenLayer, outputLayer, ActivationAlgorithm.Sigmoid);

                //update all weights and bias values
                adjustWeightsAtLayer(ref hiddenLayer, learningRate, momentum);  //hidden-to-output weights
                adjustWeightsAtLayer(ref inputLayer, learningRate, momentum);  //input-to-hidden weights
            }
            catch(Exception ex)
            {
                throw ex;
            }
        }       

        private void calculateGradientsForHiddenLayer<T>(ref IList<HiddenNode> hiddenLayer, IList<T> downstreamLayer, 
                                                        ActivationAlgorithm activationAlgorithm)
        {
            double derivative = 0.0;
            double sum = 0.0;


            foreach (HiddenNode hNode in hiddenLayer)
            {
                sum = 0.0;

                if (activationAlgorithm == ActivationAlgorithm.HyperTan)
                    derivative = (1 - hNode.Val) * (1 + hNode.Val); // f' of tanh is (1-y)(1+y)
                else if (activationAlgorithm == ActivationAlgorithm.Sigmoid)
                    derivative = hNode.Val * (1 - hNode.Val); // f' of sigmoid is y(1-y)
                else if (activationAlgorithm == ActivationAlgorithm.SoftMax)
                    derivative = hNode.Val * (1 - hNode.Val); //f' of softmax is y(1-y)

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

        private void calculateGradientsForOutputLayer(ref IList<OutputNode> outputLayer, double[] targetValues, 
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

        private void calculateActivatedSumsForLayer<T>(ref IList<T> layer, ActivationAlgorithm activationAlgorithm)
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
                    {
                        var sigmoid = new Sigmoid();
                        activationFormulaDelegate = sigmoid;
                        oNode.Val = activationFormulaDelegate.applyActivation(oNode.Val);
                    }
                    else if (activationAlgorithm == ActivationAlgorithm.HyperTan)
                    {
                        var hTan = new HyperTan();
                        activationFormulaDelegate = hTan;
                        oNode.Val = activationFormulaDelegate.applyActivation(oNode.Val);
                    }
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
                    var softMax = new SoftMax();
                    activationFormulaDelegate = softMax;
                    temp = activationFormulaDelegate.applyActivation(temp);
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
                    {
                        var sigmoid = new Sigmoid();
                        activationFormulaDelegate = sigmoid;
                        hNode.Val = activationFormulaDelegate.applyActivation(hNode.Val);
                    }
                    else if (activationAlgorithm == ActivationAlgorithm.HyperTan)
                    {
                        var hTan = new HyperTan();
                        activationFormulaDelegate = hTan;
                        hNode.Val = activationFormulaDelegate.applyActivation(hNode.Val);
                    }
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
                    var softMax = new SoftMax();
                    activationFormulaDelegate = softMax;
                    temp = activationFormulaDelegate.applyActivation(temp);
                    foreach (HiddenNode oNode in (List<HiddenNode>)layer)
                    {
                        oNode.Val = temp[index];
                        index++;
                    }
                }
            }
        }

        //always adjusts the downstream weights in the node's outbound connectors
        private void adjustWeightsAtLayer<T>(ref IList<T> layer, double learningRate, double momentum)
        {
            //OutputNode
            if (typeof(T) == typeof(InputNode))
            {
                foreach (InputNode iNode in (List<InputNode>)layer)
                {
                    foreach (Connector connector in iNode.OutboundConnectors)
                    {
                        double delta = learningRate * connector.ToNode.Gradient * iNode.Val;
                        connector.Weight += (delta + (momentum * connector.WeightDelta));
                        connector.WeightDelta = delta;
                    }
                }
            }
            else if (typeof(T) == typeof(HiddenNode))
            {
                //HiddenNode
                foreach (HiddenNode hNode in (List<HiddenNode>)layer)
                {
                    foreach (Connector connector in hNode.OutboundConnectors)
                    {
                        double delta = learningRate * connector.ToNode.Gradient * hNode.Val;
                        connector.Weight += (delta + (momentum * connector.WeightDelta));
                        connector.WeightDelta = delta;
                    }
                }
            }
        }
    }
}
