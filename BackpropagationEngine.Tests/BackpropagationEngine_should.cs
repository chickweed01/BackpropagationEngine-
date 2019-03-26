using BackpropagationEngine.Activation;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NN.Utility;
using NN.Utility.Nodes;
using System;
using System.Collections.Generic;
using static BackpropagationEngine.BackpropagationEngine;

namespace BackpropagationEngine.Tests
{
    [TestClass]
    public class BackpropagationEngine_should
    {
        int numHidden;
        int numOutputs;
        double[] targetValues;
        double momentum;
        double learningRate;
        IList<HiddenNode> hiddenLayer;
        IList<OutputNode> outputLayer;
        IList<Connector> hiddenToOutputConnectors;

        //public static Activation activation = new Activation();
        //public static BackpropagationEngine..ApplyActivationDelegate logSigActivation = activation.Sigmoid;
        //public static ApplyActivationForVector softMaxActivation = activation.Softmax;

        [TestInitialize()]
        public void Initialize()
        {
            learningRate = .05;
            momentum = .01;
            numHidden = 2;
            numOutputs = 2;
            targetValues = new double[] { .25, .75 };

            hiddenLayer = Utility.GenericNodeFactory.CreateGenericNode<HiddenNode>(numHidden);
            outputLayer = Utility.GenericNodeFactory.CreateGenericNode<OutputNode>(numOutputs);

            Utility.createConnectors(4, out hiddenToOutputConnectors);

            foreach (HiddenNode hNode in hiddenLayer)
            {
                hNode.OutboundConnectors = new List<Connector>();
            }

            foreach (OutputNode oNode in outputLayer)
            {
                oNode.InboundConnectors = new List<Connector>();
            }

            /* Hook together the connectors between the hidden and output layers */
            Utility.makeConnectionsBetweenNodes(ref hiddenLayer, ref outputLayer, ref hiddenToOutputConnectors, 0);

            //mock values for activated hidden layer sums
            hiddenLayer[0].Val = .5678;
            hiddenLayer[1].Val = .4322;
        }

        [TestMethod]
        public void TestCalculateActivationsForOutputLayer()
        {
            /* Calculating the activations for a particular layer
             * requires that the sum at each node in the
             * layer be calculated first, then the activation
             * function is applied to each node's sum. */

            calculateActivatedSumsForLayer(ref outputLayer, ActivationAlgorithm.Sigmoid);
            Console.WriteLine("OutputLayer activation values using ActivationAlgorithm.Sigmoid {0}, {1}",
                                outputLayer[0].Val.ToString("F4"), outputLayer[1].Val.ToString("F4"));

            calculateActivatedSumsForLayer(ref outputLayer, ActivationAlgorithm.SoftMax);
            Console.WriteLine("OutputLayer activation values using ActivationAlgorithm.SoftMax {0}, {1}",
                                outputLayer[0].Val.ToString("F4"), outputLayer[1].Val.ToString("F4"));
        }

        [TestMethod]
        public void testCalculateGradientsAtOutputLayer()
        {
            /*
             * Calculate the output gradients by multiplying the derivative of the 
             * output activations times the output errors (defined as target - calculated) */

            /* mock some values for activated output layer values
             * to make this test independent of other tests */
            outputLayer[0].Val = 0.5066;
            outputLayer[1].Val = 0.5097;

            calculateGradientsForOutputLayer(ref outputLayer, targetValues, ActivationAlgorithm.Sigmoid);

            Console.WriteLine("OutputLayer gradients {0}, {1}", outputLayer[0].Gradient.ToString("F4"),
                                                                outputLayer[1].Gradient.ToString("F4"));
        }

        [TestMethod]
        public void testCalculateGradientsAtHiddenLayer()
        {
            /* mock some values for activated output layer values
             * to make this test independent of other tests */
            //same for output layer activated values
            outputLayer[0].Val = 0.5066;
            outputLayer[1].Val = 0.5097;
            // same for output layer gradients
            outputLayer[0].Gradient = -0.0641;
            outputLayer[1].Gradient = 0.0601;

            calculateGradientsForHiddenLayer(ref hiddenLayer, outputLayer, ActivationAlgorithm.HyperTan);

            Console.WriteLine("HiddenLayer gradients {0}, {1}", hiddenLayer[0].Gradient.ToString("F4"),
                                                                hiddenLayer[1].Gradient.ToString("F4"));
        }

        [TestMethod]
        public void testAdjustWeightsForLayer()
        {
            /*
             * Required inputs:
             * momentum, learningRate, 
             * activated sums at the layer, 
             * gradients at the layer
             */

            // For a node's outbound connector weight:
            // delta = learningRate * downstream node's Gradient * node's activated sum

            // outboundConnector.weight += delta + momentum

            double[] outputGradient = new double[numOutputs];

            /* mock some values for activated output layer values
             * to make this test independent of other tests */
            outputLayer[0].Val = 0.5066;
            outputLayer[1].Val = 0.5097;

            calculateGradientsForOutputLayer(ref outputLayer, targetValues,
                                            ActivationAlgorithm.Sigmoid);

            //print pre-adjusted weights
            Console.WriteLine("Hidden layer pre-adjusted weights {0}, {1}, {2}, {3}",
                                                                    hiddenLayer[0].OutboundConnectors[0].Weight,
                                                                    hiddenLayer[0].OutboundConnectors[1].Weight,
                                                                    hiddenLayer[1].OutboundConnectors[0].Weight,
                                                                    hiddenLayer[1].OutboundConnectors[1].Weight);

            adjustWeightsAtLayer(ref hiddenLayer, learningRate, momentum);

            //print adjusted weights
            Console.WriteLine("Hidden layer adjusted weights {0}, {1}, {2}, {3}",
                                                                    hiddenLayer[0].OutboundConnectors[0].Weight,
                                                                    hiddenLayer[0].OutboundConnectors[1].Weight,
                                                                    hiddenLayer[1].OutboundConnectors[0].Weight,
                                                                    hiddenLayer[1].OutboundConnectors[1].Weight);
        }

        private void calculateGradientsForHiddenLayer<T>(ref IList<HiddenNode> hiddenLayer, IList<T> downstreamLayer, ActivationAlgorithm activationAlgorithm)
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

        private void calculateGradientsForOutputLayer(ref IList<OutputNode> outputLayer, double[] targetValues, ActivationAlgorithm activationAlgorithm)
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
                        IActivationFormulaDelegate activationFormulaDelegate = sigmoid;
                        oNode.Val = activationFormulaDelegate.applyActivation(oNode.Val);
                    }
                    else if (activationAlgorithm == ActivationAlgorithm.HyperTan)
                    {
                        var hTan = new HyperTan();
                        IActivationFormulaDelegate activationFormulaDelegate = hTan;
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
                    IActivationFormulaDelegate activationFormulaDelegate = softMax;
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
                        IActivationFormulaDelegate activationFormulaDelegate = sigmoid;
                        hNode.Val = activationFormulaDelegate.applyActivation(hNode.Val);
                    }
                    else if (activationAlgorithm == ActivationAlgorithm.HyperTan)
                    {
                        var hTan = new HyperTan();
                        IActivationFormulaDelegate activationFormulaDelegate = hTan;
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
                    IActivationFormulaDelegate activationFormulaDelegate = softMax;
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
