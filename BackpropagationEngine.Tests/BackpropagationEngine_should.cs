using BackpropagationEngine;
using BackpropagationEngine.Nodes;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;
using static BackpropagationEngine.Activation;
using static BackpropagationEngine.BackpropagationEngine;

namespace BackpropagationEngineClient.Tests
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
        List<Connector> hiddenToOutputConnectors;

        public static Activation activation = new Activation();
        public static ApplyActivationForScalar logSigActivation = activation.Sigmoid;
        public static ApplyActivationForVector softMaxActivation = activation.Softmax;

        [TestInitialize()]
        public void Initialize()
        {
            learningRate = .05;
            momentum = .01;
            numHidden = 2;
            numOutputs = 2;
            targetValues = new double[] { .25, .75};

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
            Utility.makeConnectionsBetweenNodes(ref hiddenLayer, ref outputLayer, ref hiddenToOutputConnectors);

            //mock values for activated hidden layer sums
            hiddenLayer[0].Val = logSigActivation(.1234);
            hiddenLayer[1].Val = logSigActivation(.8766);
        }

        [TestMethod]
        public void testCalculateActivationsForOutputLayer()
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
    }
}
