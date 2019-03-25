using NN.Utility;
using NN.Utility.Nodes;
using System;
using System.Collections.Generic;

namespace BackpropagationEngineClient
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("\nBegin UML-based Artificial Neural Network demo\n");

            int numInput = 3;
            int numHidden = 4;
            int numOutput = 2;
            int numBiasNodes = numHidden + numOutput;
            int numberOfConnectors = (numInput * numHidden) + (numHidden * numOutput) + numBiasNodes;
            int epoch = 0;
            int maxEpochs = 400;
            var learningRate = 0.05;
            var momentum = 0.01;
            int iConnectorCounter = 0;
            int numHiddenBias = numHidden;
            int numOutputBias = numOutput;
            double[] xValues = new double[] { 1.0, 2.0, 3.0 };
            double[] outputs = new double[numOutput];
            double[] targetValues = new double[numOutput];
            IList<InputNode> inputLayer;
            IList<HiddenNode> hiddenLayer;
            IList<OutputNode> outputLayer;
            IList<BiasNode> hiddenBiasNodes;
            IList<BiasNode> outputBiasNodes;
            IList<Connector> connectors;

            try
            {
                inputLayer = Utility.GenericNodeFactory.CreateGenericNode<InputNode>(numInput);
                hiddenLayer = Utility.GenericNodeFactory.CreateGenericNode<HiddenNode>(numHidden);
                outputLayer = Utility.GenericNodeFactory.CreateGenericNode<OutputNode>(numOutput);
                hiddenBiasNodes = Utility.GenericNodeFactory.CreateGenericNode<BiasNode>(numHidden);
                outputBiasNodes = Utility.GenericNodeFactory.CreateGenericNode<BiasNode>(numOutput);

                //specify target output values
                targetValues[0] = .25;
                targetValues[1] = .75;

                //assign a value to each input node
                inputLayer[0].Val = 1.0;
                inputLayer[1].Val = 2.0;
                inputLayer[2].Val = 3.0;

                /* create 26 connector nodes to hook up the 
                 * input to hidden and hidden to output nodes and the bias nodes. */
                Utility.createConnectors(numberOfConnectors, out connectors);

                // input nodes only have an output connector list
                foreach (InputNode iNode in inputLayer)
                {
                    iNode.OutboundConnectors = new List<Connector>();
                }

                //hidden nodes have both input and output connector lists
                foreach (HiddenNode hNode in hiddenLayer)
                {
                    hNode.InboundConnectors = new List<Connector>();
                    hNode.OutboundConnectors = new List<Connector>();
                }

                //output nodes have only an input connector list
                foreach (OutputNode oNode in outputLayer)
                {
                    oNode.InboundConnectors = new List<Connector>();
                }

                //set connectors for Hidden Bias nodes
                foreach (BiasNode bNode in hiddenBiasNodes)
                {
                    bNode.OutboundConnectors = new List<Connector>();
                }

                //set connectors for Output Bias nodes
                foreach (BiasNode bNode in outputBiasNodes)
                {
                    bNode.OutboundConnectors = new List<Connector>();
                }

                Utility.makeConnectionsBetweenNodes(ref inputLayer, ref hiddenLayer, ref connectors, 0);
                Utility.makeConnectionsBetweenNodes(ref hiddenLayer, ref outputLayer, ref connectors, (numInput * numHidden));
                Utility.makeConnectionsBetweenNodes(ref hiddenBiasNodes, ref hiddenLayer, ref connectors, 
                                                    (numInput * numHidden) + (numHidden * numOutput));
                Utility.makeConnectionsBetweenNodes(ref outputBiasNodes, ref outputLayer, ref connectors, 
                                                    (numInput * numHidden) + (numHidden * numOutput) + numHidden);

                Console.WriteLine("\nInputs are:");
                Utility.ShowVector(xValues, 3, 1, true);

                Console.WriteLine("\nSetting default weights and biases:");
                Utility.ShowVector(Utility.getWeights(connectors), 8, 2, true);

                /*** loop until maxEpochs is reached ***/
                while (epoch <= maxEpochs)
                {
                    BackpropagationEngine.BackpropagationEngine backpropagationEngine = new BackpropagationEngine.BackpropagationEngine();
                    backpropagationEngine.doBackProp(learningRate, momentum, targetValues, ref inputLayer, ref hiddenLayer, ref outputLayer);

                    if (epoch % 100 == 0)
                    {
                        Console.WriteLine("\nEpoch = " + epoch.ToString() + " curr outputs = ");

                        outputs[0] = outputLayer[0].Val;
                        outputs[1] = outputLayer[1].Val;
                        Utility.ShowVector(outputs, 2, 4, true);

                        Console.WriteLine("\nIntermediate weights and biases:");
                        Utility.ShowVector(Utility.getWeights(connectors), 8, 2, true);
                    }

                    ++epoch;
                }// *** end loop ***

                Console.WriteLine("\nFinal weights and biases:");
                Utility.ShowVector(Utility.getWeights(connectors), 8, 2, true);
            }
            catch (Exception ex)
            {
                Console.WriteLine(ex.Message);
            }

            Console.WriteLine("\nEnd UML-based Artificial Neural Network demo\n");
            Console.ReadLine();
        }        
    }
}
