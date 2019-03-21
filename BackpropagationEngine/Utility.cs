using BackpropagationEngine.Nodes;
using System;
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

        public static void makeConnectionsBetweenNodes<T, K>(ref IList<T> fromLayer, ref IList<K> toLayer,
                                                ref List<Connector> connectors)
        {
            int iConnectorCounter = 0;

            if (typeof(T) == typeof(HiddenNode) && typeof(K) == typeof(OutputNode))
            {
                for (int i = 0; i < fromLayer.Count; i++)
                {
                    for (int j = 0; j < toLayer.Count; j++)
                    {
                        ((HiddenNode)(object)fromLayer[i]).OutboundConnectors.Add(connectors[iConnectorCounter]);
                        connectors[iConnectorCounter].FromNode = (HiddenNode)(object)fromLayer[i];
                        ((OutputNode)(object)toLayer[j]).InboundConnectors.Add(connectors[iConnectorCounter]);
                        connectors[iConnectorCounter].ToNode = (OutputNode)(object)toLayer[j];
                        iConnectorCounter++;
                    }
                }
            }
            else if (typeof(T) == typeof(HiddenNode) && typeof(K) == typeof(HiddenNode))
            {
                for (int i = 0; i < fromLayer.Count; i++)
                {
                    for (int j = 0; j < toLayer.Count; j++)
                    {
                        ((HiddenNode)(object)fromLayer[i]).OutboundConnectors.Add(connectors[iConnectorCounter]);
                        connectors[iConnectorCounter].FromNode = (HiddenNode)(object)fromLayer[i];
                        ((HiddenNode)(object)toLayer[j]).InboundConnectors.Add(connectors[iConnectorCounter]);
                        connectors[iConnectorCounter].ToNode = (HiddenNode)(object)toLayer[j];
                        iConnectorCounter++;
                    }
                }
            }
        }

    }
}
