/* Backpropagation is the process of adjusting the weight of a connector
* that joins two nodes based on an error value. It begins with the calculation 
* of the gradients of the errors at each of the final output layer nodes. 
* Then the output gradients are applied to the weights between the final 
* hidden and output layer nodes.
* 1) calculate the output gradients by multiplying the derivative of the 
* 	output activations times the output error (defined as actual - calculated)
* 2) calculate the hidden layer gradients by multiplying the derivative of the 
* 	hidden output activation times the corresponding output gradient
* 
* For the hidden-to-output weights:
* The updated_weight is the current_weight  - (output_gradient * hidden_layer_node_output)

*/