import NeuralNetwork from '../neural-network/neuralNetwork';

const inputnodes = 28*28; // Pixels in the image
const hiddennodes = 100;
const outputnodes = 10;   // Training each node to predict different number
const learningrate = 0.2;
const threshold = 0.5;
let iter = 0;
const iterations = 5;

const neuralNetwork = new NeuralNetwork(inputnodes, hiddennodes, outputnodes, learningrate)

const train = (data) => {
}

export default train