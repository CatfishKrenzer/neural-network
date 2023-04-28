import { parentPort } from "worker_threads";
import NeuralNetwork from '../neural-network/neuralNetwork.js';
import train from '../functionality/train.js';

// Move to const or in network file
const inputnodes = 28*28; // Pixels in the image
const hiddennodes = 100;
const outputnodes = 10;   // Training each node to predict different number
const learningrate = 0.2;

const neuralNetwork = new NeuralNetwork(inputnodes, hiddennodes, outputnodes, learningrate)

parentPort.on("message", message => {
    console.log('Training Started')
    train(message, neuralNetwork)
});
