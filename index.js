import express from 'express';
import https from 'https';
import http from 'http';
import train from './functionality/train.js';
import NeuralNetwork from './neural-network/neuralNetwork.js';
import {Worker} from 'worker_threads';

const app = express();
app.use(express.json());

const httpListenerPort = 9090;
const httpsListenerPort = 8443;
const basePath = '/neural-network';

const inputnodes = 28*28; // Pixels in the image
const hiddennodes = 100;
const outputnodes = 10;   // Training each node to predict different number
const learningrate = 0.2;

const worker = new Worker('./workers/trainingWorker.js');

app.post(basePath + '/train', (req, res) => {
  // const neuralNetwork = new NeuralNetwork(inputnodes, hiddennodes, outputnodes, learningrate)
  // train('', neuralNetwork)

  worker.postMessage('train');
  res.send({status:'Training Started'})
})

app.post(basePath + '/predict', (req, res) => {
    const neuralNetwork = new NeuralNetwork(inputnodes, hiddennodes, outputnodes, learningrate)
    res.send(neuralNetwork.predict(req.data))
})

app.get(basePath + '/training-status', (req, res) => {
  const tmpNeuralNetwork = new NeuralNetwork(inputnodes, hiddennodes, outputnodes, learningrate)
  res.send(tmpNeuralNetwork.getStats())
})

const httpServer = http.createServer(app).listen(httpListenerPort, () => {
    console.log('app is listening at localhost:' + httpListenerPort);
  });