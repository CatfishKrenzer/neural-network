import express from 'express';
import https from 'https';
import http from 'http';
import train from './functionality/train.js';
import NeuralNetwork from './neural-network/neuralNetwork.js';

const app = express();
app.use(express.json());

const httpListenerPort = 9090;
const httpsListenerPort = 8443;
const basePath = '/neural-network';


const inputnodes = 28*28; // Pixels in the image
const hiddennodes = 100;
const outputnodes = 10;   // Training each node to predict different number
const learningrate = 0.2;
const threshold = 0.5;
let iter = 0;
const iterations = 5;

const neuralNetwork = new NeuralNetwork(inputnodes, hiddennodes, outputnodes, learningrate)

app.post(basePath + '/train', (req, res) => {
    res.send(train(req, neuralNetwork))
  })
app.post(basePath + '/predict', (req, res) => {
    res.send(neuralNetwork.predict(req.data))
})

const httpServer = http.createServer(app).listen(httpListenerPort, () => {
    console.log('app is listening at localhost:' + httpListenerPort);
  });