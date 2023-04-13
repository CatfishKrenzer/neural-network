import express from 'express';
import https from 'https';
import http from 'http';
import train from './functionality/train.js';

const app = express();
app.use(express.json());

const httpListenerPort = 9090;
const httpsListenerPort = 8443;
const basePath = '/neural-network';


app.post(basePath + '/train', (req, res) => {
    res.send(train(req))
  })
const httpServer = http.createServer(app).listen(httpListenerPort, () => {
    console.log('app is listening at localhost:' + httpListenerPort);
  });