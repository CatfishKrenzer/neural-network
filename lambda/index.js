import NeuralNetwork from './neuralNetwork.js';
const inputnodes = 28*28; // Pixels in the image
const hiddennodes = 100;
const outputnodes = 10;   // Training each node to predict different number
const learningrate = 0.2;

function formatPrediction(prediction) {
    const flattened = prediction.toArray().map((x) => x[0]);
    return flattened.indexOf(Math.max(...flattened));
  }
  
export const handler = (event, context) => {
    try{
        const neuralNetwork = new NeuralNetwork(inputnodes, hiddennodes, outputnodes, learningrate)
        const normalizedData = NeuralNetwork.normalizeData(event.predictData);
        const predictData = neuralNetwork.predict(normalizedData)
        
        return context.succeed({
            statusCode: 200,
            headers: {
                'Access-Control-Allow-Headers': 'Content-Type',
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Methods': 'OPTIONS, POST',
            },
            data: {predictData, prediction: formatPrediction(predictData)},
            isBase64Encoded: false,
        });
    }catch(error){
        console.log(error)
        return context.succeed({
            statusCode: 400,
            headers: {
                'Access-Control-Allow-Headers': 'Content-Type',
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Methods': 'OPTIONS, POST',
            },
            error: error.toString(),
            isBase64Encoded: false,
        });
    }
};