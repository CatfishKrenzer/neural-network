import NeuralNetwork from '../neural-network/neuralNetwork.js';
import { readFile }  from 'fs/promises';

const iterations = 5;
const trainingData = [];
const trainingLabels = [];
const testData = [];
const testLabels = [];

/* path to the data sets */
const trainingDataPath = "./mnist/mnist_train.csv";
const testDataPath = "./mnist/mnist_test.csv";

function prepareData(rawData, target, labels) {
    rawData = rawData.split("\n"); // create an array where each element correspondents to one line in the CSV file
    rawData.pop(); // remove the last element which is empty because it refers to a last blank line in the CSV file

    rawData.forEach((current) => {
        let sample = current.split(",").map((x) => +x); // create an array where each element has a gray color value

        labels.push(sample[0]); // extract the first element of the sample which is (mis)used as the label
        sample.shift(); // remove the first element

        sample = NeuralNetwork.normalizeData(sample);

        target.push(sample);
    });
}


function formatPrediction(prediction) {
    const flattened = prediction.toArray().map((x) => x[0]);
    return flattened.indexOf(Math.max(...flattened));
}

function test(inTraining = false, neuralNetwork) { 
  
    let correctPredicts = 0;
    
    testData.forEach((current, index) => {
      const actual = testLabels[index];
      
      const predict = formatPrediction(neuralNetwork.predict(current));
      predict === actual ? correctPredicts++ : null;
  
      /* check if training is complete */
      /* if test is called from within training and the training is not complete yet, continue training */
      if (index >= testData.length - 1 && inTraining) {
        console.log(`Prediction Accuracy: ${((correctPredicts*100)/testData.length).toFixed(4)}%`)
        train('', neuralNetwork);
      }     
    });
  }

const train = async (data, neuralNetwork) => {
    const trainCSV = await readFile(trainingDataPath, 'utf8');
    if (trainCSV) {
        prepareData(trainCSV, trainingData, trainingLabels);
    }
    const testCSV = await readFile(testDataPath, 'utf8');
    if (testCSV) {
        prepareData(testCSV, testData, testLabels);
    }

    // TRAINING TIME
    let iter = 0;
    if (iter < iterations) {
        iter++;
        trainingData.forEach((current, index) => {
          if(index % 150000 === 0){
            console.log(`Training Data ${index} of ${trainingData.length} - Iteration ${iter} of ${iterations} - ${((100 * ((iter - 1) * trainingData.length + index))/(trainingData.length * iterations)).toFixed(4)}%`)
          }

           // Set the value expected for each as "hot" or 0.99 and the rest as 0
           const label = trainingLabels[index];
           const oneHotLabel = Array(10).fill(0);
           oneHotLabel[label] = 0.99;
    
           neuralNetwork.train(current, oneHotLabel);
    
           /* check if the end of the training iteration is reached */
           if (index === trainingData.length - 1) {
             test(true, neuralNetwork); // true to signal "test" that it is called from within training 
           }
        });
      }
}

export default train