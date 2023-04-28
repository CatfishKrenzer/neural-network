import * as math from "mathjs";
import * as fs from 'fs'
const mmap = math.map;
const rand = math.random;
const transp = math.transpose;
const mat = math.matrix;
const e = math.evaluate;
const sub = math.subtract;
const sqr = math.square;
const sum = math.sum;

//Reference: https://javascript.plainenglish.io/make-your-own-neural-network-app-with-plain-javascript-and-a-tiny-bit-of-math-js-30ab5ff4cbd5

class NeuralNetwork {
    constructor(inputNodes, hiddenNodes, outputNodes, learningRate, wih, who){
        this.inputNodes = inputNodes;
        this.hiddenNodes = hiddenNodes;
        this.outputNodes = outputNodes;
        this.learningRate = learningRate;
        this.trainingStats = {
            trainingStatus: '0.0%',
            currentAccuracy: '0.0%'
          }
        try{
            this.loadState();
        }
        catch(err){
            console.log(err)
            // WIH - weights of input-to-hidden layer
            // WHO - weights of hidden-to-output layer
            // If weights are not passed in, they will be randomly generated
            this.wih = wih || sub(mat(rand([hiddenNodes, inputNodes])), 0.5);
            this.who = who || sub(mat(rand([outputNodes, hiddenNodes])), 0.5);
            this.saveState();
        }
    
        // Sigmoid Function - Applies activation function (2nd param) to each element of input matrix (1st function)
        this.act = (matrix) => mmap(matrix, (x) => 1 / (1 + Math.exp(-x)));
    }

    cache = { loss: [] };

    static normalizeData = (data) => {
        // Convert from 0-255 to 0.01 to 1.00 to prevent saturation in sigmoid
        return data.map((e) => (e / 255) * 0.99 + 0.01);
    };

    getStats = ()=>this.trainingStats;
    setTrainingStatusPercent = (percentage)=>{
        this.trainingStats.trainingStatus = percentage;
    }
    setTrainingAccuracyPercent = (percentage)=>{
        this.trainingStats.currentAccuracy = percentage;
    }

    // Forward Propagation
    forward = (input) => { 
        // Z = WX
        // W = weights matrix of the input-to-hidden or the hidden-to-output layer
        // X = the matrix with the output values of the hidden layer or just the input values
        // A = sigmoid(Z) - forces the values of 
        const wih = this.wih;
        const who = this.who;
        const act = this.act;
    
        // update input to be a transposed matrix
        input = transp(mat([input]));
    
        //Input Layer -> Hidden Layer
        const hiddenIn = e("wih * input", { wih, input }); // dot product of matricies (input weights)
        const hiddenOut = act(hiddenIn);                   // Activation (Sigmoid) Function
    
        //Hidden Layer -> Output Layer
        const o_in = e("who * hiddenOut", { who, hiddenOut }); // dot product of matricies (output weights)
        const actual = act(o_in);                              // Activation (Sigmoid) Function
    
        //Store in "cache" for backward propogations
        this.cache.input = input;
        this.cache.h_out = hiddenOut;
        this.cache.actual = actual;
    
        //Prediction
        return actual;
    };

    // Backward Propogation 
    /*
    * Purpose: Assign the error (difference between prediction and reality) 
    *          to the weights of the layer before
    *   E - Error   A - Activation Function
    *   W - Weight  Z - W*X (Frm Forward Prop)
    * 
    *   dE   dE dA dZ
    *   -- = -- -- --
    *   dW   dA dZ dW
    */
    backward = (target) => { 
        const who = this.who;
        const input = this.cache.input;
        const h_out = this.cache.h_out;
        const actual = this.cache.actual;

        target = transp(mat([target]));

        // Gradient of Error Function WRT Activation Function
        const dEdA = sub(target, actual);

        // Gradient of Activation Function WRT Weighted Sum of Output Layer
        const o_dAdZ = e("actual .* (1 - actual)", {actual});

        // Gradient of Loss Function WRT Weights of Hidden Layer to Output Layer
        const dwho = e("(dEdA .* o_dAdZ) * h_out'", {dEdA, o_dAdZ, h_out});

        // Weighted error for Hidden Layer
        const h_err = e("who' * (dEdA .* o_dAdZ)", {who, dEdA, o_dAdZ});

        // Gradient of Activation Function WRT Weighted Sum of Hidden Layer
        const h_dAdZ = e("h_out .* (1 - h_out)", {h_out});

        // Gradient of Loss Function WRT Weights of Input to Hidden Layer
        const dwih = e("(h_err .* h_dAdZ) * input'", {h_err, h_dAdZ, input});  

        this.cache.dwih = dwih;
        this.cache.dwho = dwho;
        // this.cache.loss.push(sum(mmap(dEdA,sqr)));
    };

    update = () => {
        const wih = this.wih;
        const who = this.who;
        const dwih = this.cache.dwih;
        const dwho = this.cache.dwho;
        const r = this.learningRate;

        // Update the weights based on the calculated errors
        this.wih = e("wih + (r .* dwih)", { wih, r, dwih });
        this.wih = e("wih + (r .* dwih)", { wih, r, dwih });
        this.who = e("who + (r .* dwho)", { who, r, dwho });
    };

    saveState = ()=>{
        // Write the current config to a file
        fs.writeFileSync('./trainingValues.json', JSON.stringify(this));
        console.log('State Saved')
    }

    loadState = () => {
        let loadedState = fs.readFileSync('./trainingValues.json')
        loadedState = JSON.parse(loadedState.toString());
        this.wih = mat(loadedState.wih.data)
        this.who = mat(loadedState.who.data)
        this.trainingStats = loadedState.trainingStats;
        console.log('State Loaded')
    }
    
    predict = (input) => {
        return this.forward(input);
    };
    
    train = (input, target) => {
        try{
        this.forward(input);
        this.backward(target);
        this.update(true);
        }catch(err){
            console.log(err)
        }
    };
}

export default NeuralNetwork;