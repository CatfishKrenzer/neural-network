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
    constructor(inputNodes, hiddenNodes /*Why?*/, outputNodes, learningRate, wih, who){
        this.inputNodes = inputNodes;
        this.hiddenNodes = hiddenNodes;
        this.outputNodes = outputNodes;
        this.learningRate = learningRate;
        
        // WIH - weights of input-to-hidden layer
        // WHO - weights of hidden-to-output layer
        // If weights are not passed in, they will be randomly generated
        this.wih = wih || sub(mat(rand([hiddennodes, inputnodes])), 0.5);
        this.who = who || sub(mat(rand([outputnodes, hiddennodes])), 0.5);
    
        // Sigmoid Function - Applies activation function (2nd param) to each element of input matrix (1st function)
        this.act = (matrix) => mmap(matrix, (x) => 1 / (1 + Math.exp(-x)));
    }

    cache = { loss: [] };

    static normalizeData = (data) => { /*...*/ }

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
    backward = (input, target) => { 
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
        this.cache.loss.push(sum(sqr(dEdA)));
    };

    update = () => {
        const wih = this.wih;
        const who = this.who;
        const dwih = this.cache.dwih;
        const dwho = this.cache.dwho;
        const r = this.learningrate;

        // Update the weights based on the calculated errors
        this.wih = e("wih + (r .* dwih)", { wih, r, dwih });
        this.who = e("who + (r .* dwho)", { who, r, dwho });
    };
    
    predict = (input) => {
        return this.forward(input);
    };
    
    train = (input, target) => {
        this.forward(input);
        this.backward(target);
        this.update();
    };
}

export default NeuralNetwork;