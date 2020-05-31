let mobilenet;
let model;
const webcam = new Webcam(document.getElementById('wc'));
const dataset = new RPSDataset();
var withmaskSamples=0, withoutmaskSamples=0;
let isPredicting = false;

async function loadMobilenet() {
  const mobilenet = await tf.loadLayersModel('https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json');
  const layer = mobilenet.getLayer('conv_pw_13_relu');
  return tf.model({inputs: mobilenet.inputs, outputs: layer.output});
}

async function train() {
  dataset.ys = null;
  dataset.encodeLabels(3);
  model = tf.sequential({
    layers: [
      tf.layers.flatten({inputShape: mobilenet.outputs[0].shape.slice(1)}),
      tf.layers.dense({ units: 100, activation: 'relu'}),
      tf.layers.dense({ units: 3, activation: 'softmax'})
    ]
  });

  const optimizer = tf.train.adam(0.0001);
  model.compile({optimizer: optimizer, loss: 'categoricalCrossentropy'});
  let loss = 0;
  model.fit(dataset.xs, dataset.ys, {
    epochs: 10,
    callbacks: {
      onBatchEnd: async (batch, logs) => {
        loss = logs.loss.toFixed(5);
        console.log('LOSS: ' + loss);
        }
      }
   });
   model.save();
}


function handleButton(elem){
	switch(elem.id){
		case "0":
			withmaskSamples++;
			document.getElementById("withmasksamples").innerText = "With Mask Samples:" + withmaskSamples;
			break;
		case "1":
			withoutmaskSamples++;
			document.getElementById("withoutmasksamples").innerText = "With Out Samples:" + withoutmaskSamples;
			break;
	}
	label = parseInt(elem.id);
  const img = webcam.capture();
  console.log(img)
	dataset.addExample(mobilenet.predict(img), label);

}

async function predict() {
  while (isPredicting) {
      const predictedClass = tf.tidy(() => {
      const img = webcam.capture();

      const activation = mobilenet.predict(img);
      const predictions = model.predict(activation);
      return predictions.as1D().argMax();
    });

    const classId = (await predictedClass.data())[0];
    var predictionText = "";
    switch(classId){
		case 0:
			predictionText = "<font color='green'><b>The Person With Mask</b></font>";
			break;
		case 1:
			predictionText = "<font color='red'><b>The Person Without Mask</b></font>";
			break;

	}
	document.getElementById("prediction").innerHTML = predictionText;


    predictedClass.dispose();
    await tf.nextFrame();
  }
}


function doTraining(){
	train();
}

function startPredicting(){
	isPredicting = true;
	predict();
}

function stopPredicting(){
	isPredicting = false;
	predict();
}

async function init(){
	await webcam.setup();
	mobilenet = await loadMobilenet();
	tf.tidy(() => mobilenet.predict(webcam.capture()));

}

async function PredictingPre(){
  const MODEL_URL = "my_model.json"
  const model = await tf.loadLayersModel(MODEL_URL)
  while (isPredicting) {
  const predictedClass = tf.tidy(() => {
    const img = webcam.capture();

    const activation = mobilenet.predict(img);
    const predictions = model.predict(activation);
    return predictions.as1D().argMax();
  });
  const classId = (await predictedClass.data())[0];
    var predictionText = "";
    switch(classId){
		case 0:
			predictionText = "<font color='green'><b>The Person With Mask</b></font>";
			break;
		case 1:
			predictionText = "<font color='red'><b>The Person Without Mask</b></font>";
      break;
    }
    document.getElementById("prediction").innerHTML = predictionText;
  }



}
function startPredictingPre(){
  isPredicting = true;
  PredictingPre();
}

function stopPredictingPre(){
  isPredicting = false;
  PredictingPre();
}
init();


