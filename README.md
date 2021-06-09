# NODLETEN

Node tfjs distribution learning tools

# Docs https://graphnull.github.io/nodleten/
# Dataset

Cache large data on disk with compression

![schema](doc/schema.svg)

## example

```javascript
let {Dataset, zip} = require('nodleten')

let inpSize = 28 * 28; 
const xDataset = new Dataset({shape:[inpSize]})
const yDataset = new Dataset({shape:[1]})
 
// generate dataset
for(let i=0;i!==300;i++){
  let data = new Float32Array(inpSize)

  xDataset.push(data)
  yDataset.push(new Float32Array(1))
}

const xyDataset = zip({xs: xDataset, ys: yDataset})
// .batch(4) //TODO
// .shuffle(4); //TODO

const model = tf.sequential({
  layers: [tf.layers.dense({units: 1, inputShape: [inpSize]})]
});

model.compile({optimizer: 'sgd', loss: 'meanSquaredError'});
await model.fitDataset(xyDataset, {
    epochs: 100,
    callbacks: {onEpochEnd: (epoch, logs) => {console.log( 'onEnd',epoch, logs.loss)}}
});

xDataset.destroy();
yDataset.destroy();
  
```

# TODO

- [x] Zip datasets
- [x] Write cache header
- [ ] Read cache header
- [ ] Batch, shuffle, repeat, skip, take
- [ ] Int32, Uint8 types
