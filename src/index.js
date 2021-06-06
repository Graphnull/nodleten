var lz4 = require('lz4');
let fs = require('fs')
const {
    Worker, isMainThread, parentPort, workerData
  } = require('worker_threads');



if (isMainThread) {
    const worker = new Worker(__filename, {

    });
      worker.on('message', resolve);
      worker.on('error', reject);
      worker.on('exit', (code) => {
        if (code !== 0)
          reject(new Error(`Worker stopped with exit code ${code}`));
      });

}else{
    
    let compressedlz = new Buffer(1024*1024*4)
    let compressedlz2 = new Buffer(1024*1024*4)
    let compressedlz3 = new Buffer(1024*1024*4)

    let lzTempOut = new Buffer(1024*1024*4)
    let lzTempOut2 = new Buffer(1024*1024*4)
    let lzTempOut3 = new Buffer(1024*1024*4)

    //from benchmark
    var compressedSize = lz4.encodeBlock(compressedlz3.subarray(0, compressedSize), compressedlz2)
    compressedSize = lz4.encodeBlock(compressedlz2.subarray(0, compressedSize), compressedlz)



    var uncompressedSize = lz4.decodeBlock(compressedlz.subarray(0, compressedSize), lzTempOut)
    uncompressedSize = lz4.decodeBlock(lzTempOut.subarray(0, uncompressedSize), lzTempOut2)
}


