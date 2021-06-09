let start = async ()=>{

    let Dataset = require('./../src/index').Dataset;
var width = 1024;
var height = 768;
//var sharp = require('sharp/lib/index')
var path = require('path')
  
//download diode dataset
var normalDataset = new Dataset({name:'normal', shape:[height, width,3]});
//var rgbDataset = new Dataset({name:'rgb', shape:[height, width,3]});
let normals = {};
let rgbs = {};
normalDataset.push(new Float32Array(width*height*3))

    let http = require('http');
    let tar = require('tar');

      let len = 0;
      let bcount = 0;
      let interval = setInterval(() => {
        console.log(((bcount / len) * 100).toFixed(6)+'% ')
      }, 1000)
      let download1 = new Promise((resolve) => {
        const rr = http.request({
          host: 'diode-dataset.s3.amazonaws.com',
          path: '/train_normals.tar.gz'
        }, (req) => {
          len += parseInt(req.headers['content-length']);
          req.on('data', (data) => { bcount += data.length; })
          req
            .pipe(new tar.Parse({
              filter: (path) => path.slice(-4) === '.npy',
              onentry: (entry) => {
                let data = [];
                entry.on('data', (d) => { data.push(d) })
                entry.on('end', () => {
                  let nfile = Buffer.concat(data).slice(-(width * height * 3 * 4))
                  let ndata = new Float32Array(nfile.buffer.slice(nfile.byteOffset, nfile.byteOffset + nfile.byteLength));
                  normals[entry.path]=ndata;
                    console.log(entry.path);
                  //let rgbPath = entry.path.slice(0,-11)+'.png'
                  //console.log(1, rgbPath, entry.path)
                  //if(rgbs[rgbPath]){
                      normalDataset.push(normals[entry.path])
                      //rgbDataset.push(rgbs[rgbPath])
                      delete normals[entry.path]
                      //delete rgbs[rgbPath]
                  //}

                })

              }
            }))
          req.on('end', resolve)
        });
        rr.end()
      })
    //   let download2 = new Promise((resolve) => {
    //     const rr = http.request({
    //       host: 'diode-dataset.s3.amazonaws.com',
    //       path: '/train.tar.gz'
    //     }, (req) => {
    //       len += parseInt(req.headers['content-length']);
    //       req.on('data', (data) => { bcount += data.length; })
    //       req
    //         .pipe(new tar.Parse({
    //           filter: (path) => path.slice(-4) === '.png',
    //           onentry: (entry) => {
                  
    //             let data = [];
    //             entry.on('data', (d) => { data.push(d) })
    //             entry.on('end', () => {
    //                 try{
    //               let buf = Buffer.concat(data)

    //               sharp(buf).raw().toBuffer().then((img)=>{
    //                 let buf = new Float32Array(img)


    //                 rgbs[entry.path]=buf;
    //                 let normalPath = entry.path.slice(0, -4)+'_normal.npy'
    //                 //console.log(2, entry.path, normalPath)
    //                 if(normals[normalPath]){
    //                     console.log(2,entry.path)
    //                     normalDataset.push(normals[normalPath])
    //                     rgbDataset.push(rgbs[entry.path])
    //                     delete normals[normalPath]
    //                     delete rgbs[entry.path]
    //                 }
      

    //               })
    //             }catch(err){
    //                 console.log(err)
    //             }
    //             })
    //           }
    //         }))
    //       req.on('end', resolve)
    //     });
    //     rr.end()
    //   })
      await Promise.all([download1, null])
      clearInterval(interval)

await normalDataset.writeHeaderFile();
await rgbDataset.writeHeaderFile();
delete global.normals;
delete global.rgbs;
  
}


start().catch(console.error)