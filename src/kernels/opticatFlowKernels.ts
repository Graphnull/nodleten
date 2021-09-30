import * as tf from '@tensorflow/tfjs-core'
//let gles = require('node-gles')
//let webgl = require('@tensorflow/tfjs-backend-webgl');

let registerKernel = tf.registerKernel;
//let gl = gles.createWebGLRenderingContext()
//tf.env().set('WEBGL_VERSION', 1);

// tf.env().set('WEBGL_RENDER_FLOAT32_ENABLED', true);
// tf.env().set('WEBGL_DOWNLOAD_FLOAT_ENABLED', true);
// tf.env().set('WEBGL_FENCE_API_ENABLED', true);  // OpenGL ES 3.0 and higher..
// tf.env().set('WEBGL_MAX_TEXTURE_SIZE', gl.getParameter(gl.MAX_TEXTURE_SIZE));

//webgl.setWebGLContext(2, gl);

// tf.registerBackend('webgl', () => {
//     console.log('registerBackend: ');
//     // TODO(kreeger): Consider moving all GL creation here. However, weak-ref to
//     // GL context tends to cause an issue when running unit tests:
//     // https://github.com/tensorflow/tfjs/issues/1732
//     return new webgl.MathBackendWebGL(new webgl.GPGPUContext(gl));
// }, 3 );

let glslProgramForCpu = ({ variableNames, outputShape, userCode }: { variableNames: string[], outputShape: number[], userCode: string }) => {
    return (...args: tf.Tensor[]) => {
        if (variableNames.length !== args.length) {
            throw new Error('variableNames.length!==' + variableNames.length);
        }
        let size = 1;
        outputShape.forEach(n => size *= n);
        let shapeSize = outputShape.map((v, i, arr) => { let size = 1; arr.slice(i + 1).forEach(v => size *= v); return size })
        let out = new Float32Array(size)

        let params = ['out', 'int', 'abs', 'float', 'floor']
        variableNames.forEach(p => {
            params.push(p)
        })

        let code = userCode.replace(/void\ main\(\)/gi, '').replace(/float\ /gi, 'let ').replace(/float\(/gi, '(').replace(/ivec3\ /gi, 'let ').replace(/ivec4\ /gi, 'let ').replace(/int\ /gi, 'let ')
        code = `for (; i !== ${out.length}; i++)` + code
        code = `let setOutput = (val) => out[i] = val;\n` + code
        code = `let getOutputCoords = () => [
            Math.floor(i / ${shapeSize[0]}) % ${outputShape[0]},
            Math.floor(i / ${shapeSize[1]}) % ${outputShape[1]},
            Math.floor(i / ${shapeSize[2]}) % ${outputShape[2]},
            Math.floor(i / ${shapeSize[3]}) % ${outputShape[3]},
            ];\n` + code
        let passArgs: any[] = [out, Math.round, Math.abs, (v: unknown) => v, Math.floor]
        args.forEach((aa, j) => {
            let s = aa.shape
            let t = aa.dataSync();
            let shapeSizet = s.map((v, i, arr) => { let size = 1; arr.slice(i + 1).forEach(v => size *= v); return size })

            if (s.length === 4) {
                passArgs.push(t)
                code = `let get${variableNames[j]}=(x,y,z)=>${variableNames[j]}[x * ${shapeSizet[1]} + y * ${shapeSizet[2]} + z];\n` + code
            } else {
                passArgs.push(t)
                code = `let get${variableNames[j]}=(x,y,z)=>${variableNames[j]}[x * ${shapeSizet[0]} + y * ${shapeSizet[1]} + z];\n` + code
            }
        })

        code = 'let i=0;\n' + code
        params.push(code)
        let main = new Function(...params);

        main(...passArgs)

        return tf.tensor(out, outputShape.slice(), 'float32')
    }
}

const opticalFlowFindKernel = (inputShape: number[], backSize: [number, number], kernelSize: number, startOffset: number, step: number) => {
    return {
        variableNames: ['BACK', 'PRED', 'NEXT'],
        outputShape: inputShape.slice(),
        userCode: `//GLSL
        void main() {
          ivec4 coords = getOutputCoords();
          int dx = coords[3];
          int dy = coords[2];
          int cx = coords[1] + int(getBACK(int(floor(float(coords[0])/${inputShape[0]}.0*${backSize[0]}.0)),int(floor(float(coords[1])/${inputShape[1]}.0*${backSize[1]}.0)),0));
          int cy = coords[0] + int(getBACK(int(floor(float(coords[0])/${inputShape[0]}.0*${backSize[0]}.0)),int(floor(float(coords[1])/${inputShape[1]}.0*${backSize[1]}.0)),1));
          float acc = 0.0;
          for(int y=0;y!=${kernelSize};y++){
            for(int x = 0;x!=${kernelSize};x++){
              float r = getPRED(cy*${step}+${startOffset}+y, cx*${step}+${startOffset}+x, 0);
              float g = getPRED(cy*${step}+${startOffset}+y, cx*${step}+${startOffset}+x, 1);
              float b = getPRED(cy*${step}+${startOffset}+y, cx*${step}+${startOffset}+x, 2);
              acc+=abs(r-getNEXT(cy*${step}+dy+y, cx*${step}+dx+x, 0));
              acc+=abs(g-getNEXT(cy*${step}+dy+y, cx*${step}+dx+x, 1));
              acc+=abs(b-getNEXT(cy*${step}+dy+y, cx*${step}+dx+x, 2));
            }
          }
          setOutput((acc)/(${kernelSize * kernelSize}.0*1.0));
          }
      `
    }
}

const opticalFlowFindWebgl = (params: any) => {
    let kernelSize = params.attrs.kernelSize || 7;
    let startOffset = Math.floor(kernelSize / 2)
    let step = params.attrs.step || 1;
    let px = Math.floor((params.inputs.current.shape[2] - kernelSize - startOffset) / step)
    let py = Math.floor((params.inputs.current.shape[1] - kernelSize - startOffset) / step)
    let inputShape = [py, px, kernelSize, kernelSize]
    let backSize: [number, number] = [params.inputs.offset.shape[0], params.inputs.offset.shape[1]];
    let program = opticalFlowFindKernel(inputShape, backSize, kernelSize, startOffset, step);
    return params.backend.runWebGLProgram(program, [params.inputs.offset, params.inputs.pred, params.inputs.current], 'float32');
}

registerKernel({
    backendName: 'webgl',
    kernelName: 'opticalFlowFind',
    kernelFunc: opticalFlowFindWebgl
})

const opticalFlowFindCpu = (params: any) => {
    let kernelSize = params.attrs.kernelSize || 7;
    let startOffset = Math.floor(kernelSize / 2)
    let step = params.attrs.step || 1;
    let px = Math.floor((params.inputs.current.shape[2] - kernelSize - startOffset) / step)
    let py = Math.floor((params.inputs.current.shape[1] - kernelSize - startOffset) / step)
    let inputShape = [py, px, kernelSize, kernelSize]
    let backSize: [number, number] = [params.inputs.offset.shape[0], params.inputs.offset.shape[1]];
    let program = glslProgramForCpu(opticalFlowFindKernel(inputShape, backSize, kernelSize, startOffset, step));
    return program(params.inputs.offset, params.inputs.pred, params.inputs.current);

}
registerKernel({
    backendName: 'cpu',
    kernelName: 'opticalFlowFind',
    kernelFunc: opticalFlowFindCpu
})
registerKernel({
    backendName: 'tensorflow',
    kernelName: 'opticalFlowFind',
    kernelFunc: opticalFlowFindCpu
})
registerKernel({
    backendName: 'tensorflow-gpu',
    kernelName: 'opticalFlowFind',
    kernelFunc: opticalFlowFindCpu
})
module.exports.opticalFlowFind = ([offset, pred, current]: tf.Tensor[], { kernelSize = 7, step = 1 }: { kernelSize: number, step: number } = { kernelSize: 7, step: 1 }) => tf.engine().runKernel('opticalFlowFind', { offset, pred, current }, { kernelSize, step })

const blurKernel = (inputShape: number[]) => {
    return {
        variableNames: ['IMG'],
        outputShape: inputShape.slice(),
        userCode: `
        void main() {
          ivec3 coords = getOutputCoords();

          int d = coords[2];
          float acc = 0.0;
          
          acc+=getIMG(coords[0]*2-1+1,coords[1]*2-1+1,d);
          acc+=getIMG(coords[0]*2-0+1,coords[1]*2-1+1,d);
          acc+=getIMG(coords[0]*2+1+1,coords[1]*2-1+1,d);
          acc+=getIMG(coords[0]*2-1+1,coords[1]*2-0+1,d);
          acc+=getIMG(coords[0]*2-0+1,coords[1]*2-0+1,d);
          acc+=getIMG(coords[0]*2+1+1,coords[1]*2-0+1,d);
          acc+=getIMG(coords[0]*2-1+1,coords[1]*2+1+1,d);
          acc+=getIMG(coords[0]*2-0+1,coords[1]*2+1+1,d);
          acc+=getIMG(coords[0]*2+1+1,coords[1]*2+1+1,d);
          
          setOutput(acc/(9.0));
          }
      `
    }
}

const blur = (params: any) => {
    let s = params.inputs.a.shape;
    let program = blurKernel([Math.floor((s[0] - 2) / 2), Math.floor((s[1] - 2) / 2), s[2]]);
    return params.backend.runWebGLProgram(program, [params.inputs.a], 'float32');
}
registerKernel({
    backendName: 'webgl',
    kernelName: 'blur',
    kernelFunc: blur
})

const blurCpu = (params: any) => {
    let s = params.inputs.a.shape;

    let program = glslProgramForCpu(blurKernel([Math.floor(s[0] / 2), Math.floor(s[1] / 2), s[2]]));
    return program(params.inputs.a);

}
registerKernel({
    backendName: 'cpu',
    kernelName: 'blur',
    kernelFunc: blurCpu
})
registerKernel({
    backendName: 'tensorflow',
    kernelName: 'blur',
    kernelFunc: blurCpu
})
registerKernel({
    backendName: 'tensorflow-gpu',
    kernelName: 'blur',
    kernelFunc: blurCpu
})
module.exports.blur = (a: tf.Tensor) => tf.engine().runKernel('blur', { a })
