/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

let tf = require('@tensorflow/tfjs')
let { opticalFlowFind } = require('./../src/kernels/opticatFlowKernels')


let offset = tf.tensor([0, 0, 0, 0, 0, 0, 0, 0], [2, 2, 2])
let pred = tf.ones([1, 222, 222, 3])
let current = tf.ones([1, 222, 222, 3])
let opFl = opticalFlowFind([offset, pred, current], { kernelSize: 7, step: 7 })
if (opFl.shape[0] !== 30 || opFl.shape[1] !== 30 || opFl.shape[2] !== 7 || opFl.shape[3] !== 7) {
    throw new Error('opFl.shape not valid')
}