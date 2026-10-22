"use strict";

// All CPU-heavy work stays off the UI thread. No network requests or libraries.
importScripts("./engine.js");
self.onmessage = ({data}) => {
  try {
    const result = NeuralEngine.train(data, (progress) => self.postMessage({type: "progress", ...progress}));
    self.postMessage({type: "result", result});
  } catch (error) {
    self.postMessage({type: "error", error: error.message});
  }
};
