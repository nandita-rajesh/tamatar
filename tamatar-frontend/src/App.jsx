import { useState } from 'react';
import ImageUpload from './components/ImageUpload';
import ResultDisplay from './components/ResultDisplay';
import { predictDisease } from './services/api';
import diseaseMeta from './data/diseaseMeta.json';
import './App.css';
import tomatoLogo from "./assets/tomato.png";

function App() {
  const [selectedImage, setSelectedImage] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const handleImageSelect = (file) => {
    setSelectedImage(file);
    setResult(null);
    setError(null);
  };

  const handleSubmit = async () => {
    if (!selectedImage) {
      setError('Please select an image first');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      // Call real API: upload `selectedImage` as FormData (key: `image`)
      const data = await predictDisease(selectedImage);

      // Backend returns: { prediction: "Tomato_Late_blight", confidence: 0.40, class_index: 2 }
      // Use `prediction` directly and look it up in `diseaseMeta.json`.
      const predictedRaw = data && (data.prediction || data.class) ? (data.prediction || data.class) : '';
      const confidence = data && (typeof data.confidence === 'number' ? data.confidence : null);

      // fallback: if no prediction string but class_index present, map index -> backend class
      const indexToClass = {
        0: 'Tomato_Bacterial_spot',
        1: 'Tomato_Early_blight',
        2: 'Tomato_Late_blight',
        3: 'Tomato_Leaf_Mold',
        4: 'Tomato_Septoria_leaf_spot',
        5: 'Tomato_Spider_mites_Two_spotted_spider_mite',
        6: 'Tomato__Target_Spot',
        7: 'Tomato__Tomato_YellowLeaf__Curl_Virus',
        8: 'Tomato__Tomato_mosaic_virus',
        9: 'Tomato_healthy'
      };

      let key = predictedRaw;
      if (!key && data && typeof data.class_index === 'number') {
        key = indexToClass[data.class_index] || '';
      }

      const mapped = (key && diseaseMeta[key]) ? diseaseMeta[key] : {
        label: key || 'Unknown',
        severity: "Unknown",
        description: "No disease information available.",
        actions: [
          "Retake the photo",
          "Consult an expert"
        ]
      };

      setResult({
        label: mapped.label,
        severity: mapped.severity,
        description: mapped.description,
        actions: mapped.actions,
        confidence: confidence ? (confidence * 100).toFixed(1) : '0.0'
      });

    } catch (err) {
      setError("Something went wrong");
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setSelectedImage(null);
    setResult(null);
    setError(null);
  };

  return (
    <div className="app">

      {/* Header */}
      <header className="app-header">
        <img src={tomatoLogo} alt="tomato" className="logo-img" />

        <h1>Tomato Disease Detector</h1>
        <p>
          Upload a photo of your tomato plant leaf for instant disease detection
        </p>
      </header>

      <main className="app-main">
        <div className="main-card">

          {/* ALWAYS show image if selected */}
          <ImageUpload
            onImageSelect={handleImageSelect}
            image={selectedImage}
            onClear={handleReset}
          />

          {/* Show analyze button only before result */}
          {selectedImage && !result && (
            loading ? (
              <div className="loading-box">
                <span className="loader"></span>
                  <img src={tomatoLogo} alt="tomato" className="inline-icon" />
                  Analyzing tomato leaf...
              </div>
            ) : (
              <button className="submit-button" onClick={handleSubmit}>
                Analyze Image
              </button>
            )
          )}

          {/* Show result BELOW image */}
          {result && (
            <ResultDisplay
              result={result}
              onReset={handleReset}
            />
          )}

          {/* Error */}
          {error && error.trim() !== "" && (
            <div className="error-message">
              ⚠️ {error}
            </div>
          )}

        </div>
      </main>

    </div>
  );
}

export default App;