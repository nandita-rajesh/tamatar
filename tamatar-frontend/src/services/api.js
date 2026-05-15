/**
 * API Service for communicating with the backend
 * Handles image upload and disease prediction
 */

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:5000';

/**
 * Predicts disease from uploaded image
 * @param {File} imageFile - The image file to analyze
 * @returns {Promise<{disease: string, confidence: number}>}
 * @throws {Error} Network or API errors
 */
export const predictDisease = async (imageFile) => {
  try {
    // Create FormData to send image as multipart/form-data
    const formData = new FormData();
    formData.append('image', imageFile);

    // Make POST request to backend
    const response = await fetch(`${API_BASE_URL}/predict/`, {
      method: 'POST',
      body: formData,
    });

    // Handle non-200 responses
    if (!response.ok) {
      throw new Error(`Server error: ${response.status} ${response.statusText}`);
    }

    // Parse and return JSON response
    const data = await response.json();
    return data;
  } catch (error) {
    // Check if it's a network error (offline)
    if (error.message === 'Failed to fetch') {
      throw new Error('Network error. Please check your internet connection.');
    }
    throw error;
  }
};