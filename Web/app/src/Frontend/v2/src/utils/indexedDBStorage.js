// Create a utility file for IndexedDB operations

const dbName = "videoAnomalyDetection";
const storeName = "videos";
const version = 1;

// Initialize the database
export const initDB = () => {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(dbName, version);
    
    request.onupgradeneeded = (event) => {
      const db = event.target.result;
      if (!db.objectStoreNames.contains(storeName)) {
        db.createObjectStore(storeName);
      }
    };
    
    request.onerror = (event) => reject(event.target.error);
    request.onsuccess = (event) => resolve(event.target.result);
  });
};

// Store a video
export const storeVideo = async (key, videoBlob, metadata) => {
  try {
    const db = await initDB();
    return new Promise((resolve, reject) => {
      const transaction = db.transaction(storeName, "readwrite");
      const store = transaction.objectStore(storeName);
      const request = store.put({
        blob: videoBlob,
        metadata: metadata,
        timestamp: Date.now()
      }, key);
      
      request.onerror = (event) => reject(event.target.error);
      request.onsuccess = (event) => resolve(event.target.result);
    });
  } catch (error) {
    console.error("Error storing video in IndexedDB:", error);
    throw error;
  }
};

// Retrieve a video
export const getVideo = async (key) => {
  try {
    const db = await initDB();
    return new Promise((resolve, reject) => {
      const transaction = db.transaction(storeName, "readonly");
      const store = transaction.objectStore(storeName);
      const request = store.get(key);
      
      request.onerror = (event) => reject(event.target.error);
      request.onsuccess = (event) => resolve(event.target.result);
    });
  } catch (error) {
    console.error("Error retrieving video from IndexedDB:", error);
    throw error;
  }
};

// Clear a video
export const clearVideo = async (key) => {
  try {
    const db = await initDB();
    return new Promise((resolve, reject) => {
      const transaction = db.transaction(storeName, "readwrite");
      const store = transaction.objectStore(storeName);
      const request = store.delete(key);
      
      request.onerror = (event) => reject(event.target.error);
      request.onsuccess = (event) => resolve();
    });
  } catch (error) {
    console.error("Error clearing video from IndexedDB:", error);
    throw error;
  }
};

// Check if a key exists
export const hasVideo = async (key) => {
  try {
    const result = await getVideo(key);
    return !!result;
  } catch (error) {
    return false;
  }
};