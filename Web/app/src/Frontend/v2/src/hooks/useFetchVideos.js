import { useState, useEffect, useCallback, useRef } from "react";
import { getAllVideosApi } from "../apis/Video";

export const useFetchVideos = () => {
  const [videos, setVideos] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const initialFetchDone = useRef(false);
  const fetchTimeoutRef = useRef(null);
  const abortControllerRef = useRef(null);
  const videosRef = useRef([]);  // Add this ref to store videos without causing re-renders

  const fetchVideos = useCallback(async (forceRefresh = false) => {
    // Skip if already fetched and not forcing refresh
    if (initialFetchDone.current && !forceRefresh) {
      return { videos: videosRef.current };
    }
    
    // Cancel any in-progress fetch
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
    
    // Clear any existing timeout
    if (fetchTimeoutRef.current) {
      clearTimeout(fetchTimeoutRef.current);
    }
    
    // Create new abort controller
    abortControllerRef.current = new AbortController();
    
    setLoading(true);
    setError(null);
    
    // Set a timeout to prevent getting stuck loading
    fetchTimeoutRef.current = setTimeout(() => {
      console.warn("Fetch videos timed out after 30 seconds");
      setLoading(false);
      setError("Request took too long. The server might be busy.");
      abortControllerRef.current?.abort("timeout");
    }, 30000);
    
    try {
      console.log("Fetching videos from API", forceRefresh ? "(forced refresh)" : "");
      const fetchedVideos = await getAllVideosApi(abortControllerRef.current.signal);
      
      // Clean up the timeout since we got a response
      clearTimeout(fetchTimeoutRef.current);
      
      if (!fetchedVideos || !Array.isArray(fetchedVideos)) {
        console.error("Invalid response format:", fetchedVideos);
        setVideos([]);
        videosRef.current = [];  // Update the ref too
        setError("Received invalid data format from server");
      } else {
        console.log(`Fetched ${fetchedVideos.length} videos successfully`);
        setVideos(fetchedVideos);
        videosRef.current = fetchedVideos;  // Update the ref too
        setError(null);
      }
      
      initialFetchDone.current = true;
      return { videos: fetchedVideos };
    } catch (err) {
      // Clean up the timeout since we got an error
      clearTimeout(fetchTimeoutRef.current);
      
      // Don't set error state for aborted requests
      if (err.name !== 'AbortError') {
        console.error("Error fetching videos:", err);
        setError(`Connection error: ${err.message}`);
      } else {
        console.log("Fetch videos request was aborted");
      }
      
      return { videos: videosRef.current };  // Return from ref instead
    } finally {
      fetchTimeoutRef.current = null;
      setLoading(false);
    }
  }, []);  // Remove videos from dependency array

  useEffect(() => {
    console.log("Initial useFetchVideos effect running");
    fetchVideos();
    
    return () => {
      console.log("Cleaning up useFetchVideos effect");
      if (fetchTimeoutRef.current) {
        clearTimeout(fetchTimeoutRef.current);
      }
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
      }
    };
  }, [fetchVideos]);

  const refetch = useCallback((forceRefresh = true) => {
    console.log("Manual refresh of videos requested");
    return fetchVideos(forceRefresh);
  }, [fetchVideos]);

  return { 
    videos, 
    loading, 
    error, 
    refetch,
    hasVideos: videos.length > 0 
  };
};