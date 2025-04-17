/**
 * Synchronizes playback between a main video and a secondary video
 * @param {HTMLVideoElement} mainVideo - The main video element that controls the playback
 * @param {HTMLVideoElement} secondaryVideo - The secondary video to be synchronized
 * @param {Object} options - Configuration options
 * @param {boolean} options.keepSecondaryMuted - Whether to always keep the secondary video muted
 * @returns {Function} - Cleanup function to remove all event listeners
 */
export const synchronizeVideos = (
    mainVideo,
    secondaryVideo,
    { keepSecondaryMuted = true } = {}
  ) => {
    if (!mainVideo || !secondaryVideo) {
      console.error("Both video elements are required for synchronization");
      return () => {};
    }
  
    // Logging để debug
    console.log("Setting up video synchronization");
    console.log("Main video readyState:", mainVideo.readyState);
    console.log("Secondary video readyState:", secondaryVideo.readyState);
  
    // Reference để theo dõi xem sự thay đổi có được kích hoạt bởi đồng bộ hay không (để tránh vòng lặp)
    let syncing = false;
  
    // Xử lý sự kiện play
    const handlePlay = () => {
      if (syncing) return;
      syncing = true;
      console.log("[SYNC] Main video played, syncing secondary");
  
      try {
        // Đảm bảo cả hai video đều ở cùng thời gian
        secondaryVideo.currentTime = mainVideo.currentTime;
        
        // Play secondary video
        const playPromise = secondaryVideo.play();
        
        if (playPromise !== undefined) {
          playPromise.catch((error) => {
            console.error("[SYNC] Error playing secondary video:", error);
            
            // Một số trình duyệt yêu cầu tương tác người dùng trước khi chạy video
            // Thêm một nút ẩn và tự động click để giải quyết vấn đề này
            const handleUserInteraction = () => {
              secondaryVideo.play().catch(e => {
                console.error("[SYNC] Second attempt to play failed:", e);
              });
            };
            
            // Đăng ký sự kiện click toàn cục
            document.addEventListener('click', handleUserInteraction, { once: true });
            
            // Cố gắng phát lại sau một khoảng thời gian ngắn
            setTimeout(() => {
              if (secondaryVideo.paused && !mainVideo.paused) {
                secondaryVideo.play().catch(() => {});
              }
            }, 1000);
          });
        }
      } catch (err) {
        console.error("[SYNC] Error in play handler:", err);
      } finally {
        syncing = false;
      }
    };
  
    // Xử lý sự kiện pause
    const handlePause = () => {
      if (syncing) return;
      syncing = true;
      console.log("[SYNC] Main video paused, syncing secondary");
  
      try {
        secondaryVideo.pause();
      } catch (err) {
        console.error("[SYNC] Error in pause handler:", err);
      } finally {
        syncing = false;
      }
    };
  
    // Xử lý sự kiện seeking (đang tìm kiếm)
    const handleSeeking = () => {
      if (syncing) return;
      syncing = true;
      console.log("[SYNC] Main video seeking to:", mainVideo.currentTime);
  
      try {
        // Đồng bộ thời gian của secondary video
        secondaryVideo.currentTime = mainVideo.currentTime;
      } catch (err) {
        console.error("[SYNC] Error in seeking handler:", err);
      } finally {
        syncing = false;
      }
    };
  
    // Xử lý sự kiện seeked (đã tìm kiếm xong)
    const handleSeeked = () => {
      if (syncing) return;
      syncing = true;
      console.log("[SYNC] Main video seeked to:", mainVideo.currentTime);
  
      try {
        // Đảm bảo thời gian được đồng bộ sau khi tìm kiếm hoàn tất
        secondaryVideo.currentTime = mainVideo.currentTime;
        
        // Nếu main video đang chạy, đảm bảo secondary video cũng chạy
        if (!mainVideo.paused && secondaryVideo.paused) {
          secondaryVideo.play().catch(err => {
            console.error("[SYNC] Error playing after seek:", err);
          });
        }
      } catch (err) {
        console.error("[SYNC] Error in seeked handler:", err);
      } finally {
        syncing = false;
      }
    };
  
    // Theo dõi thời gian đồng bộ cuối cùng để giảm tần suất
    let lastSyncTime = 0;
    const MIN_SYNC_INTERVAL = 1000; // 1 giây giữa các lần đồng bộ
    
    // Xử lý sự kiện timeupdate (để giữ các video đồng bộ nếu chúng bị trôi)
    const handleTimeUpdate = () => {
      if (syncing) return;
  
      // Chỉ đồng bộ định kỳ để giảm tác động hiệu suất
      const now = Date.now();
      if (now - lastSyncTime > MIN_SYNC_INTERVAL) {
        
        // Nếu chênh lệch thời gian lớn hơn 0.5 giây, đồng bộ chúng
        const timeDiff = Math.abs(mainVideo.currentTime - secondaryVideo.currentTime);
        if (timeDiff > 0.5) {
          syncing = true;
          lastSyncTime = now;
          console.log(`[SYNC] Time drift detected (${timeDiff.toFixed(2)}s), syncing to:`, mainVideo.currentTime);
          
          try {
            secondaryVideo.currentTime = mainVideo.currentTime;
          } catch (err) {
            console.error("[SYNC] Error in time update handler:", err);
          } finally {
            syncing = false;
          }
        }
      }
    };
  
    // Xử lý sự kiện ratechange (đồng bộ tốc độ phát)
    const handleRateChange = () => {
      if (syncing) return;
      syncing = true;
      console.log("[SYNC] Main video rate changed to:", mainVideo.playbackRate);
  
      try {
        secondaryVideo.playbackRate = mainVideo.playbackRate;
      } catch (err) {
        console.error("[SYNC] Error in rate change handler:", err);
      } finally {
        syncing = false;
      }
    };
  
    // Xử lý sự kiện volumechange (giữ cho secondary video luôn tắt tiếng nếu được chỉ định)
    const handleVolumeChange = () => {
      if (syncing || !keepSecondaryMuted) return;
      
      try {
        // Luôn giữ cho secondary video bị tắt tiếng nếu được chỉ định
        if (!secondaryVideo.muted && keepSecondaryMuted) {
          syncing = true;
          secondaryVideo.muted = true;
          syncing = false;
        }
      } catch (err) {
        console.error("[SYNC] Error in volume change handler:", err);
        syncing = false;
      }
    };
  
    // Thiết lập trạng thái ban đầu khi các video được tải
    const initializeSyncState = () => {
      console.log("[SYNC] Initializing sync state");
      
      try {
        // Đồng bộ hóa trạng thái
        if (keepSecondaryMuted) {
          secondaryVideo.muted = true;
        }
        
        secondaryVideo.playbackRate = mainVideo.playbackRate;
        secondaryVideo.currentTime = mainVideo.currentTime;
        
        // Nếu main video đang chạy, bắt đầu chạy secondary video
        if (!mainVideo.paused) {
          secondaryVideo.play().catch((error) => {
            console.error("[SYNC] Error starting secondary video during init:", error);
          });
        }
        
        console.log("[SYNC] Initial state synchronized");
      } catch (err) {
        console.error("[SYNC] Error initializing sync state:", err);
      }
    };
  
    // Thêm tất cả các sự kiện vào main video
    mainVideo.addEventListener("play", handlePlay);
    mainVideo.addEventListener("pause", handlePause);
    mainVideo.addEventListener("seeking", handleSeeking);
    mainVideo.addEventListener("seeked", handleSeeked);
    mainVideo.addEventListener("timeupdate", handleTimeUpdate);
    mainVideo.addEventListener("ratechange", handleRateChange);
    mainVideo.addEventListener("volumechange", handleVolumeChange);
  
    // Thêm sự kiện cho secondary video để duy trì trạng thái tắt tiếng
    if (keepSecondaryMuted) {
      secondaryVideo.addEventListener("volumechange", () => {
        if (!secondaryVideo.muted) {
          secondaryVideo.muted = true;
        }
      });
    }
  
    // Thử khởi tạo đồng bộ ngay lập tức
    if (mainVideo.readyState >= 2 && secondaryVideo.readyState >= 2) { // HAVE_CURRENT_DATA hoặc tốt hơn
      initializeSyncState();
    } else {
      // Chờ cho cả hai video sẵn sàng
      const mainVideoReady = new Promise(resolve => {
        if (mainVideo.readyState >= 2) {
          resolve();
        } else {
          mainVideo.addEventListener("canplay", resolve, { once: true });
        }
      });
      
      const secondaryVideoReady = new Promise(resolve => {
        if (secondaryVideo.readyState >= 2) {
          resolve();
        } else {
          secondaryVideo.addEventListener("canplay", resolve, { once: true });
        }
      });
      
      // Khi cả hai video đều sẵn sàng
      Promise.all([mainVideoReady, secondaryVideoReady])
        .then(initializeSyncState)
        .catch(err => console.error("[SYNC] Error waiting for videos to be ready:", err));
    }
  
    // Trả về hàm dọn dẹp
    return () => {
      console.log("[SYNC] Cleaning up video synchronization");
      
      mainVideo.removeEventListener("play", handlePlay);
      mainVideo.removeEventListener("pause", handlePause);
      mainVideo.removeEventListener("seeking", handleSeeking);
      mainVideo.removeEventListener("seeked", handleSeeked);
      mainVideo.removeEventListener("timeupdate", handleTimeUpdate);
      mainVideo.removeEventListener("ratechange", handleRateChange);
      mainVideo.removeEventListener("volumechange", handleVolumeChange);
      
      if (keepSecondaryMuted) {
        secondaryVideo.removeEventListener("volumechange", () => {
          if (!secondaryVideo.muted) secondaryVideo.muted = true;
        });
      }
    };
  };