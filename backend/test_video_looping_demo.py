#!/usr/bin/env python3
"""Test to demonstrate video looping functionality."""

import os
import subprocess

# Check environment settings
print("=== Video Generation Settings ===")
print(f"USE_SHORT_VIDEOS: {os.getenv('USE_SHORT_VIDEOS', 'not set')}")
print(f"SHORT_VIDEO_DURATION: {os.getenv('SHORT_VIDEO_DURATION', 'not set')}")
print(f"ENABLE_VIDEO_LOOPING: {os.getenv('ENABLE_VIDEO_LOOPING', 'not set')}")
print()

# Check latest video file
video_dir = "../storage/videos"
if os.path.exists(video_dir):
    videos = sorted([f for f in os.listdir(video_dir) if f.endswith('.mp4')], 
                   key=lambda x: os.path.getmtime(os.path.join(video_dir, x)), 
                   reverse=True)
    
    if videos:
        latest_video = os.path.join(video_dir, videos[0])
        latest_audio = latest_video.replace("_video.mp4", "_audio.wav").replace("/videos/", "/audio/")
        
        print(f"=== Latest Video Analysis ===")
        print(f"Video: {latest_video}")
        print(f"Audio: {latest_audio}")
        print()
        
        # Get video info
        video_cmd = ["ffprobe", "-v", "quiet", "-show_entries", "format=duration", "-of", "csv=p=0", latest_video]
        video_duration = subprocess.run(video_cmd, capture_output=True, text=True).stdout.strip()
        
        # Get audio info  
        if os.path.exists(latest_audio):
            audio_cmd = ["ffprobe", "-v", "quiet", "-show_entries", "format=duration", "-of", "csv=p=0", latest_audio]
            audio_duration = subprocess.run(audio_cmd, capture_output=True, text=True).stdout.strip()
        else:
            audio_duration = "N/A"
            
        print(f"Video Duration: {video_duration}s")
        print(f"Audio Duration: {audio_duration}s")
        
        if video_duration and audio_duration != "N/A":
            vid_dur = float(video_duration)
            aud_dur = float(audio_duration)
            print(f"Duration Match: {'✅ YES' if abs(vid_dur - aud_dur) < 0.5 else '❌ NO'}")
            print(f"Difference: {abs(vid_dur - aud_dur):.3f}s")
            
            # If the video was supposed to be 5 seconds but is longer, it was looped
            short_duration = int(os.getenv('SHORT_VIDEO_DURATION', '5'))
            if vid_dur > short_duration * 2:
                print(f"\n✅ Video was looped! Original would be ~{short_duration}s, final is {vid_dur:.1f}s")
                print(f"   Estimated loops: {int(vid_dur / short_duration)}")
    else:
        print("No video files found in storage/videos")