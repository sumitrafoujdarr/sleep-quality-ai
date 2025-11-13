from moviepy.editor import VideoFileClip

# Load your 2-second MP4 video
clip = VideoFileClip("background.mp4")  # Make sure background.mp4 is in the same folder

# Loop for 5 seconds (optional, if your video is short)
clip = clip.loop(duration=5)

# Export as GIF
clip.write_gif("background.gif")
