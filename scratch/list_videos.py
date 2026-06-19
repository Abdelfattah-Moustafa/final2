import os

video_dir = 'c:/Users/abody/Downloads/sign-language-pipeline/data/signs_videos'
files = os.listdir(video_dir)
print(f"Total videos: {len(files)}")
print("First 100 video files:")
for f in sorted(files)[:100]:
    print(f)
