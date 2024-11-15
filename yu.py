from pytube import YouTube

#YouTube('https://youtu.be/kTJczUoc26U').streams.first().download()
YouTube('https://youtu.be/kTJczUoc26U').streams.get_highest_resolution().download()
