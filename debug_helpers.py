def format_duration(start, end):
    duration = end - start
    hours, remainder = divmod(duration.seconds, 3600)
    minutes, seconds = divmod(remainder, 60) 
    duration_formatted = '%02d:%02d:%02d' % (hours, minutes, seconds)
    return duration_formatted
