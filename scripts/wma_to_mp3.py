import os
import subprocess
import time


# Format estimated finish time into HH:MM:SS
def format_time(seconds):
    total_hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{int(total_hours):02}:{int(minutes):02}:{int(secs):02}"


os.chdir("/lustre/projects/Research_Project-T116269/")

# separate unconverted files
processed_files = [f[:-4] for f in os.listdir("cobalt-audio-mp3")]
unprocessed_files = [
    f for f in os.listdir("cobalt-audio-wma") if f[:-4] not in processed_files
]
total_processed_files = len(processed_files)
total_unprocessed_files = len(unprocessed_files)

wmas = 1
total_time = 60
for file in unprocessed_files:

    # Calculate ETA using rolling average time
    average_time = total_time / wmas
    eta_seconds = average_time * (total_unprocessed_files - wmas)
    eta_formatted = format_time(eta_seconds)
    print(f"Converting {file} to mp3...")

    start = time.time()
    subprocess.run(
        [
            "ffmpeg",
            "-loglevel",
            "panic",
            "-y",
            "-i",
            f"cobalt-audio-wma/{file}",
            "-q:a",
            "0",
            f"cobalt-audio-mp3/{file[:-4]}.mp3",
        ]
    )
    end = time.time() - start

    # Append timing info to logfile
    with open("wma_to_mp3_log.txt", "a", encoding="utf-8") as f:
        f.write(
            "\n"
            + f"{wmas + total_processed_files}\t{file}\t{end:.2f}s\t{(((wmas + total_processed_files)/(total_unprocessed_files + total_processed_files)) * 100):.2f}%\t{eta_formatted}"
        )

    print(f"Converted {file} to mp3 in {end:.2f}s")
    wmas += 1
    total_time += end


print(f"Successfully converted {wmas} .wma files to mp3")
