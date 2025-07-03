import os
import re 


os.chdir("/lustre/projects/Research_Project-T116269/")

processed_files = [f[:-4] for f in os.listdir("cobalt-text-txt")]
unprocessed_files = [f for f in os.listdir("cobalt-text-srt") if f[:-4] not in processed_files]


for file in unprocessed_files:

    with open(f"cobalt-text-srt/{file}", "r", encoding="utf-8") as f:
        content = f.read()

    regex = r'Speaker.*'
    dialogue = re.findall(regex, content)

    last_speaker = dialogue[0][:9]
    transcript = dialogue[0]

    for line in dialogue[1:]:
        speaker = line[:9]
        if speaker != last_speaker:
            transcript += "\n\n" + line
            last_speaker = speaker
        else:
            transcript += " " + line[10:]
        
    with open(f"cobalt-text-converted/{file[:-4]}.txt", "w", encoding="utf-8") as f:
        f.write(transcript)

    print(f"Successfully wrote {file} to txt")