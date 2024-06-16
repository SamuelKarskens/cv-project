import verovio
import subprocess
from PIL import Image
import random

# Initialize the Verovio toolkit
toolkit = verovio.toolkit()

# Music encoding template for a single note
mei_template = '''
<?xml version="1.0" encoding="UTF-8"?>
<mei xmlns="http://www.music-encoding.org/ns/mei">
    <meiHead>
        <fileDesc>
            <titleStmt>
                <title>{note} Note</title>
            </titleStmt>
            <pubStmt></pubStmt>
        </fileDesc>
        <encodingDesc>
            <appInfo>
                <application isodate="2024-06-16" version="1.0">
                    <name>Python Verovio Script</name>
                    <p>Generates SVG for musical notes.</p>
                </application>
            </appInfo>
        </encodingDesc>
    </meiHead>
    <music>
        <body>
            <mdiv>
                <score>
                    <scoreDef>
                        <staffGrp>
                            <staffDef n="1" lines="5" clef.shape="G" clef.line="2"/>
                        </staffGrp>
                    </scoreDef>
                    <section>
                        <measure n="1">
                            <staff n="1">
                                <layer n="1">
                                    <note xml:id="{note_id}" pname="{pname}" oct="{oct}" dur="{dur}" stem.dir="up"/>
                                </layer>
                            </staff>
                        </measure>
                    </section>
                </score>
            </mdiv>
        </body>
    </music>
</mei>
'''

# Define notes and octaves range
full_notes = ['C', 'D', 'E', 'F', 'G', 'A', 'B']
start_note, start_octave = 'D', 3
end_note, end_octave = 'G', 6

# Collect all notes from D3 to G6
notes = []
for oct in range(start_octave, end_octave + 1):
    for note in full_notes:
        if oct == start_octave and full_notes.index(note) < full_notes.index(start_note):
            continue
        if oct == end_octave and full_notes.index(note) > full_notes.index(end_note):
            break
        notes.append(f"{note}{oct}")

### 
# Define notes and octaves range
full_notes = ['C', 'D', 'E', 'F', 'G', 'A', 'B']
start_note, start_octave = 'D', 3
end_note, end_octave = 'G', 6
durations = [1, 2, 4, 8, 16]  # Whole, half, quarter, eighth, and sixteenth notes

# Collect all notes from D3 to G6
notes = []
for oct in range(start_octave, end_octave + 1):
    for note in full_notes:
        if oct == start_octave and full_notes.index(note) < full_notes.index(start_note):
            continue
        if oct == end_octave and full_notes.index(note) > full_notes.index(end_note):
            break
        notes.append(f"{note}{oct}")

# Generate SVGs for each note and each duration
for note in notes:
    pname = note[:-1].lower()  # Extract the pitch name
    oct = note[-1]             # Extract the octave
    for dur in durations:
        note_id = f"{pname}{oct}"
        mei_data = mei_template.format(note=note_id, note_id=note_id, pname=pname, oct=oct, dur=dur)
        toolkit.loadData(mei_data)
        svg_data = toolkit.renderToSVG()
        with open(f"data_images_verovio/{note_id}_{dur}.svg", "w") as file:
            file.write(svg_data)

        filename_png = f"data_images_verovio/{note_id}_{dur}.png"
        subprocess.run(['/Applications/Inkscape.app/Contents/MacOS/inkscape', f"data_images_verovio/{note_id}_{dur}.svg", '--export-filename', filename_png], check=True)
        img = Image.open(filename_png)
        # Define the box to crop to (left, upper, right, lower)
        # Adjust these values as needed for your specific crop
        left_bound = random.randint(110,120)
        left_upper_bound_addition = random.randint(0,10)
        if dur == 16:
            box = (left_bound , 60+left_upper_bound_addition, 180, 350)  # Enlarged to capture more area
        else:
            box = (left_bound, 100+left_upper_bound_addition, 180, 350)
        cropped_img = img.crop(box)
        # cropped_img.save(filename_png)
        # Create a new white background image

        background = Image.new('RGB', cropped_img.size, (255, 255, 255))

        # Paste the cropped image onto the white background
        background.paste(cropped_img, (0, 0), cropped_img)

        # Save the new image with a white background
        background.save(filename_png)

        # print(f"Generated {note_id}.svg")
        print(f"Generated {note_id}_{dur}.svg")

print("All SVG files have been generated.")
