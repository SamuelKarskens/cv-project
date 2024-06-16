import verovio
from lxml import etree
import cairosvg
from PIL import Image
import subprocess

# Initialize the Verovio toolkit
vrvToolkit = verovio.toolkit()

# Load the music file
# musicFilePath = './dataset_generation/extract_verovio_durations/example.mei'  # or '.mei'
musicFilePath = 'example_2.mei'  # or '.mei'

vrvToolkit.loadData(open(musicFilePath, 'r').read())

# Extract all notes
mei_data = vrvToolkit.getMEI(True)

mei_xml = etree.fromstring(mei_data.encode('utf-8'))

namespace_map = {'mei': mei_xml.nsmap[None]}  # Adjust the prefix 'mei' as needed

# Use the namespace in your XPath
notes = mei_xml.xpath('//mei:note', namespaces=namespace_map)

print(notes)

note_count = 1
for note in notes:
    duration = note.get('dur', 'nodur')  # Get the duration attribute
    pname = note.get('pname', 'nopitch')  # Get the pitch name, default to 'nopitch' if not found
    oct = note.get('oct', '')  # Get the octave, default to empty string if not found
    pitch = f"{pname}{oct}" if pname != 'nopitch' else "rest"  # Construct the pitch string

    new_mei = etree.Element('mei', nsmap=mei_xml.nsmap)
    mei_head = etree.SubElement(new_mei, 'meiHead')
    file_desc = etree.SubElement(mei_head, 'fileDesc')
    title_stmt = etree.SubElement(file_desc, 'titleStmt')
    title = etree.SubElement(title_stmt, 'title')
    title.text = "Single Note Extraction"
    music = etree.SubElement(new_mei, 'music')
    body = etree.SubElement(music, 'body')
    mdiv = etree.SubElement(body, 'mdiv')
    score = etree.SubElement(mdiv, 'score')
    score_def = etree.SubElement(score, 'scoreDef')
    staff_grp = etree.SubElement(score_def, 'staffGrp')
    staff_def = etree.SubElement(staff_grp, 'staffDef', n="1", lines="5")
    section = etree.SubElement(score, 'section')
    measure = etree.SubElement(section, 'measure')
    staff = etree.SubElement(measure, 'staff', n="1")
    layer = etree.SubElement(staff, 'layer', n=str(note_count))
    layer.append(etree.fromstring(etree.tostring(note)))

    new_mei_str = etree.tostring(new_mei, pretty_print=True, xml_declaration=True, encoding='UTF-8').decode('utf-8')

    vrvToolkit.loadData(new_mei_str)
    svg_output = vrvToolkit.renderToSVG()

    filename = f'note_{pitch}_{duration}_{note_count}.svg'  # Include pitch and duration in the filename
    filename_png = f'note_{pitch}_{duration}_{note_count}.png'

    with open(filename, 'w') as f:
        f.write(svg_output)
    
#     # Convert SVG to PNG
#     cairosvg.svg2png(url=filename, write_to=filename_png)

#     # Open the PNG and crop it
#     img = Image.open(filename_png)
#     # Define the box to crop to (left, upper, right, lower)
#     box = (50, 50, 250, 300)  # Adjust these values as needed for your specific crop
#     cropped_img = img.crop(box)
#     cropped_img.save(filename_png)
    
#     note_count += 1

# print(f"Processed {note_count-1} notes.")
    # Convert SVG to PNG using Inkscape
    subprocess.run(['/Applications/Inkscape.app/Contents/MacOS/inkscape', filename, '--export-filename', filename_png], check=True)

    # Open the PNG and crop it
    img = Image.open(filename_png)
    # Define the box to crop to (left, upper, right, lower)
    # Adjust these values as needed for your specific crop
    box = (50, 50, 120, 300)  # Enlarged to capture more area
    cropped_img = img.crop(box)
    cropped_img.save(filename_png)
    
    note_count += 1

print(f"Processed {note_count-1} notes.")